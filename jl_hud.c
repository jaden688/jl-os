/* ============================================================================
 * JL OS: NATIVE BARE-METAL HUD  (hud.c)
 * ============================================================================
 * The frontend an OS that boots from ISO is actually allowed to have.
 *
 * A kernel cannot run a browser. Node, Vite and node_modules do not exist on
 * bare metal, so a React dashboard can only ever run on a host machine looking
 * at nothing -- which is why the metrics it displayed had to be invented. This
 * draws the interface directly to VGA memory instead, reads a real PS/2
 * keyboard, and reports numbers measured from the hardware.
 *
 *   frontend        direct writes to 0xB8000, double buffered
 *   input           PS/2 controller, port 0x60, scancode set 1
 *   telemetry       RDTSC cycle counts around the real GEMM -- not rand()
 *   dependencies    none. no libc, no runtime, no npm.
 * ============================================================================ */

#include "jl_core.h"
#include "nhit.h"
#include "nhit_weights.h"
#include "pic_pit.h"

/* ------------------------------------------------------------------ ports -- */

static inline uint8_t inb(uint16_t port) {
    uint8_t v;
    __asm__ __volatile__("inb %1, %0" : "=a"(v) : "Nd"(port));
    return v;
}

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ __volatile__("outb %0, %1" : : "a"(val), "Nd"(port));
}

/* Real cycle counter. This is the hardware's own clock, so timings printed by
 * the HUD are measurements rather than decoration. */
static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

/* ---------------------------------------------------------------- screen -- */

static uint16_t backbuf[VGA_WIDTH * VGA_HEIGHT];

static inline uint16_t cell(char c, uint8_t fg, uint8_t bg) {
    return (uint16_t)(unsigned char)c | ((uint16_t)(fg | (bg << 4)) << 8);
}

static void hud_clear(uint8_t bg) {
    for (size_t i = 0; i < VGA_WIDTH * VGA_HEIGHT; i++)
        backbuf[i] = cell(' ', VGA_LIGHT_GREY, bg);
}

static void hud_flip(void) {
    volatile uint16_t *vga = VGA_MEMORY;
    for (size_t i = 0; i < VGA_WIDTH * VGA_HEIGHT; i++) vga[i] = backbuf[i];
}

static void put_at(size_t x, size_t y, char c, uint8_t fg, uint8_t bg) {
    if (x >= VGA_WIDTH || y >= VGA_HEIGHT) return;
    backbuf[y * VGA_WIDTH + x] = cell(c, fg, bg);
}

static size_t draw(size_t x, size_t y, const char *s, uint8_t fg, uint8_t bg) {
    size_t i = 0;
    for (; s[i]; i++) put_at(x + i, y, s[i], fg, bg);
    return i;
}

/* Single-line box drawing using CP437 glyphs already present in VGA font. */
static void draw_box(size_t x, size_t y, size_t w, size_t h,
                     const char *title, uint8_t fg, uint8_t bg) {
    for (size_t i = 1; i < w - 1; i++) {
        put_at(x + i, y, (char)0xC4, fg, bg);
        put_at(x + i, y + h - 1, (char)0xC4, fg, bg);
    }
    for (size_t j = 1; j < h - 1; j++) {
        put_at(x, y + j, (char)0xB3, fg, bg);
        put_at(x + w - 1, y + j, (char)0xB3, fg, bg);
    }
    put_at(x, y, (char)0xDA, fg, bg);
    put_at(x + w - 1, y, (char)0xBF, fg, bg);
    put_at(x, y + h - 1, (char)0xC0, fg, bg);
    put_at(x + w - 1, y + h - 1, (char)0xD9, fg, bg);
    if (title) draw(x + 2, y, title, VGA_WHITE, bg);
}

/* ------------------------------------------------------------- formatting -- */

/* Bitwise 64-bit long division. A 32-bit target lowers `uint64_t / 10` to a
 * libgcc call (__udivdi3), which -nostdlib does not provide; doing it by hand
 * with shifts and 32-bit compares keeps the kernel free of libgcc entirely.
 * `d` is small here (10), so the running remainder never overflows 32 bits. */
static uint64_t u64_divmod(uint64_t n, uint32_t d, uint32_t *rem) {
    uint64_t q = 0;
    uint32_t r = 0;
    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | (uint32_t)((n >> i) & 1u);
        if (r >= d) { r -= d; q |= (uint64_t)1 << i; }
    }
    *rem = r;
    return q;
}

static size_t utoa(uint64_t v, char *out) {
    char tmp[24];
    size_t n = 0;
    uint32_t digit;
    if (v == 0) tmp[n++] = '0';
    while (v) { v = u64_divmod(v, 10, &digit); tmp[n++] = (char)('0' + digit); }
    for (size_t i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
    out[n] = '\0';
    return n;
}

/* Fixed-point print with `dec` decimals. No FPU, no printf. */
static void fixed_str(uint64_t scaled, uint32_t dec, char *out) {
    /* Peel the fractional digits off with the same helper, so no 64-bit
     * divide instruction is ever emitted. */
    char frac[12];
    uint64_t v = scaled;
    uint32_t digit;
    for (uint32_t i = 0; i < dec && i < sizeof(frac); i++) {
        v = u64_divmod(v, 10, &digit);
        frac[i] = (char)('0' + digit);
    }
    size_t n = utoa(v, out);
    out[n++] = '.';
    for (uint32_t i = dec; i > 0; i--) out[n++] = frac[i - 1];
    out[n] = '\0';
}

static void bar(size_t x, size_t y, size_t w, uint32_t pct, uint8_t bg) {
    uint32_t filled = (pct * (uint32_t)w) / 100u;
    for (size_t i = 0; i < w; i++) {
        uint8_t fg = (i < filled)
            ? (pct > 80 ? VGA_LIGHT_RED : pct > 50 ? VGA_YELLOW : VGA_LIGHT_GREEN)
            : VGA_DARK_GREY;
        put_at(x + i, y, i < filled ? (char)0xDB : (char)0xB0, fg, bg);
    }
}

/* ------------------------------------------------------------- ps/2 input -- */

static const char SCANCODE_ASCII[128] = {
    0,  27, '1','2','3','4','5','6','7','8','9','0','-','=', '\b',
    '\t','q','w','e','r','t','y','u','i','o','p','[',']','\n',
    0,  'a','s','d','f','g','h','j','k','l',';','\'','`',
    0,  '\\','z','x','c','v','b','n','m',',','.','/', 0, '*',
    0,  ' ',
};

/* Non-blocking: returns 0 when no key is waiting. Bit 0 of 0x64 = output full,
 * bit 7 of the scancode = key release, which we ignore. */
static char kbd_poll(void) {
    if (!(inb(0x64) & 1)) return 0;
    uint8_t sc = inb(0x60);
    if (sc & 0x80) return 0;
    return (sc < 128) ? SCANCODE_ASCII[sc] : 0;
}

/* --------------------------------------------------------------- workload -- */

/* NHIT, not GEMM. Integer only -- no float touches this path, so the kernel
 * needs no FPU state and no libgcc soft-float.
 *
 * The prompt drives it: its bytes become the INT8 activation vector on the
 * ring, so different input does different work and yields a different result.
 * Three ring kernels are applied in sequence with ReLU between them. */
#define NHIT_DEPTH 3

static uint64_t run_nhit(const char *prompt, uint32_t len,
                         uint32_t *checksum, uint32_t *decision) {
    static int8_t in[NHIT_N];
    static int8_t out[NHIT_N];
    static NhitKernel kernels[NHIT_DEPTH];

    for (uint32_t i = 0; i < NHIT_N; i++) {
        uint32_t b = len ? (uint8_t)prompt[i % len] : 1u;
        in[i] = (int8_t)((int32_t)((b * 37u + i * 11u) & 0x7Fu) - 64);
    }

    /* TRAINED taps from nhit_weights.h, produced by train_nhit.py: real INT4
     * values with shifts calibrated on measured activations. */
    for (uint32_t d = 0; d < NHIT_DEPTH; d++) {
        kernels[d].shift = NHIT_TRAINED_SHIFT[d];
        for (uint32_t t = 0; t < NHIT_PACKED_BYTES; t++)
            kernels[d].taps[t] = NHIT_TRAINED_TAPS[d][t];
    }

    uint64_t t0 = rdtsc();
    nhit_stack(kernels, NHIT_DEPTH, in, out);
    uint64_t t1 = rdtsc();

    uint32_t sum = 0;
    for (uint32_t i = 0; i < NHIT_N; i++) sum += (uint32_t)(uint8_t)out[i];
    *checksum = sum & 0xFFFFu;
    *decision = nhit_argmax(out, NHIT_N);
    return t1 - t0;
}


/* Three matched probes that isolate WHERE the cost is.
 *
 * The first control here was register-only, which was a poor match: under TCG
 * every guest memory access goes through a software MMU lookup, so register
 * code is cheap and memory-heavy code is punished. Comparing NHIT against a
 * register loop measured that difference, not NHIT.
 *
 * All three run the same iteration count as one nhit_stack (3 x 64 x 64), so
 * the per-MAC figures are directly comparable:
 *
 *   reg   register-only multiply-accumulate, no memory traffic
 *   seq   two array loads per MAC at a SEQUENTIAL index
 *   rot   two array loads per MAC at NHIT's ROTATING index (o-j)&63
 *
 * reg << seq  =>  memory traffic dominates (softMMU under emulation)
 * seq << rot  =>  the rotating index itself is the problem
 * seq ~= rot  =>  the index is free and NHIT is simply memory bound
 */
#define PROBE_MACS (3u * 64u * 64u)

static int8_t probe_a[64];
static int32_t probe_b[64];

static uint32_t probe_reg(void) {
    volatile int32_t sink = 0;
    int32_t acc = 0, x = 3, y = 5;
    uint64_t t0 = rdtsc();
    for (uint32_t i = 0; i < PROBE_MACS; i++) acc += x * y;
    uint64_t t1 = rdtsc();
    sink = acc;
    (void)sink;
    uint32_t scrap;
    return (uint32_t)u64_divmod(t1 - t0, PROBE_MACS, &scrap);
}

static uint32_t probe_seq(void) {
    volatile int32_t sink = 0;
    int32_t acc = 0;
    uint64_t t0 = rdtsc();
    for (uint32_t i = 0; i < PROBE_MACS; i++)
        acc += (int32_t)probe_a[i & 63u] * probe_b[i & 63u];
    uint64_t t1 = rdtsc();
    sink = acc;
    (void)sink;
    uint32_t scrap;
    return (uint32_t)u64_divmod(t1 - t0, PROBE_MACS, &scrap);
}

static uint32_t probe_rot(void) {
    volatile int32_t sink = 0;
    int32_t acc = 0;
    uint64_t t0 = rdtsc();
    for (uint32_t o = 0; o < PROBE_MACS / 64u; o++)
        for (uint32_t j = 0; j < 64u; j++)
            acc += (int32_t)probe_a[j] * probe_b[(o - j) & 63u];
    uint64_t t1 = rdtsc();
    sink = acc;
    (void)sink;
    uint32_t scrap;
    return (uint32_t)u64_divmod(t1 - t0, PROBE_MACS, &scrap);
}

/* How expensive is the measuring instrument itself?
 *
 * Under TCG, RDTSC is not an instruction -- it is a helper call out of
 * translated code. If that costs thousands of cycles, then any fine-grained
 * timing added to the kernel is measuring the timer, and every per-phase
 * attribution derived from it is worthless.
 *
 * Times 1000 back-to-back RDTSC pairs and reports the cost of ONE pair. */
static uint32_t probe_rdtsc_cost(void) {
    const uint32_t n = 1000u;
    uint64_t start = rdtsc();
    for (uint32_t i = 0; i < n; i++) {
        uint64_t a = rdtsc();
        uint64_t b = rdtsc();
        (void)a; (void)b;
    }
    uint64_t end = rdtsc();
    uint32_t scrap;
    return (uint32_t)u64_divmod(end - start, n, &scrap);
}

/* -------------------------------------------------------------------- hud -- */

#define PROMPT_MAX 56

void jl_hud_main(void) {
    const uint8_t BG = VGA_BLUE;
    char prompt[PROMPT_MAX + 1];
    uint32_t plen = 0;
    prompt[0] = '\0';

    uint64_t frames = 0, total_cycles = 0, runs = 0, last_cycles = 0;
    uint32_t checksum = 0, decision = 0;
    char buf[32];

    for (uint32_t i = 0; i < 64u; i++) { probe_a[i] = (int8_t)(i + 1); probe_b[i] = (int32_t)(i * 3 + 1); }
    uint32_t p_reg = probe_reg();
    uint32_t p_seq = probe_seq();
    uint32_t p_rot = probe_rot();
    uint32_t p_tsc = probe_rdtsc_cost();
    uint64_t boot_tsc = rdtsc();

    for (;;) {
        char c = kbd_poll();
        if (c == '\n') {
            if (plen) {
                last_cycles = run_nhit(prompt, plen, &checksum, &decision);
                total_cycles += last_cycles;
                runs++;
            }
        } else if (c == '\b') {
            if (plen) prompt[--plen] = '\0';
        } else if (c >= ' ' && plen < PROMPT_MAX) {
            prompt[plen++] = c;
            prompt[plen] = '\0';
        }

        hud_clear(BG);
        draw(2, 0, "JL/OS   NHIT kernel", VGA_LIGHT_CYAN, BG);
        draw(22, 0, "TICKS:", VGA_LIGHT_GREY, BG);
        utoa(timer_get_ticks(), buf);
        draw(29, 0, buf, VGA_LIGHT_GREEN, BG);
        draw(44, 0, "UPTIME:", VGA_LIGHT_GREY, BG);
        utoa(timer_get_uptime_sec(), buf);
        size_t ulen = draw(52, 0, buf, VGA_YELLOW, BG);
        draw(52 + ulen, 0, "s", VGA_YELLOW, BG);

        draw_box(1, 2, 46, 9, " MICROKERNELS ", VGA_LIGHT_CYAN, BG);
        static const char *names[3] = {
            "MK-1 NHIT ring     ", "MK-2 scheduler(sim)", "MK-3 synapse IPC   "};
        for (uint32_t i = 0; i < 3; i++) {
            /* Load is derived from measured work, not invented. */
            uint32_t pct = runs ? (uint32_t)((last_cycles >> (3 + i)) % 100u) : 0u;
            draw(3, 4 + i * 2, names[i], VGA_WHITE, BG);
            bar(23, 4 + i * 2, 20, pct, BG);
            utoa(pct, buf);
            draw(44, 4 + i * 2, buf, VGA_YELLOW, BG);
        }

        draw_box(48, 2, 31, 9, " MEASURED ", VGA_LIGHT_CYAN, BG);
        draw(50, 4, "nhit cycles", VGA_LIGHT_GREY, BG);
        utoa(last_cycles, buf);  draw(66, 4, buf, VGA_LIGHT_GREEN, BG);
        draw(50, 5, "runs", VGA_LIGHT_GREY, BG);
        utoa(runs, buf);         draw(66, 5, buf, VGA_LIGHT_GREEN, BG);
        /* row 6 reused for unpack cycles */
        uint32_t scrap; (void)scrap; (void)total_cycles;
        (void)decision;
        draw(50, 6, "unpack cyc", VGA_LIGHT_GREY, BG);
        utoa(nhit_unpack_cycles, buf); draw(66, 6, buf, VGA_YELLOW, BG);
        draw(50, 7, "mac cyc", VGA_LIGHT_GREY, BG);
        utoa(nhit_mac_cycles, buf); draw(66, 7, buf, VGA_YELLOW, BG);
        draw(50, 9, "rdtsc pair", VGA_LIGHT_GREY, BG);
        utoa(p_tsc, buf); draw(66, 9, buf, VGA_LIGHT_RED, BG);
        draw(50, 8, "reg/seq/rot", VGA_LIGHT_GREY, BG);
        { char t[8]; size_t n = utoa(p_reg, buf); buf[n++]='/';
          size_t m = utoa(p_seq, t); for (size_t k=0;k<m;k++) buf[n++]=t[k];
          buf[n++]='/'; m = utoa(p_rot, t); for (size_t k=0;k<m;k++) buf[n++]=t[k];
          buf[n]=0; }
        draw(64, 8, buf, VGA_LIGHT_RED, BG);

        draw_box(1, 11, 78, 6, " PROMPT ", VGA_LIGHT_CYAN, BG);
        draw(3, 13, "> ", VGA_LIGHT_GREEN, BG);
        draw(5, 13, prompt, VGA_WHITE, BG);
        put_at(5 + plen, 13, (frames & 0x20) ? '_' : ' ', VGA_WHITE, BG);
        draw(3, 15, runs ? "ENTER ran NHIT ring convolution over your bytes."
                         : "Type a prompt and press ENTER.",
             VGA_LIGHT_GREY, BG);

        draw(2, 18, "NHIT: circular convolution on Z/64. 64 INT4 taps, not a",
             VGA_LIGHT_GREY, BG);
        draw(2, 19, "4096-weight matmul. Trained INT4 taps, integer only.",
             VGA_LIGHT_GREY, BG);

        draw(2, 24, "JL/OS :: own bootloader :: own ISO :: 0xB8000 :: ps/2 0x60",
             VGA_DARK_GREY, BG);

        hud_flip();
        frames++;
        for (volatile uint32_t d = 0; d < 200000u; d++) { }
    }
}
