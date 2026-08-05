/* ============================================================================
 * JL/OS  --  COM1 serial driver and panic (serial.c)
 * ============================================================================
 * 16550-compatible UART programmed through direct port I/O. Polled, not
 * interrupt driven: there is no IDT yet, and boot diagnostics must work before
 * interrupts are safe to enable.
 * ============================================================================ */

#include "serial.h"

/* ------------------------------------------------------------------ ports -- */

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ __volatile__("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t v;
    __asm__ __volatile__("inb %1, %0" : "=a"(v) : "Nd"(port));
    return v;
}

/* 16550 register offsets from the base port. With DLAB set in the Line
 * Control Register, offsets 0 and 1 become the baud divisor latch instead of
 * data and interrupt-enable. */
#define REG_DATA        0   /* RBR / THR, or divisor low  when DLAB=1 */
#define REG_INT_ENABLE  1   /* IER,       or divisor high when DLAB=1 */
#define REG_FIFO_CTRL   2   /* FCR */
#define REG_LINE_CTRL   3   /* LCR */
#define REG_MODEM_CTRL  4   /* MCR */
#define REG_LINE_STATUS 5   /* LSR */

#define LSR_THR_EMPTY   0x20  /* transmit holding register ready */

static int serial_ready = 0;

int serial_init(void) {
    outb(COM1_PORT + REG_INT_ENABLE, 0x00);  /* mask all UART interrupts */
    outb(COM1_PORT + REG_LINE_CTRL, 0x80);   /* DLAB on: expose divisor latch */
    /* 115200 baud = 115200 / divisor, so divisor 1. */
    outb(COM1_PORT + REG_DATA, 0x01);
    outb(COM1_PORT + REG_INT_ENABLE, 0x00);
    outb(COM1_PORT + REG_LINE_CTRL, 0x03);   /* DLAB off, 8 bits, no parity, 1 stop */
    outb(COM1_PORT + REG_FIFO_CTRL, 0xC7);   /* enable + clear FIFOs, 14-byte trigger */
    outb(COM1_PORT + REG_MODEM_CTRL, 0x1E);  /* loopback on, RTS/DSR set */

    /* Self-test: in loopback the byte written must come straight back. If it
     * does not, no UART is present and every later write would spin forever. */
    outb(COM1_PORT + REG_DATA, 0xAE);
    if (inb(COM1_PORT + REG_DATA) != 0xAE) {
        serial_ready = 0;
        return 1;
    }

    outb(COM1_PORT + REG_MODEM_CTRL, 0x0F);  /* loopback off, normal operation */
    serial_ready = 1;
    return 0;
}

void serial_putc(char c) {
    if (!serial_ready) return;
    /* Bounded spin. An unbounded wait on absent hardware hangs the boot with
     * no diagnostic, which is the exact failure this driver exists to avoid. */
    for (uint32_t spin = 0; spin < 100000u; spin++) {
        if (inb(COM1_PORT + REG_LINE_STATUS) & LSR_THR_EMPTY) break;
    }
    if (c == '\n') outb(COM1_PORT + REG_DATA, '\r');  /* terminals expect CRLF */
    outb(COM1_PORT + REG_DATA, (uint8_t)c);
}

void serial_write(const char *s) {
    while (*s) serial_putc(*s++);
}

void serial_dec(uint32_t v) {
    char tmp[12];
    int n = 0;
    if (v == 0) { serial_putc('0'); return; }
    while (v && n < 11) { tmp[n++] = (char)('0' + (v % 10u)); v /= 10u; }
    while (n--) serial_putc(tmp[n]);
}

void serial_hex32(uint32_t v) {
    static const char digits[] = "0123456789ABCDEF";
    serial_write("0x");
    for (int shift = 28; shift >= 0; shift -= 4)
        serial_putc(digits[(v >> shift) & 0xFu]);
}

void serial_stage(const char *stage) {
    serial_write("[boot] ");
    serial_write(stage);
    serial_putc('\n');
}

/* ------------------------------------------------------------------ panic -- */

/* Written straight to the VGA text buffer rather than through the HUD, because
 * panic must work even when the HUD is the thing that failed. */
static void panic_vga(const char *reason) {
    volatile uint16_t *vga = (volatile uint16_t *)0xB8000;
    const uint8_t attr = 0x4F;  /* white on red */
    const char *prefix = "JL/OS PANIC: ";
    size_t i = 0;

    for (size_t k = 0; k < 80u * 25u; k++) vga[k] = ((uint16_t)attr << 8) | ' ';
    while (prefix[i] && i < 80u) {
        vga[i] = ((uint16_t)attr << 8) | (uint8_t)prefix[i];
        i++;
    }
    for (size_t j = 0; reason[j] && i < 80u * 25u; j++, i++)
        vga[i] = ((uint16_t)attr << 8) | (uint8_t)reason[j];
}

void panic(const char *reason) {
    __asm__ __volatile__("cli");
    serial_write("\n*** JL/OS PANIC: ");
    serial_write(reason);
    serial_write(" ***\n");
    panic_vga(reason);
    for (;;) __asm__ __volatile__("hlt");
}

uint32_t serial_lock(void) {
    uint32_t flags;
    /* pushfl/cli captures the CURRENT interrupt state before masking, so
     * serial_unlock can put it back exactly as it was. */
    __asm__ __volatile__("pushfl; popl %0; cli" : "=r"(flags) :: "memory");
    return flags;
}

void serial_unlock(uint32_t flags) {
    __asm__ __volatile__("pushl %0; popfl" :: "r"(flags) : "memory", "cc");
}
