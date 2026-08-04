/* ============================================================================
 * JL/OS  --  freestanding kernel entry (kmain.c)
 * ============================================================================
 * No C runtime, no libc, no host headers. Reached from boot.s after the stack
 * exists and .bss has been cleared.
 *
 * Boot evidence is emitted to COM1 as well as VGA. Serial is the authoritative
 * channel: a blank screen cannot distinguish "display init failed" from "the
 * kernel never ran", whereas a serial line proves the CPU executed the line
 * that wrote it.
 * ============================================================================ */

#include "jl_core.h"
#include "serial.h"
#include "idt.h"
#include "pic_pit.h"

/* Passed in EAX by any Multiboot 1 compliant loader. Distinct from the header
 * magic 0x1BADB002 that GRUB searched for in our image. */
#define MULTIBOOT_BOOTLOADER_MAGIC 0x2BADB002u

/* Provided by linker.ld. */
extern char kernel_start[], kernel_end[], bss_start[], bss_end[];

void jl_hud_main(void);

/* ---------------------------------------------------------------- vga text -- */

static uint16_t *const HARDWARE_VGA_BUFFER = (uint16_t *)0xB8000;
static size_t terminal_row = 0;
static size_t terminal_column = 0;

static inline uint8_t make_vga_color(VGAColor fg, VGAColor bg) {
    return (uint8_t)(fg | (bg << 4));
}

static inline uint16_t make_vga_entry(unsigned char uc, uint8_t color) {
    return (uint16_t)uc | ((uint16_t)color << 8);
}

void k_vga_freestanding_init(void) {
    uint8_t color = make_vga_color(VGA_LIGHT_CYAN, VGA_BLACK);
    terminal_row = 0;
    terminal_column = 0;
    for (size_t y = 0; y < VGA_HEIGHT; y++)
        for (size_t x = 0; x < VGA_WIDTH; x++)
            HARDWARE_VGA_BUFFER[y * VGA_WIDTH + x] = make_vga_entry(' ', color);
}

void k_vga_freestanding_puts(const char *data, uint8_t color) {
    for (size_t i = 0; data[i] != '\0'; i++) {
        char c = data[i];
        if (c == '\n') {
            terminal_column = 0;
            if (++terminal_row == VGA_HEIGHT) terminal_row = 0;
            continue;
        }
        HARDWARE_VGA_BUFFER[terminal_row * VGA_WIDTH + terminal_column] =
            make_vga_entry((unsigned char)c, color);
        if (++terminal_column == VGA_WIDTH) {
            terminal_column = 0;
            if (++terminal_row == VGA_HEIGHT) terminal_row = 0;
        }
    }
}

/* Mirror a line to both channels so the transcript matches what is on screen. */
static void boot_line(const char *text, uint8_t color) {
    k_vga_freestanding_puts(text, color);
    serial_write(text);
}

/* ------------------------------------------------------------------- kmain -- */

void kmain(uint32_t magic, const void *mb_info) __attribute__((noreturn));

void kmain(uint32_t magic, const void *mb_info) {
    /* Serial before anything else: if the next step faults, the last line on
     * the wire localises the failure. */
    int serial_failed = serial_init();
    serial_write("\n");
    serial_stage("C kernel entered (kmain)");

    if (serial_failed)
        k_vga_freestanding_puts("[WARN] COM1 absent; VGA only.\n", 0x0E);

    /* Validate the loader contract before trusting anything it handed us. A
     * wrong magic means we were not started by a Multiboot loader and the
     * info pointer is meaningless. */
    serial_write("[boot] multiboot magic = ");
    serial_hex32(magic);
    serial_write(" expected ");
    serial_hex32(MULTIBOOT_BOOTLOADER_MAGIC);
    serial_write("\n");

    if (magic != MULTIBOOT_BOOTLOADER_MAGIC)
        panic("bad multiboot magic in EAX -- not loaded by a Multiboot loader");

    serial_stage("multiboot magic validated");

    serial_write("[boot] multiboot_info at ");
    serial_hex32((uint32_t)mb_info);
    serial_write("\n");
    if (mb_info == NULL)
        panic("multiboot_info pointer is NULL");

    /* .bss was cleared in _start; report the range that was zeroed so the
     * claim is checkable against the linker map rather than assumed. */
    serial_write("[boot] bss ");
    serial_hex32((uint32_t)bss_start);
    serial_write(" .. ");
    serial_hex32((uint32_t)bss_end);
    serial_write(" cleared by _start\n");

    serial_write("[boot] kernel image ");
    serial_hex32((uint32_t)kernel_start);
    serial_write(" .. ");
    serial_hex32((uint32_t)kernel_end);
    serial_write(" (");
    serial_dec((uint32_t)(kernel_end - kernel_start));
    serial_write(" bytes)\n");

    /* Initialize IDT gate table and exception handlers */
    idt_init();
    serial_stage("IDT gate table initialized");

    /* Remap 8259 PIC and start 100 Hz PIT hardware timer interrupts */
    pic_pit_init();
    serial_stage("8259 PIC and PIT timer active (100 Hz IRQ0)");

    k_vga_freestanding_init();

    uint8_t header_col = make_vga_color(VGA_BLACK, VGA_LIGHT_CYAN);
    uint8_t cyan_col   = make_vga_color(VGA_LIGHT_CYAN, VGA_BLACK);
    uint8_t green_col  = make_vga_color(VGA_LIGHT_GREEN, VGA_BLACK);
    uint8_t yellow_col = make_vga_color(VGA_YELLOW, VGA_BLACK);

    boot_line("================================================================\n", cyan_col);
    boot_line("  JL/OS  --  freestanding x86 kernel  (own bootloader, own ISO)\n", header_col);
    boot_line("================================================================\n", cyan_col);
    boot_line("[hw]   protected mode, 32-bit, IDT active, PIT timer (100 Hz)\n", green_col);
    boot_line("[hw]   VGA text buffer mapped at 0xB8000\n", green_col);
    boot_line("[hw]   COM1 at 0x3F8, 115200 8N1\n", green_col);
    boot_line("[kern] no Linux, no libc, no POSIX, no host runtime\n", yellow_col);

    /* JlOsState holds the MK-1/2/3 descriptors. Scheduling and token selection
     * inside it are SIMULATED -- see ARCHITECTURE.md. NHIT arithmetic is real. */
    static JlOsState state;
    jl_os_init(&state);
    serial_stage("JL runtime initialized (JlOsState)");

    boot_line("[jl]   MK-1 tensor/NHIT ring engine    : online\n", green_col);
    boot_line("[jl]   MK-2 scheduler descriptors      : online (simulated)\n", green_col);
    boot_line("[jl]   MK-3 IPC ring bus               : online\n", green_col);

    serial_stage("entering HUD (owns display and PS/2 from here)");
    boot_line("[hud]  handing display to native HUD\n", yellow_col);

    jl_hud_main();

    /* jl_hud_main does not return. Reaching here is a control-flow bug. */
    panic("HUD returned to kmain");
}
