/* ============================================================================
 * JL/OS  --  8259 PIC & PIT Hardware Timer Driver (pic_pit.c)
 * ============================================================================
 * Establishes legacy PIC hardware timer interrupts (IRQ0 -> Vector 32) at 100Hz.
 * ============================================================================ */

#include "pic_pit.h"
#include "idt.h"
#include "serial.h"

static volatile uint64_t g_timer_ticks = 0;

extern void irq0_timer_stub(void);

static inline uint8_t inb(uint16_t port) {
    uint8_t v;
    __asm__ __volatile__("inb %1, %0" : "=a"(v) : "Nd"(port));
    return v;
}

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ __volatile__("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline void io_wait(void) {
    outb(0x80, 0);
}

void pic_remap(uint8_t offset1, uint8_t offset2) {
    uint8_t a1 = inb(PIC1_DATA);
    uint8_t a2 = inb(PIC2_DATA);

    outb(PIC1_COMMAND, 0x11); io_wait(); /* ICW1: Init mode */
    outb(PIC2_COMMAND, 0x11); io_wait();

    outb(PIC1_DATA, offset1); io_wait(); /* ICW2: Master vector offset (0x20 / 32) */
    outb(PIC2_DATA, offset2); io_wait(); /* ICW2: Slave vector offset (0x28 / 40) */

    outb(PIC1_DATA, 0x04); io_wait();    /* ICW3: tell Master PIC that Slave is at IRQ2 */
    outb(PIC2_DATA, 0x02); io_wait();    /* ICW3: tell Slave PIC its cascade identity */

    outb(PIC1_DATA, 0x01); io_wait();    /* ICW4: 8086 mode */
    outb(PIC2_DATA, 0x01); io_wait();

    /* Restore masks */
    outb(PIC1_DATA, a1);
    outb(PIC2_DATA, a2);
}

void pit_init(uint32_t frequency_hz) {
    uint32_t divisor = 1193182 / frequency_hz;
    outb(PIT_COMMAND, 0x36); /* Channel 0, lobyte/hibyte, Mode 2 (Rate Generator) */
    outb(PIT_CHANNEL0, (uint8_t)(divisor & 0xFF));
    outb(PIT_CHANNEL0, (uint8_t)((divisor >> 8) & 0xFF));
}

void pic_pit_init(void) {
    /* 1. Remap 8259 PIC offsets */
    pic_remap(PIC1_OFFSET, PIC2_OFFSET);

    /* 2. Register IRQ0 handler stub at IDT vector 32 */
    idt_set_gate(PIC1_OFFSET, (uint32_t)irq0_timer_stub, 0x08, 0x8E);

    /* 3. Configure PIT to fire at 100 Hz (10 ms per tick) */
    pit_init(100);

    /* 4. Unmask IRQ0 (PIT Timer) on Master PIC, mask all other IRQs (0xFE = 11111110b).
     * Keeping IRQ1 (keyboard) masked on the PIC allows jl_hud.c to continue reading
     * port 0x60 via PS/2 polling without ISR interference. */
    outb(PIC1_DATA, 0xFE);
    outb(PIC2_DATA, 0xFF);

    serial_write("[pic]   8259 PIC remapped (Master=0x20, Slave=0x28)\n");
    serial_write("[pit]   PIT Channel 0 configured for 100 Hz interrupts\n");

    /* 5. Enable hardware interrupts on the CPU */
    __asm__ __volatile__("sti");
    serial_write("[hw]    Hardware interrupts unmasked (sti)\n");
}

void irq0_timer_c_handler(void) {
    g_timer_ticks++;

    /* Send End-of-Interrupt (EOI) to Master PIC */
    outb(PIC1_COMMAND, 0x20);

    /* Emit periodic serial proof every 100 ticks (1 second) */
    if (g_timer_ticks % 100 == 0) {
        serial_write("[timer] tick ");
        serial_dec((uint32_t)g_timer_ticks);
        serial_write(" (uptime: ");
        serial_dec((uint32_t)(g_timer_ticks / 100));
        serial_write("s)\n");
    }
}

uint64_t timer_get_ticks(void) {
    return g_timer_ticks;
}

uint32_t timer_get_uptime_sec(void) {
    return (uint32_t)(g_timer_ticks / 100);
}
