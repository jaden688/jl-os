/* ============================================================================
 * JL/OS  --  COM1 serial driver and panic (serial.h)
 * ============================================================================
 * Independent boot evidence. VGA output proves nothing when the display path
 * itself is what failed: a blank screen is indistinguishable from a kernel
 * that never ran. Serial goes out a different piece of hardware, so a message
 * arriving on COM1 is proof the CPU reached that line of code.
 *
 * QEMU forwards COM1 to the terminal with `-serial stdio`. On real hardware it
 * needs a physical port or a USB adapter on the target.
 *
 * This is direct port I/O only. No libc, no buffering, no interrupts.
 * ============================================================================ */

#ifndef JL_SERIAL_H
#define JL_SERIAL_H

#include "jl_core.h"

#define COM1_PORT 0x3F8

/* Configure COM1 for 115200 8N1. Returns 0 on success, non-zero if the UART
 * loopback self-test fails (no chip present, or it is not responding). */
int serial_init(void);

void serial_putc(char c);
void serial_write(const char *s);

/* Unsigned decimal and zero-padded 32-bit hex. No printf, no varargs. */
void serial_dec(uint32_t v);
void serial_hex32(uint32_t v);

/* Boot stage marker: "[boot] <stage>\n". Called at each milestone so a hang
 * can be localised to the gap between two known-good points. */
void serial_stage(const char *stage);

/* A log LINE is usually several serial_write/serial_dec calls, and the timer
 * IRQ writes to the same port. Without a lock, "[pewter] 31 MiB free" comes out
 * with "[timer] tick 100" spliced into the middle of it -- which looks like a
 * cosmetic annoyance and is actually two writers sharing a device with no
 * mutual exclusion. Wrap a whole line:
 *
 *     uint32_t f = serial_lock();
 *     serial_write("[pewter] "); serial_dec(n); serial_write(" MiB free\n");
 *     serial_unlock(f);
 *
 * Saves and RESTORES the interrupt flag rather than unconditionally re-enabling,
 * so calling this from inside an IRQ handler does not silently turn interrupts
 * back on partway through one. */
uint32_t serial_lock(void);
void serial_unlock(uint32_t flags);

/* Disable interrupts, report the reason on serial and VGA, halt forever.
 * Never returns. */
void panic(const char *reason) __attribute__((noreturn));

#endif
