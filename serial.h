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

/* Disable interrupts, report the reason on serial and VGA, halt forever.
 * Never returns. */
void panic(const char *reason) __attribute__((noreturn));

#endif
