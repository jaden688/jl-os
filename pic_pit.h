/* ============================================================================
 * JL/OS  --  8259 PIC & PIT Hardware Timer Header (pic_pit.h)
 * ============================================================================ */

#ifndef JL_PIC_PIT_H
#define JL_PIC_PIT_H

#include "jl_core.h"

/* PIC 8259 Ports */
#define PIC1_COMMAND 0x20
#define PIC1_DATA    0x21
#define PIC2_COMMAND 0xA0
#define PIC2_DATA    0xA1

/* PIT 8254 Ports */
#define PIT_CHANNEL0 0x40
#define PIT_COMMAND  0x43

/* Base vector for IRQ0..7 */
#define PIC1_OFFSET  0x20
#define PIC2_OFFSET  0x28

void pic_pit_init(void);
uint64_t timer_get_ticks(void);
uint32_t timer_get_uptime_sec(void);

#endif /* JL_PIC_PIT_H */
