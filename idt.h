/* ============================================================================
 * JL/OS  --  Interrupt Descriptor Table (IDT) Header (idt.h)
 * ============================================================================
 * Freestanding 32-bit x86 IDT gate definitions and CPU exception structures.
 * ============================================================================ */

#ifndef JL_IDT_H
#define JL_IDT_H

#include "jl_core.h"

#define IDT_ENTRIES 256

/* IDT Entry Gate structure (8 bytes packed) */
struct idt_entry {
    uint16_t base_low;   /* Lower 16 bits of handler address */
    uint16_t selector;   /* Kernel code segment selector (0x08) */
    uint8_t  zero;       /* Always 0 */
    uint8_t  flags;      /* Gate type & attributes (0x8E = 32-bit Interrupt Gate) */
    uint16_t base_high;  /* Upper 16 bits of handler address */
} __attribute__((packed));

/* IDTR Register Pointer structure (6 bytes packed for LIDT) */
struct idt_ptr {
    uint16_t limit;      /* IDT size - 1 */
    uint32_t base;       /* Linear address of IDT array */
} __attribute__((packed));

/* Register frame pushed onto stack during CPU interrupt / exception */
struct interrupt_frame {
    uint32_t ds;
    uint32_t edi, esi, ebp, esp, ebx, edx, ecx, eax; /* Pushed by pusha */
    uint32_t int_no;
    uint32_t err_code;
    uint32_t eip, cs, eflags;                       /* Pushed automatically by CPU */
} __attribute__((packed));

void idt_init(void);
void idt_set_gate(uint8_t num, uint32_t base, uint16_t selector, uint8_t flags);
void idt_exception_handler(struct interrupt_frame *frame);

#endif /* JL_IDT_H */
