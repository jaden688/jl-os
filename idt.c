/* ============================================================================
 * JL/OS  --  Interrupt Descriptor Table (IDT) Implementation (idt.c)
 * ============================================================================
 * Zero stdio/libc freestanding IDT driver with diagnostic serial reporting.
 * ============================================================================ */

#include "idt.h"
#include "serial.h"

static struct idt_entry idt[IDT_ENTRIES];
static struct idt_ptr   idtp;

extern void idt_load(uint32_t ptr);
extern void isr0(void);
extern void isr6(void);
extern void isr13(void);
extern void isr14(void);
extern void isr_default(void);

static const char *EXCEPTION_MESSAGES[32] = {
    "Divide Error (#DE)",
    "Debug (#DB)",
    "NMI Interrupt",
    "Breakpoint (#BP)",
    "Overflow (#OF)",
    "BOUND Range Exceeded (#BR)",
    "Invalid Opcode (#UD)",
    "Device Not Available (#NM)",
    "Double Fault (#DF)",
    "Coprocessor Segment Overrun",
    "Invalid TSS (#TS)",
    "Segment Not Present (#NP)",
    "Stack-Segment Fault (#SS)",
    "General Protection Fault (#GP)",
    "Page Fault (#PF)",
    "Reserved",
    "x87 FPU Floating-Point Error (#MF)",
    "Alignment Check (#AC)",
    "Machine Check (#MC)",
    "SIMD Floating-Point Exception (#XM)",
    "Virtualization Exception (#VE)",
    "Control Protection Exception (#CP)",
    "Reserved", "Reserved", "Reserved", "Reserved",
    "Reserved", "Reserved", "Reserved", "Reserved",
    "Security Exception (#SX)",
    "Reserved"
};

void idt_set_gate(uint8_t num, uint32_t base, uint16_t selector, uint8_t flags) {
    idt[num].base_low  = (uint16_t)(base & 0xFFFF);
    idt[num].base_high = (uint16_t)((base >> 16) & 0xFFFF);
    idt[num].selector  = selector;
    idt[num].zero      = 0;
    idt[num].flags     = flags;
}

void idt_init(void) {
    idtp.limit = (sizeof(struct idt_entry) * IDT_ENTRIES) - 1;
    idtp.base  = (uint32_t)&idt;

    /* Set default stub for all 256 gates */
    for (int i = 0; i < IDT_ENTRIES; i++) {
        idt_set_gate((uint8_t)i, (uint32_t)isr_default, 0x08, 0x8E);
    }

    /* Register specific exception handlers */
    idt_set_gate(0,  (uint32_t)isr0,  0x08, 0x8E);  /* #DE Divide Error */
    idt_set_gate(6,  (uint32_t)isr6,  0x08, 0x8E);  /* #UD Invalid Opcode */
    idt_set_gate(13, (uint32_t)isr13, 0x08, 0x8E);  /* #GP General Protection */
    idt_set_gate(14, (uint32_t)isr14, 0x08, 0x8E);  /* #PF Page Fault */

    /* Load IDTR register */
    idt_load((uint32_t)&idtp);
    serial_write("[idt]   IDT initialized with 256 gates loaded into IDTR\n");
}

/* Helper to read CR2 page fault linear address */
static inline uint32_t read_cr2(void) {
    uint32_t val;
    __asm__ __volatile__("mov %%cr2, %0" : "=r"(val));
    return val;
}

void idt_exception_handler(struct interrupt_frame *frame) {
    serial_write("\n");
    serial_write("================================================================\n");
    serial_write("                  FATAL CPU EXCEPTION OCCURRED                  \n");
    serial_write("================================================================\n");
    serial_write("[fault] Vector: ");
    serial_dec(frame->int_no);
    serial_write(" (");
    if (frame->int_no < 32) {
        serial_write(EXCEPTION_MESSAGES[frame->int_no]);
    } else {
        serial_write("Unknown Exception");
    }
    serial_write(")\n");

    serial_write("[fault] Error Code: ");
    serial_hex32(frame->err_code);
    serial_write("\n");

    serial_write("[fault] EIP: ");
    serial_hex32(frame->eip);
    serial_write("  CS: ");
    serial_hex32(frame->cs);
    serial_write("  EFLAGS: ");
    serial_hex32(frame->eflags);
    serial_write("\n");

    if (frame->int_no == 14) { /* Page Fault */
        uint32_t cr2 = read_cr2();
        serial_write("[fault] CR2 (Linear Address): ");
        serial_hex32(cr2);
        serial_write("\n");
    }

    serial_write("[fault] EAX: "); serial_hex32(frame->eax);
    serial_write(" EBX: "); serial_hex32(frame->ebx);
    serial_write(" ECX: "); serial_hex32(frame->ecx);
    serial_write(" EDX: "); serial_hex32(frame->edx);
    serial_write("\n");

    serial_write("[fault] ESP: "); serial_hex32(frame->esp);
    serial_write(" EBP: "); serial_hex32(frame->ebp);
    serial_write(" ESI: "); serial_hex32(frame->esi);
    serial_write(" EDI: "); serial_hex32(frame->edi);
    serial_write("\n");

    serial_write("================================================================\n");
    serial_write("[fault] System halted safely.\n");

    /* Safe halt loop */
    while (1) {
        __asm__ __volatile__("cli; hlt");
    }
}
