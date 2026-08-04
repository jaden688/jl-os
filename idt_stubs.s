/* ============================================================================
 * JL/OS  --  IDT Assembly Stubs (idt_stubs.s)
 * ============================================================================
 * Low-level interrupt entry points, stack frame setup, and IDTR loader.
 * ============================================================================ */

.section .text

.globl idt_load
idt_load:
    movl 4(%esp), %eax
    lidt (%eax)
    ret

/* Common ISR stub wrapper */
common_isr_stub:
    pusha                   /* Pushes edi, esi, ebp, esp, ebx, edx, ecx, eax */
    pushl %ds
    pushl %es
    pushl %fs
    pushl %gs

    movw $0x10, %ax         /* Load kernel data segment descriptor */
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %fs
    movw %ax, %ax           /* %gs */

    pushl %esp              /* Pass pointer to interrupt_frame to C handler */
    call idt_exception_handler
    addl $4, %esp           /* Clean up frame argument */

    popl %gs
    popl %fs
    popl %es
    popl %ds
    popa
    addl $8, %esp           /* Clean up error code and vector number */
    iret

/* Exception Stubs */
.globl isr0
isr0:
    pushl $0                /* Dummy error code */
    pushl $0                /* Vector 0: #DE Divide Error */
    jmp common_isr_stub

.globl isr6
isr6:
    pushl $0                /* Dummy error code */
    pushl $6                /* Vector 6: #UD Invalid Opcode */
    jmp common_isr_stub

.globl isr13
isr13:
    /* #GP pushes its own error code */
    pushl $13               /* Vector 13: #GP General Protection */
    jmp common_isr_stub

.globl isr14
isr14:
    /* #PF pushes its own error code */
    pushl $14               /* Vector 14: #PF Page Fault */
    jmp common_isr_stub

.globl isr_default
isr_default:
    pushl $0                /* Dummy error code */
    pushl $255              /* Default handler */
    jmp common_isr_stub

/* IRQ0 Timer Handler Stub (Vector 32) */
.globl irq0_timer_stub
irq0_timer_stub:
    pusha
    pushl %ds
    pushl %es
    pushl %fs
    pushl %gs

    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %fs
    movw %ax, %gs

    call irq0_timer_c_handler

    popl %gs
    popl %fs
    popl %es
    popl %ds
    popa
    iret
