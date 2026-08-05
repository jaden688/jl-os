/* ============================================================================
 * JL OS :: context switch
 * ============================================================================
 * void switch_context(uint32_t *save_esp_here, uint32_t load_esp);
 *
 * The whole of a kernel thread's context is its stack pointer. Everything else
 * either lives on that stack or does not need preserving:
 *
 *   - EAX/ECX/EDX are caller-saved under the i386 System V ABI, so the compiler
 *     has already spilled anything live across this call.
 *   - EBP/EBX/ESI/EDI are callee-saved and are pushed here.
 *   - EFLAGS is pushed because IF must survive the switch. A task that yields
 *     with interrupts enabled has to be resumed with them enabled, or the
 *     timer stops and the scheduler never runs again.
 *   - Segment registers are identical for every kernel task (flat GDT), and CR3
 *     is identical because the address space is a single identity map.
 *
 * When this is called from the timer IRQ, the interrupt frame is already on the
 * outgoing task's stack. It stays there untouched and is popped by that task's
 * own iret when it is resumed, which is why no special interrupt-aware path is
 * needed here.
 * ============================================================================ */

.section .text
.globl switch_context
.type switch_context, @function
switch_context:
    /* Read both arguments BEFORE pushing anything -- the pushes below move
     * ESP and would invalidate these offsets. */
    movl    4(%esp), %eax       /* uint32_t *save_esp_here */
    movl    8(%esp), %edx       /* uint32_t  load_esp      */

    pushl   %ebp
    pushl   %ebx
    pushl   %esi
    pushl   %edi
    pushfl

    movl    %esp, (%eax)        /* outgoing task remembers where it stopped */
    movl    %edx, %esp          /* incoming task's stack becomes current */

    /* From here on we are executing on a different task's stack. The pops
     * below unwind whatever that task pushed when IT was switched away --
     * or, for a task that has never run, the frame task_spawn built by hand. */
    popfl
    popl    %edi
    popl    %esi
    popl    %ebx
    popl    %ebp
    ret                         /* into the resumed task's next instruction */

.size switch_context, . - switch_context
