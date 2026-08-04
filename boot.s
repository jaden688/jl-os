/* ============================================================================
 * JL/OS  --  Multiboot 1 assembly entry (boot.s)
 * ============================================================================
 * This is MULTIBOOT 1, not Multiboot2. Magic 0x1BADB002 is the Multiboot 1
 * header magic; Multiboot2 uses 0xE85250D6 and an entirely different tag
 * layout. GRUB loads this with the `multiboot` command, not `multiboot2`.
 * The two specifications are not mixed anywhere in this repository.
 *
 * State on entry, guaranteed by the Multiboot 1 specification:
 *   EAX = 0x2BADB002   bootloader magic -- proof a Multiboot-compliant loader
 *                      ran us. This is NOT the same constant as the header
 *                      magic 0x1BADB002.
 *   EBX = physical address of the multiboot_info structure
 *   CS  = 32-bit read/execute code segment, base 0
 *   DS/ES/FS/GS/SS = 32-bit read/write data segment, base 0
 *   A20 gate enabled, interrupts disabled, paging disabled
 *   ESP  = UNDEFINED   -- the kernel must establish its own stack
 *   .bss = NOT zeroed  -- the kernel must clear it
 * ============================================================================ */

.set ALIGN,    1<<0             /* align loaded modules on page boundaries */
.set MEMINFO,  1<<1             /* ask the loader for a memory map */
.set FLAGS,    ALIGN | MEMINFO
.set MAGIC,    0x1BADB002       /* Multiboot 1 header magic */
.set CHECKSUM, -(MAGIC + FLAGS) /* magic + flags + checksum must sum to zero */

/* Kept by linker.ld via KEEP(); must land in the first 8 KiB, 4-byte aligned. */
.section .multiboot, "a"
.align 4
.long MAGIC
.long FLAGS
.long CHECKSUM

/* 16 KiB kernel stack, 16-byte aligned for the i386 System V ABI. This lives
 * in .bss, so it is inside the region cleared below -- which is safe only
 * because nothing is pushed until after the clear completes. */
.section .bss, "aw", @nobits
.align 16
.globl stack_bottom
.globl stack_top
stack_bottom:
.skip 16384
stack_top:

.section .text
.globl _start
.type _start, @function
_start:
    /* ESP is undefined on entry: no memory access through the stack is safe
     * until this instruction retires. */
    movl $stack_top, %esp

    /* String operations must count upward. The loader's DF state is not
     * specified by the spec, and a set DF would make `rep stosl` walk
     * backwards out of .bss and corrupt memory below it. */
    cld

    /* Park the Multiboot values in registers that `rep stosl` does not touch.
     * stosl consumes EDI, ECX and EAX, so EBP and ESI are the safe choices. */
    movl %eax, %ebp          /* bootloader magic, expected 0x2BADB002 */
    movl %ebx, %esi          /* multiboot_info physical address */

    /* Zero .bss. Multiboot makes no promise about its contents, but C code
     * assumes static storage begins at zero. Bounds come from linker.ld and
     * are 4 KiB aligned, so a dword count is exact. */
    movl $bss_start, %edi
    movl $bss_end, %ecx
    subl %edi, %ecx
    shrl $2, %ecx
    xorl %eax, %eax
    rep stosl

    /* Hand both values to C. i386 SysV pushes arguments right to left and
     * requires ESP to be 16-byte aligned at the CALL instruction.
     * ESP starts 16-aligned at stack_top; 8 bytes of padding plus two 4-byte
     * pushes consumes exactly 16, so the boundary is preserved. */
    subl $8, %esp
    pushl %esi               /* arg 2: const void *mb_info */
    pushl %ebp               /* arg 1: uint32_t magic       */

    call kmain

    /* kmain is declared noreturn. If it ever returns, stop deterministically
     * rather than execute whatever bytes follow. Interrupts stay masked --
     * there is no IDT, so any interrupt would triple-fault. */
    cli
1:  hlt
    jmp 1b

.size _start, . - _start
