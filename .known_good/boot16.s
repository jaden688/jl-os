/* ============================================================================
 * JL/OS  --  stage-1 bootloader (boot16.s)   NO GRUB, NO LINUX
 * ============================================================================
 * Loaded by BIOS from an El Torito "no emulation" boot catalog entry at
 * physical 0x7C00, in 16-bit real mode, with DL = the BIOS drive number of the
 * CD we came from.
 *
 * Responsibilities, in order:
 *   1. establish a real-mode stack and save the boot drive
 *   2. read the flat kernel image off the disc with INT 13h extended read
 *   3. enable the A20 line so addresses above 1 MiB are reachable
 *   4. load a flat GDT and switch the CPU to 32-bit protected mode
 *   5. relocate the kernel from the low buffer to 1 MiB
 *   6. enter the kernel with the Multiboot 1 loader contract satisfied
 *
 * On the last point: this bootloader deliberately presents the SAME interface
 * GRUB would. EAX = 0x2BADB002 and EBX = a minimal multiboot_info structure,
 * so kmain's validation is unchanged and the kernel cannot tell the difference.
 * We replaced the loader, not the contract.
 *
 * KERNEL_LBA and KERNEL_SECTORS are patched by mkiso.py once it knows where
 * the kernel landed inside the ISO image.
 * ============================================================================ */

.code16
.section .text
.globl _boot_start

/* Physical layout used below:
 *   0x0600  our own real-mode stack top
 *   0x7C00  this bootloader (placed by BIOS)
 *   0x10000 low staging buffer for the kernel image (64 KiB mark)
 *   0x100000 final kernel location (1 MiB), where linker.ld links it
 */
.set STAGE_BUF,   0x10000
.set KERNEL_DEST, 0x100000
.set KERNEL_ENTRY,0x101000      /* _start; .multiboot occupies 0x100000 */

_boot_start:
    cli
    xorw    %ax, %ax
    movw    %ax, %ds
    movw    %ax, %es
    movw    %ax, %ss
    movw    $0x7C00, %sp        /* stack grows down from just below us */
    cld

    movb    %dl, boot_drive     /* BIOS gave us the drive we booted from */

    movw    $msg_stage1, %si
    call    print_string

    /* ---- read the kernel image from the disc -------------------------- */
    /* INT 13h AH=42h takes a Disk Address Packet rather than CHS, which is
     * the only sane option for a CD: 2048-byte sectors and LBA addressing. */
    movw    $dap, %si
    movb    $0x42, %ah
    movb    boot_drive, %dl
    int     $0x13
    jc      disk_error

    movw    $msg_loaded, %si
    call    print_string

    /* ---- A20 ---------------------------------------------------------- */
    /* Without A20 the 21st address line is forced low and every access to
     * 1 MiB wraps to 0. The kernel lives at 1 MiB, so this is mandatory. */
    call    enable_a20

    /* ---- protected mode ------------------------------------------------ */
    lgdt    gdt_descriptor
    movl    %cr0, %eax
    orl     $0x1, %eax          /* CR0.PE */
    movl    %eax, %cr0

    /* Far jump reloads CS with the 32-bit code selector and flushes the
     * prefetch queue; the CPU is only truly in 32-bit mode after this. */
    ljmp    $0x08, $protected_mode

disk_error:
    movw    $msg_disk_err, %si
    call    print_string
1:  hlt
    jmp     1b

/* ------------------------------------------------------------ real mode -- */

/* BIOS teletype output. Only available before protected mode. */
print_string:
    pushw   %ax
1:  lodsb
    testb   %al, %al
    jz      2f
    movb    $0x0E, %ah
    movw    $0x0007, %bx
    int     $0x10
    jmp     1b
2:  popw    %ax
    ret

enable_a20:
    /* Fast A20 via the PS/2 system control port. Adequate on everything that
     * still has a BIOS; the keyboard-controller dance is the legacy path. */
    inb     $0x92, %al
    testb   $0x02, %al
    jnz     1f
    orb     $0x02, %al
    andb    $0xFE, %al          /* never set bit 0 -- that is a fast reset */
    outb    %al, $0x92
1:  ret

/* --------------------------------------------------------- protected mode -- */

.code32
protected_mode:
    movw    $0x10, %ax          /* flat 32-bit data selector */
    movw    %ax, %ds
    movw    %ax, %es
    movw    %ax, %fs
    movw    %ax, %gs
    movw    %ax, %ss
    movl    $0x90000, %esp      /* 32-bit stack below the kernel */

    /* ---- relocate the kernel to 1 MiB --------------------------------- */
    /* Real mode could not write above 1 MiB, so the image was staged low.
     * Now that segmentation is flat, copy it to where it was linked. */
    movl    $STAGE_BUF, %esi
    movl    $KERNEL_DEST, %edi
    movl    $kernel_dwords, %ecx
    movl    (%ecx), %ecx
    rep movsl

    /* ---- Multiboot 1 loader contract ---------------------------------- */
    /* Present exactly what GRUB would, so the kernel is loader-agnostic. */
    movl    $mb_info, %ebx
    movl    $0x2BADB002, %eax
    ljmp    $0x08, $KERNEL_ENTRY

/* ------------------------------------------------------------------ data -- */

.align 4
/* Disk Address Packet for INT 13h AH=42h. mkiso.py patches the LBA and count. */
dap:
    .byte   0x10                /* packet size */
    .byte   0                   /* reserved */
kernel_sectors:
    .word   0x0001              /* PATCHED: 2048-byte sectors to read */
    .word   0x0000              /* buffer offset  (0x10000 = 0x1000:0x0000) */
    .word   0x1000              /* buffer segment */
kernel_lba:
    .quad   0x0000000000000000  /* PATCHED: LBA of the kernel inside the ISO */

kernel_dwords:
    .long   0x00000400          /* PATCHED: image size in dwords for rep movsl */

boot_drive:
    .byte   0

/* Minimal multiboot_info. flags=0 means "no fields valid", which is legal and
 * honest: we do not yet supply a memory map. The kernel only requires the
 * pointer to be non-NULL. */
.align 8
mb_info:
    .long   0                   /* flags */
    .space  84, 0               /* remainder of the structure, zeroed */

/* Flat GDT: identity-mapped 4 GiB code and data. No paging, no segmentation
 * tricks -- protected mode simply so we can address memory above 1 MiB. */
.align 8
gdt_start:
    .quad   0x0000000000000000  /* null descriptor */
    .quad   0x00CF9A000000FFFF  /* 0x08: code, base 0, limit 4G, r/x, 32-bit */
    .quad   0x00CF92000000FFFF  /* 0x10: data, base 0, limit 4G, r/w, 32-bit */
gdt_end:

gdt_descriptor:
    .word   gdt_end - gdt_start - 1
    .long   gdt_start

msg_stage1:   .asciz "JL/OS stage1: reading kernel\r\n"
msg_loaded:   .asciz "JL/OS stage1: kernel loaded, entering protected mode\r\n"
msg_disk_err: .asciz "JL/OS stage1: INT 13h read failed\r\n"

/* El Torito no-emulation loads whole 2048-byte sectors. Pad so the image is
 * exactly one sector and the boot catalog's sector count is honest. */
.org 2046
.word 0xAA55
