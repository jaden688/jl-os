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

/* E820 memory map, built here and handed to the kernel in Multiboot format.
 * Lives at 0x9000: above this bootloader (0x7C00 + 2 KiB) and below the kernel
 * staging buffer at 0x10000, so neither the disk read nor the relocation to
 * 1 MiB can touch it. 32 entries is far more than any real machine reports. */
.set MMAP_BUF,    0x9000
.set MMAP_MAX,    32
.set MMAP_END,    MMAP_BUF + MMAP_MAX * 24

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
     * the only sane option for a CD: 2048-byte sectors and LBA addressing.
     *
     * The read is CHUNKED, and it must be. The DAP's buffer is a real-mode
     * segment:offset pair, so a single transfer cannot cross a 64 KiB segment
     * boundary -- BIOSes fail the call outright rather than wrapping. A 20 KiB
     * kernel loaded in one call and hid this; a 196 KiB kernel does not.
     *
     * 16 sectors x 2048 = 32 KiB per read, comfortably inside one segment, and
     * the buffer segment advances by 0x800 paragraphs (32 KiB) each pass. */
    /* mkiso.py patches the TOTAL sector count into the DAP's count field. Read
     * it out before the loop, which then reuses that field as the per-chunk
     * count. Keeping the DAP layout byte-identical matters: mkiso.py locates it
     * by signature and patches at fixed sub-offsets, so adding a field here
     * would silently shift kernel_dwords and load the wrong size. */
    movw    dap_count, %cx          /* total 2048-byte sectors remaining */
read_loop:
    testw   %cx, %cx
    jz      read_done

    movw    $16, %ax                /* sectors this pass */
    cmpw    %ax, %cx
    jae     1f
    movw    %cx, %ax                /* final partial chunk */
1:  movw    %ax, dap_count

    pushw   %cx
    movw    $dap, %si
    movb    $0x42, %ah
    movb    boot_drive, %dl
    int     $0x13
    popw    %cx
    jc      disk_error

    /* Advance LBA by the sectors just read, and the destination segment by the
     * same number of 32-byte paragraphs (2048 bytes = 0x80 paragraphs). */
    movw    dap_count, %ax
    addw    %ax, kernel_lba
    adcw    $0, kernel_lba+2
    movw    %ax, %bx
    shlw    $7, %bx                 /* sectors * 0x80 paragraphs */
    addw    %bx, dap_segment
    subw    %ax, %cx
    jmp     read_loop

read_done:
    movw    $msg_loaded, %si
    call    print_string

    /* ---- memory map (INT 15h, EAX=E820) -------------------------------- */
    /* This MUST happen here, in real mode. Once the CPU is in protected mode
     * the BIOS is unreachable, and the kernel has no other way to learn what
     * physical memory exists. GRUB would have supplied this; we replaced GRUB,
     * so we supply it.
     *
     * Written directly in Multiboot 1 mmap layout -- a 4-byte size field
     * followed by the 20-byte E820 entry -- so the kernel parses one format
     * whether it was booted by us or by a Multiboot loader. */
    call    detect_memory
    movw    $msg_mmap, %si
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
dap_count:
    .word   0x0001              /* sectors for THIS transfer; set per chunk */
    .word   0x0000              /* buffer offset  (0x10000 = 0x1000:0x0000) */
dap_segment:
    .word   0x1000              /* buffer segment; advances each chunk */
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
/* Multiboot 1 information structure. Field offsets are fixed by the spec:
 * flags +0, mem_lower +4, mem_upper +8, mmap_length +44, mmap_addr +48.
 * detect_memory fills the mmap fields and sets flag bit 6. */
mb_info:
mb_flags:
    .long   0                   /* bit 0 = mem_*, bit 6 = mmap valid */
mb_mem_lower:
    .long   0                   /* KiB below 1 MiB */
mb_mem_upper:
    .long   0                   /* KiB above 1 MiB */
    .space  32, 0               /* boot_device, cmdline, mods, syms (+12..+43) */
mb_mmap_length:
    .long   0                   /* +44 */
mb_mmap_addr:
    .long   0                   /* +48 */
    .space  36, 0               /* remainder (+52..+87), zeroed */

/* ---- detect_memory ------------------------------------------------------
 * Builds the Multiboot memory map at MMAP_BUF using INT 15h AX=E820.
 *
 * The E820 protocol is stateful: EBX carries the continuation value and is
 * zero on the first call and again after the last entry. A BIOS may also
 * signal the end by setting CF instead of clearing EBX, so both are checked --
 * relying on only one is a known way to read one entry too many.
 * ------------------------------------------------------------------------- */
/* MUST be .code16. This routine sits below the .code32 directive that covers
 * the protected-mode entry, but it RUNS in real mode, before the switch.
 * Without this the assembler emits 32-bit encodings with operand-size prefixes
 * and the CPU decodes a completely different instruction stream -- `movw
 * $0x9000, %di` becomes a 32-bit load that swallows the next instruction as
 * immediate data. The result assembles and links cleanly and then hangs. */
.code16
detect_memory:
    pushal
    movw    $MMAP_BUF, %di
    xorl    %ebx, %ebx
    xorl    %ebp, %ebp          /* entry count */

e820_next:
    movl    $20, %es:(%di)      /* Multiboot size field precedes the entry */
    addw    $4, %di

    movl    $0x0000E820, %eax
    movl    $0x534D4150, %edx   /* 'SMAP' */
    movl    $20, %ecx
    int     $0x15
    jc      e820_done           /* CF on the first call means E820 unsupported */
    cmpl    $0x534D4150, %eax   /* BIOS echoes 'SMAP' on success */
    jne     e820_done

    /* A length of zero is a legal but useless entry; skip it rather than hand
     * the allocator a zero-sized region. */
    movl    %es:8(%di), %eax
    orl     %es:12(%di), %eax
    jz      e820_skip

    incl    %ebp
    addw    $20, %di
    jmp     e820_test

e820_skip:
    subw    $4, %di             /* reuse this slot for the next entry */

e820_test:
    testl   %ebx, %ebx          /* zero continuation = that was the last one */
    jz      e820_done
    cmpw    $MMAP_END, %di
    jb      e820_next

e820_done:
    /* mmap_length is the total byte count including the size fields. */
    movl    %ebp, %eax
    movl    $24, %edx
    mull    %edx
    movl    %eax, mb_mmap_length
    movl    $MMAP_BUF, %eax
    movl    %eax, mb_mmap_addr

    testl   %ebp, %ebp
    jz      1f
    orl     $0x40, mb_flags     /* bit 6: mmap_* are valid */
1:
    /* INT 12h reports conventional memory in KiB. Cheap, always present, and
     * it lets the kernel sanity-check the low end of the E820 map. */
    int     $0x12
    andl    $0xFFFF, %eax
    movl    %eax, mb_mem_lower
    orl     $0x01, mb_flags     /* bit 0: mem_lower/mem_upper valid */

    popal
    ret

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
msg_loaded:   .asciz "JL/OS stage1: kernel loaded\r\n"
msg_mmap:     .asciz "JL/OS stage1: E820 map built, entering protected mode\r\n"
msg_disk_err: .asciz "JL/OS stage1: INT 13h read failed\r\n"

/* El Torito no-emulation loads whole 2048-byte sectors. Pad so the image is
 * exactly one sector and the boot catalog's sector count is honest. */
.org 2046
.word 0xAA55
