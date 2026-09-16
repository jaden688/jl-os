/* ============================================================================
 * JL/OS  --  freestanding kernel link script
 * ============================================================================
 * Physical layout for a Multiboot 1 kernel loaded by GRUB at 1 MiB.
 *
 * 1 MiB is the conventional load address because everything below it is
 * fragmented by real-mode BIOS structures, the VGA aperture and option ROMs.
 *
 * The .multiboot section is placed FIRST and wrapped in KEEP(). GRUB scans
 * only the first 8 KiB of the file for the header; without KEEP the linker is
 * entitled to discard it under --gc-sections, and the image silently stops
 * being bootable while still linking successfully.
 * ============================================================================ */

ENTRY(_start)

SECTIONS
{
    . = 1M;

    kernel_start = .;

    /* Multiboot header must be within the first 8 KiB and 4-byte aligned. */
    .multiboot BLOCK(4K) : ALIGN(4K)
    {
        KEEP(*(.multiboot))
    }

    .text BLOCK(4K) : ALIGN(4K)
    {
        *(.text)
        *(.text.*)
    }

    .rodata BLOCK(4K) : ALIGN(4K)
    {
        *(.rodata)
        *(.rodata.*)
    }

    .data BLOCK(4K) : ALIGN(4K)
    {
        *(.data)
        *(.data.*)
    }

    /* Multiboot does not guarantee a zeroed .bss, so the kernel clears it
     * itself in _start using these bounds. */
    .bss BLOCK(4K) : ALIGN(4K)
    {
        bss_start = .;
        *(COMMON)
        *(.bss)
        *(.bss.*)
        bss_end = .;
    }

    kernel_end = .;

    /* Toolchain output with no business in a kernel image. */
    /DISCARD/ :
    {
        *(.comment)
        *(.note.*)
        *(.eh_frame)
    }
}
