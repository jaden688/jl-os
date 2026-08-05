/* ============================================================================
 * JL OS :: PMM  --  physical memory manager
 * ============================================================================
 * Rung 7. Everything above this -- paging, a kernel heap, task stacks -- needs
 * to ask someone for a page of physical memory, and until now there was nobody
 * to ask. The model was static in .bss precisely because of that.
 *
 * Frames are 4 KiB and tracked in a bitmap: one bit per frame, 0 = free. At
 * 4 GiB that bitmap is 128 KiB, which is why it is sized from the actual
 * memory map rather than assumed.
 *
 * The map itself comes from stage-1, which queries INT 15h AX=E820 in real mode
 * and hands it over in Multiboot 1 format. There is no other source: once the
 * CPU leaves real mode the BIOS is gone.
 * ============================================================================ */

#ifndef JL_PMM_H
#define JL_PMM_H

#include "jl_core.h"

#define PAGE_SIZE 4096u

/* Multiboot 1 memory map entry. `size` counts the bytes that FOLLOW it, so the
 * stride to the next entry is size + 4 and is not necessarily 24 -- some
 * loaders emit 24-byte payloads with ACPI attributes appended. */
typedef struct {
    uint32_t size;
    uint64_t base;
    uint64_t length;
    uint32_t type;              /* 1 = usable RAM; everything else is not */
} __attribute__((packed)) MmapEntry;

/* Parse the Multiboot info, mark usable RAM free and everything else in use.
 * The kernel image and the bitmap itself are reserved before any allocation is
 * permitted, so a caller cannot be handed the memory it is running from. */
void pmm_init(const void *mb_info);

/* One 4 KiB frame, or 0 when none remain. Returned memory is NOT zeroed --
 * callers that need zeroes ask for them, rather than paying on every path. */
uint32_t pmm_alloc_frame(void);

/* Return a frame. Double frees are ignored rather than corrupting the bitmap,
 * because a double free in a kernel with no fault handling is unrecoverable. */
void pmm_free_frame(uint32_t addr);

/* Reserve an explicit physical span, rounded outward to frame boundaries. */
void pmm_reserve(uint32_t start, uint32_t end);

uint32_t pmm_total_frames(void);
uint32_t pmm_used_frames(void);
uint32_t pmm_free_frames(void);

/* True once a usable map was found. If stage-1 reported nothing, allocation is
 * refused outright -- inventing a memory map would hand out addresses that may
 * not be backed by RAM, and the failure would appear much later as corruption. */
int pmm_ready(void);

#endif
