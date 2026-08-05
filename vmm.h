/* ============================================================================
 * JL OS :: VMM  --  paging
 * ============================================================================
 * Rung 8. Two-level x86 paging: a 1024-entry page directory, each entry
 * pointing at a 1024-entry page table, each entry mapping one 4 KiB frame.
 *
 * The mapping installed at boot is IDENTITY -- virtual address equals physical
 * address across all usable RAM. That is a deliberate choice, not a shortcut
 * deferred: the kernel is linked at a fixed physical address (1 MiB, see
 * linker.ld) and every pointer already in flight -- the model weights in
 * .rodata, the E820 map at 0x9000, the VGA buffer at 0xB8000 -- is physical.
 * A higher-half or offset mapping would invalidate all of them at the instant
 * CR0.PG is set, which is the single least debuggable moment in a kernel.
 *
 * What paging buys immediately is not isolation but ENFORCEMENT: with page
 * tables active, a null dereference or a wild write into an unmapped hole
 * raises #PF at the faulting instruction instead of silently corrupting
 * whatever happened to live there. Rung 11 (ring 3) is what turns this into
 * actual protection; this is the prerequisite.
 * ============================================================================ */

#ifndef JL_VMM_H
#define JL_VMM_H

#include "jl_core.h"

#define PAGE_PRESENT  0x001u
#define PAGE_WRITE    0x002u
#define PAGE_USER     0x004u

/* Build the identity map over usable RAM and turn on CR0.PG. Requires pmm_init
 * to have run: the page tables themselves are allocated from the frame
 * allocator. Returns 0 if paging could not be enabled, in which case the kernel
 * keeps running unpaged rather than halting. */
int vmm_init(void);

/* Map one 4 KiB page. Allocates a page table if the directory entry is empty.
 * Returns 0 on allocation failure. */
int vmm_map_page(uint32_t virt, uint32_t phys, uint32_t flags);

/* Physical frame backing `virt`, or 0 if unmapped. */
uint32_t vmm_translate(uint32_t virt);

int vmm_enabled(void);
uint32_t vmm_mapped_bytes(void);

#endif
