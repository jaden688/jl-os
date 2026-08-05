/* ============================================================================
 * JL OS :: VMM  --  paging (implementation)
 * ============================================================================ */

#include "vmm.h"
#include "pmm.h"
#include "serial.h"

#define ENTRIES 1024u
#define PD_INDEX(v) ((v) >> 22)
#define PT_INDEX(v) (((v) >> 12) & 0x3FFu)

static uint32_t *page_directory = 0;
static int enabled = 0;
static uint32_t mapped_bytes = 0;

static inline void load_cr3(uint32_t pd) {
    __asm__ __volatile__("movl %0, %%cr3" :: "r"(pd) : "memory");
}

static inline void enable_paging(void) {
    uint32_t cr0;
    __asm__ __volatile__("movl %%cr0, %0" : "=r"(cr0));
    cr0 |= 0x80000000u;                 /* CR0.PG */
    __asm__ __volatile__("movl %0, %%cr0" :: "r"(cr0) : "memory");
}

static inline void invlpg(uint32_t v) {
    __asm__ __volatile__("invlpg (%0)" :: "r"(v) : "memory");
}

static void zero_page(uint32_t phys) {
    uint32_t *p = (uint32_t *)phys;
    for (uint32_t i = 0; i < PAGE_SIZE / 4; i++) p[i] = 0;
}

int vmm_map_page(uint32_t virt, uint32_t phys, uint32_t flags) {
    if (!page_directory) return 0;
    uint32_t pdi = PD_INDEX(virt);

    if (!(page_directory[pdi] & PAGE_PRESENT)) {
        uint32_t table = pmm_alloc_frame();
        if (!table) return 0;
        /* Zeroed before it is referenced: a page table full of garbage would
         * be interpreted as present mappings to arbitrary frames. */
        zero_page(table);
        page_directory[pdi] = table | PAGE_PRESENT | PAGE_WRITE;
    }

    uint32_t *table = (uint32_t *)(page_directory[pdi] & 0xFFFFF000u);
    table[PT_INDEX(virt)] = (phys & 0xFFFFF000u) | (flags & 0xFFFu) | PAGE_PRESENT;
    if (enabled) invlpg(virt);
    return 1;
}

uint32_t vmm_translate(uint32_t virt) {
    if (!page_directory) return 0;
    uint32_t pde = page_directory[PD_INDEX(virt)];
    if (!(pde & PAGE_PRESENT)) return 0;
    uint32_t *table = (uint32_t *)(pde & 0xFFFFF000u);
    uint32_t pte = table[PT_INDEX(virt)];
    if (!(pte & PAGE_PRESENT)) return 0;
    return (pte & 0xFFFFF000u) | (virt & 0xFFFu);
}

int vmm_init(void) {
    if (!pmm_ready()) {
        serial_write("[vmm]  no physical allocator -- paging NOT enabled\n");
        return 0;
    }

    uint32_t pd = pmm_alloc_frame();
    if (!pd) {
        serial_write("[vmm]  cannot allocate page directory -- paging NOT enabled\n");
        return 0;
    }
    zero_page(pd);
    page_directory = (uint32_t *)pd;

    /* Identity-map every usable frame. The frame count from the PMM is the
     * amount of RAM that actually exists, so this maps memory rather than
     * address space -- mapping a fixed 4 GiB would cost 4 MiB of page tables
     * to describe holes.
     *
     * The map is built INCLUSIVE of the low megabyte and the kernel image even
     * though those frames are reserved in the allocator: reserved means "do not
     * hand out", not "do not address". Leaving them unmapped would fault on the
     * first VGA write. */
    uint32_t top = (pmm_total_frames() + 256u) * PAGE_SIZE;
    if (top < 0x1000000u) top = 0x1000000u;      /* at least 16 MiB */

    for (uint32_t a = 0; a < top; a += PAGE_SIZE) {
        if (!vmm_map_page(a, a, PAGE_WRITE)) {
            /* Ran out of frames building the map. Stop where we are rather
             * than enable paging over a partial map that omits the kernel. */
            serial_write("[vmm]  out of frames while mapping -- paging NOT enabled\n");
            return 0;
        }
        mapped_bytes += PAGE_SIZE;
    }

    load_cr3(pd);
    enable_paging();
    enabled = 1;

    serial_write("[vmm]  identity map ");
    serial_dec(mapped_bytes / (1024u * 1024u));
    serial_write(" MiB, CR0.PG set, ");
    serial_dec(pmm_used_frames() * 4);
    serial_write(" KiB of frames in use\n");
    return 1;
}

int vmm_enabled(void) { return enabled; }
uint32_t vmm_mapped_bytes(void) { return mapped_bytes; }
