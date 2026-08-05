/* ============================================================================
 * JL OS :: PMM  --  physical memory manager (implementation)
 * ============================================================================ */

#include "pmm.h"
#include "serial.h"

/* Multiboot info field offsets are fixed by the specification. Reading them by
 * offset rather than through a struct avoids depending on how a compiler lays
 * out a 90-byte packed structure we only need four fields from. */
#define MB_FLAGS(p)       (*(const uint32_t *)((const uint8_t *)(p) + 0))
#define MB_MEM_LOWER(p)   (*(const uint32_t *)((const uint8_t *)(p) + 4))
#define MB_MMAP_LENGTH(p) (*(const uint32_t *)((const uint8_t *)(p) + 44))
#define MB_MMAP_ADDR(p)   (*(const uint32_t *)((const uint8_t *)(p) + 48))
#define MB_FLAG_MMAP      0x40u

/* Bitmap sized for 4 GiB of 4 KiB frames: 1,048,576 bits = 128 KiB. Static
 * because the allocator cannot allocate its own storage -- something has to be
 * the fixed point, and a 128 KiB .bss cost is the cheapest available. */
#define MAX_FRAMES (1024u * 1024u)
static uint32_t frame_bitmap[MAX_FRAMES / 32];

static uint32_t total_frames = 0;
static uint32_t used_frames = 0;
static uint32_t highest_frame = 0;
static int ready = 0;

/* Linker-provided image bounds; the kernel must not hand out its own memory. */
extern uint8_t kernel_start[];
extern uint8_t kernel_end[];

static inline void bm_set(uint32_t f) { frame_bitmap[f >> 5] |= (1u << (f & 31)); }
static inline void bm_clear(uint32_t f) { frame_bitmap[f >> 5] &= ~(1u << (f & 31)); }
static inline int bm_test(uint32_t f) { return (frame_bitmap[f >> 5] >> (f & 31)) & 1u; }

void pmm_reserve(uint32_t start, uint32_t end) {
    /* Round outward: a partially covered frame is entirely unusable, and
     * rounding inward would hand out memory that is half occupied. */
    uint32_t f = start / PAGE_SIZE;
    uint32_t last = (end + PAGE_SIZE - 1) / PAGE_SIZE;
    for (; f < last && f < MAX_FRAMES; f++) {
        if (!bm_test(f)) { bm_set(f); used_frames++; }
    }
}

void pmm_init(const void *mb_info) {
    for (uint32_t i = 0; i < MAX_FRAMES / 32; i++) frame_bitmap[i] = 0xFFFFFFFFu;
    used_frames = MAX_FRAMES;
    total_frames = 0;

    if (!mb_info || !(MB_FLAGS(mb_info) & MB_FLAG_MMAP)) {
        /* Refuse rather than guess. A fabricated map would hand out addresses
         * that may not be backed by RAM, and the corruption would surface far
         * from here. */
        serial_write("[pmm]  NO MEMORY MAP from bootloader -- allocation disabled\n");
        ready = 0;
        return;
    }

    uint32_t addr = MB_MMAP_ADDR(mb_info);
    uint32_t len = MB_MMAP_LENGTH(mb_info);
    uint32_t off = 0;
    uint32_t regions = 0;

    while (off < len) {
        const MmapEntry *e = (const MmapEntry *)(addr + off);
        /* Stride is size + 4: `size` excludes itself. Trusting a fixed 24 would
         * desynchronise on any BIOS that appends ACPI attributes. */
        uint32_t stride = e->size + 4;
        if (stride < sizeof(MmapEntry)) break;

        if (e->type == 1) {
            uint64_t base = e->base;
            uint64_t end = base + e->length;
            /* This kernel is 32-bit with no PAE, so anything at or above 4 GiB
             * is unaddressable and is clamped away rather than truncated. */
            if (base < 0x100000000ULL) {
                if (end > 0x100000000ULL) end = 0x100000000ULL;
                uint32_t f = (uint32_t)((base + PAGE_SIZE - 1) / PAGE_SIZE);
                uint32_t l = (uint32_t)(end / PAGE_SIZE);
                for (; f < l && f < MAX_FRAMES; f++) {
                    if (bm_test(f)) { bm_clear(f); used_frames--; total_frames++; }
                    if (f > highest_frame) highest_frame = f;
                }
                regions++;
            }
        }
        off += stride;
    }

    /* Reclaim the accounting: frames outside any usable region were never real
     * memory, so they are not "used" -- they simply do not exist. */
    used_frames -= (MAX_FRAMES - total_frames);

    /* The first megabyte holds the IVT, BDA, our stage-1, the E820 map we are
     * still reading, and the VGA/BIOS holes. None of it is allocatable. */
    pmm_reserve(0, 0x100000);
    pmm_reserve((uint32_t)kernel_start, (uint32_t)kernel_end);

    ready = 1;

    serial_write("[pmm]  regions ");
    serial_dec(regions);
    serial_write("  frames ");
    serial_dec(total_frames);
    serial_write("  (");
    serial_dec(total_frames * 4 / 1024);
    serial_write(" MiB usable)  free ");
    serial_dec(total_frames - used_frames);
    serial_write("\n");
}

uint32_t pmm_alloc_frame(void) {
    if (!ready) return 0;
    for (uint32_t w = 0; w < MAX_FRAMES / 32; w++) {
        if (frame_bitmap[w] == 0xFFFFFFFFu) continue;   /* word is fully used */
        for (uint32_t b = 0; b < 32; b++) {
            if (!((frame_bitmap[w] >> b) & 1u)) {
                uint32_t f = w * 32 + b;
                bm_set(f);
                used_frames++;
                return f * PAGE_SIZE;
            }
        }
    }
    return 0;
}

void pmm_free_frame(uint32_t addr) {
    uint32_t f = addr / PAGE_SIZE;
    if (f >= MAX_FRAMES) return;
    /* Ignore a double free instead of decrementing used_frames twice. With no
     * fault handling, a corrupted bitmap is unrecoverable and silent. */
    if (!bm_test(f)) return;
    bm_clear(f);
    used_frames--;
}

uint32_t pmm_total_frames(void) { return total_frames; }
uint32_t pmm_used_frames(void) { return used_frames; }
uint32_t pmm_free_frames(void) { return total_frames - used_frames; }
int pmm_ready(void) { return ready; }
