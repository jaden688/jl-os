/* ============================================================================
 * JL OS :: KHEAP  --  kernel heap
 * ============================================================================
 * Rung 9. A first-fit free-list allocator over frames from the PMM.
 *
 * Every block carries an 8-byte header holding its size and a free flag, and
 * blocks are kept in a single address-ordered list so that adjacent free blocks
 * can be coalesced on release. Without coalescing, a kernel that allocates and
 * frees task stacks in a loop fragments until it cannot satisfy a request it
 * previously satisfied -- and a scheduler does exactly that.
 *
 * This is deliberately not a slab or buddy allocator. The kernel's allocation
 * pattern right now is a handful of task structures and stacks; a buddy
 * allocator's complexity buys nothing measurable at that scale and adds a class
 * of bug that is hard to see from a serial log.
 * ============================================================================ */

#include "kheap.h"
#include "pmm.h"
#include "vmm.h"
#include "serial.h"

#define HEAP_MAGIC 0x4A4C4845u          /* 'JLHE' -- detects header corruption */

typedef struct Block {
    uint32_t magic;
    uint32_t size;                      /* payload bytes, excluding header */
    struct Block *next;
    uint32_t free;
} Block;

static Block *head = 0;
static uint32_t heap_bytes = 0;
static uint32_t heap_used = 0;

/* Grow the heap by whole frames. Frames come from the PMM and are already
 * identity-mapped by vmm_init, so a physical frame is usable as-is. */
static Block *grow(uint32_t need) {
    uint32_t want = need + sizeof(Block);
    uint32_t pages = (want + PAGE_SIZE - 1) / PAGE_SIZE;

    uint32_t base = 0, prev = 0;
    for (uint32_t i = 0; i < pages; i++) {
        uint32_t f = pmm_alloc_frame();
        if (!f) return 0;
        /* Frames must be CONTIGUOUS to serve as one block. The bitmap
         * allocator returns ascending frames, so a break means memory is
         * fragmented and this request cannot be served as a single span. */
        if (i == 0) base = f;
        else if (f != prev + PAGE_SIZE) return 0;
        prev = f;
    }

    Block *b = (Block *)base;
    b->magic = HEAP_MAGIC;
    b->size = pages * PAGE_SIZE - sizeof(Block);
    b->free = 1;
    b->next = 0;
    heap_bytes += pages * PAGE_SIZE;

    if (!head) {
        head = b;
    } else {
        Block *t = head;
        while (t->next) t = t->next;
        t->next = b;
    }
    return b;
}

/* Split `b` if the remainder is large enough to hold a header plus useful
 * payload. Splitting unconditionally would produce headers with zero-byte
 * payloads that can never be allocated but must still be walked. */
static void split(Block *b, uint32_t size) {
    if (b->size < size + sizeof(Block) + 16) return;
    Block *n = (Block *)((uint8_t *)b + sizeof(Block) + size);
    n->magic = HEAP_MAGIC;
    n->size = b->size - size - sizeof(Block);
    n->free = 1;
    n->next = b->next;
    b->size = size;
    b->next = n;
}

void *kmalloc(uint32_t size) {
    if (!size) return 0;
    size = (size + 7u) & ~7u;           /* 8-byte alignment */

    for (Block *b = head; b; b = b->next) {
        if (b->free && b->size >= size) {
            split(b, size);
            b->free = 0;
            heap_used += b->size;
            return (uint8_t *)b + sizeof(Block);
        }
    }

    Block *b = grow(size);
    if (!b) return 0;
    split(b, size);
    b->free = 0;
    heap_used += b->size;
    return (uint8_t *)b + sizeof(Block);
}

void *kmalloc_aligned(uint32_t size, uint32_t align) {
    /* Over-allocate and step the pointer forward. The header of the returned
     * block is found by walking BACK from the aligned pointer, so the extra
     * space must be at least one header plus the alignment. */
    if (align < 8) return kmalloc(size);
    uint8_t *raw = kmalloc(size + align + sizeof(void *));
    if (!raw) return 0;
    uint32_t a = ((uint32_t)raw + sizeof(void *) + align - 1) & ~(align - 1);
    ((void **)a)[-1] = raw;             /* remember the real allocation */
    return (void *)a;
}

void kfree(void *p) {
    if (!p) return;
    Block *b = (Block *)((uint8_t *)p - sizeof(Block));
    if (b->magic != HEAP_MAGIC) {
        /* Either a double free of a coalesced block or a wild pointer. Refusing
         * is survivable; corrupting the list is not. */
        serial_write("[heap] kfree on a block with a bad header -- ignored\n");
        return;
    }
    if (b->free) return;
    b->free = 1;
    heap_used -= b->size;

    /* Coalesce forward only. The list is address-ordered, so a single forward
     * pass merges any run of adjacent free blocks over repeated frees. */
    while (b->next && b->next->free &&
           (uint8_t *)b + sizeof(Block) + b->size == (uint8_t *)b->next) {
        b->size += sizeof(Block) + b->next->size;
        b->next = b->next->next;
    }
}

uint32_t kheap_bytes(void) { return heap_bytes; }
uint32_t kheap_used(void) { return heap_used; }

int kheap_selftest(void) {
    /* Exercised at boot because a heap that fails under reuse fails in the
     * scheduler, where the symptom is a corrupted task list rather than a
     * visible allocation error. */
    void *a = kmalloc(64);
    void *b = kmalloc(64);
    if (!a || !b || a == b) return 0;
    kfree(a);
    void *c = kmalloc(64);           /* should reuse a's block */
    if (c != a) return 0;
    kfree(b);
    kfree(c);
    void *d = kmalloc(200);          /* should fit in the coalesced span */
    if (!d) return 0;
    kfree(d);
    return 1;
}
