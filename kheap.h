/* ============================================================================
 * JL OS :: KHEAP  --  kernel heap
 * ============================================================================
 * Rung 9. First-fit with forward coalescing, backed by PMM frames.
 *
 * This is the first dynamic memory in the kernel. Everything before it was
 * static .bss -- including the model, which is why its size was bounded by the
 * boot image rather than by available RAM.
 * ============================================================================ */

#ifndef JL_KHEAP_H
#define JL_KHEAP_H

#include "jl_core.h"

#ifdef JL_HOST
/* The host harness scores the REAL jl_lm.c, which allocates from this heap.
 * Backing it with the platform malloc keeps a single inference implementation
 * -- a host-only copy of jl_lm would be a reimplementation, and a
 * reimplementation is where a discrepancy hides. The kernel never sees this. */
#include <stdlib.h>
static inline void *kmalloc(uint32_t n) { return malloc(n); }
static inline void *kmalloc_aligned(uint32_t n, uint32_t a) {
    (void)a; return malloc(n);
}
static inline void kfree(void *p) { free(p); }
static inline uint32_t kheap_bytes(void) { return 0; }
static inline uint32_t kheap_used(void) { return 0; }
static inline int kheap_selftest(void) { return 1; }
#else

/* Returns 0 when memory is exhausted. Callers must check: there is no OOM
 * killer and no exception path, so a dereferenced null here triple-faults. */
void *kmalloc(uint32_t size);

/* For page-aligned structures (page tables, task stacks). */
void *kmalloc_aligned(uint32_t size, uint32_t align);

void kfree(void *p);

uint32_t kheap_bytes(void);
uint32_t kheap_used(void);

/* Allocate/free/reuse/coalesce round-trip. Non-zero on success. */
int kheap_selftest(void);

#endif /* JL_HOST */

#endif
