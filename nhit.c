/* ============================================================================
 * JL OS :: NHIT  --  kernel compute primitive (implementation)
 * ============================================================================
 * Circular convolution on the ring Z/N, integer only.
 *
 *     out[k] = ( sum_j in[j] * w[(k - j) mod N] ) >> shift
 *
 * The modulo is free: NHIT_N is a power of two, so (k - j) mod N is a mask.
 * No division, no float, no libc, no runtime. This compiles to loads, signed
 * multiplies, adds and one shift -- the instruction mix a bare-metal CPU and a
 * fixed-function accelerator both execute natively.
 *
 * Contrast with the GEMM this replaces: an R x C matmul carries R*C weights and
 * touches all of them once. NHIT carries N and reuses each N times, so the
 * working set is a single cache line at N = 64 while the arithmetic per output
 * is identical.
 * ============================================================================ */

#include "nhit.h"

#define NHIT_MASK (NHIT_N - 1)

/* INT4 taps are stored +8 biased, two per byte, low nibble first. */
int32_t nhit_tap(const NhitKernel *k, uint32_t i) {
    uint8_t packed = k->taps[(i >> 1) & (NHIT_PACKED_BYTES - 1)];
    uint8_t nib = (i & 1u) ? (uint8_t)(packed >> 4) : (uint8_t)(packed & 0x0Fu);
    return (int32_t)nib - 8;
}

static int8_t sat8(int32_t v) {
    if (v > 127) return (int8_t)127;
    if (v < -128) return (int8_t)(-128);
    return (int8_t)v;
}

/* Instrumentation: cycles attributed to each phase of the last nhit_stack.
 * nhit_apply does two very different things -- unpack 64 INT4 taps, then run
 * 64x64 multiply-accumulates -- and the aggregate number cannot say which one
 * costs. These counters split it. */
uint32_t nhit_unpack_cycles = 0;
uint32_t nhit_mac_cycles = 0;

static inline uint64_t nhit_rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

void nhit_apply(const NhitKernel *k, const int8_t *in, int8_t *out) {
    int32_t taps[NHIT_N];

    uint64_t t_unpack0 = nhit_rdtsc();
    for (uint32_t i = 0; i < NHIT_N; i++) taps[i] = nhit_tap(k, i);
    uint64_t t_unpack1 = nhit_rdtsc();
    nhit_unpack_cycles += (uint32_t)(t_unpack1 - t_unpack0);

    uint64_t t_mac0 = nhit_rdtsc();
    for (uint32_t o = 0; o < NHIT_N; o++) {
        int32_t acc = 0;
        for (uint32_t j = 0; j < NHIT_N; j++) {
            /* (o - j) mod N, as a mask because N is a power of two. */
            acc += (int32_t)in[j] * taps[(o - j) & NHIT_MASK];
        }
        out[o] = sat8(k->shift >= 0 ? (acc >> k->shift) : (acc << (-k->shift)));
    }
    uint64_t t_mac1 = nhit_rdtsc();
    nhit_mac_cycles += (uint32_t)(t_mac1 - t_mac0);
}

void nhit_relu(int8_t *v, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) if (v[i] < 0) v[i] = 0;
}

void nhit_stack(const NhitKernel *kernels, uint32_t depth,
                const int8_t *in, int8_t *out) {
    int8_t a[NHIT_N], b[NHIT_N];
    nhit_unpack_cycles = 0;
    nhit_mac_cycles = 0;
    for (uint32_t i = 0; i < NHIT_N; i++) a[i] = in[i];

    for (uint32_t d = 0; d < depth; d++) {
        nhit_apply(&kernels[d], a, b);
        /* Without a nonlinearity between rings, stacked convolutions collapse
         * into a single convolution and depth buys nothing. ReLU is what makes
         * the stack more expressive than one kernel. */
        if (d + 1 < depth) nhit_relu(b, NHIT_N);
        for (uint32_t i = 0; i < NHIT_N; i++) a[i] = b[i];
    }
    for (uint32_t i = 0; i < NHIT_N; i++) out[i] = a[i];
}

uint32_t nhit_argmax(const int8_t *v, uint32_t n) {
    uint32_t best = 0;
    int32_t bestv = -1000;
    for (uint32_t i = 0; i < n; i++) {
        if ((int32_t)v[i] > bestv) { bestv = v[i]; best = i; }
    }
    return best;
}

uint32_t nhit_mac_count(void) {
    return (uint32_t)NHIT_N * (uint32_t)NHIT_N;
}
