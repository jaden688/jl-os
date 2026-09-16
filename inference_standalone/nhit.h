/* ============================================================================
 * JL OS :: NHIT  --  the kernel's compute primitive
 * ============================================================================
 * NHIT replaces GEMM. It is not a matrix multiply.
 *
 * A GEMM applies an R x C weight MATRIX:
 *
 *     out[r] = sum_c  in[c] * W[r][c]              R*C weights
 *
 * NHIT applies a single length-N KERNEL VECTOR by circular convolution on the
 * ring Z/N:
 *
 *     out[k] = sum_j  in[j] * w[(k - j) mod N]     N weights
 *
 * Same class of linear transform, N weights instead of N^2. At N = 64 that is
 * 64 values rather than 4096 -- a 64x reduction, and the whole transform fits
 * in cache on any machine that can boot.
 *
 * Why a ring: convolution on Z/N is the group operation of Z/N. Displacing a
 * value by d is a rotation, and composing displacements is addition mod N, so
 * position and structure are carried by the algebra itself rather than by
 * learned coefficients. This is the operator that scored 0.9431 on held-out
 * compositional binding with 1,388 parameters, where a 272k-parameter
 * transformer scored 0.0733.
 *
 * Everything is INTEGER. No FPU, no libc, no runtime:
 *
 *     kernel weights   INT4, packed two per byte, +8 biased
 *     activations      INT8
 *     accumulator      INT32
 *     rescale          arithmetic right shift (power-of-two scales only)
 *
 * A stacked variant adds the nonlinearity that makes composition possible:
 * without one, stacked convolutions collapse into a single convolution.
 * ============================================================================ */

#ifndef JL_NHIT_H
#define JL_NHIT_H

#include "jl_core.h"

#define NHIT_N 64                    /* ring size; must be a power of two */
#define NHIT_PACKED_BYTES (NHIT_N / 2)

/* A ring kernel: NHIT_N INT4 taps packed two per byte, plus a power-of-two
 * output scale applied as a right shift.
 *
 * Deliberately NOT ALIGN16. The 16-byte alignment was there for a vectorized
 * apply that measurement rejected, and it padded 33 useful bytes to 48 -- 31%
 * of all weight storage lost to alignment for an unused path. At the scales
 * that matter (thousands of rings) that padding is tens of kilobytes against a
 * 576 KB image ceiling. The shift is int8_t for the same reason: its range is
 * 0..20, and four bytes per ring bought nothing. */
typedef struct {
    uint8_t taps[NHIT_PACKED_BYTES];  /* low nibble first, each value +8 */
    int8_t  shift;                    /* output >>= shift */
} NhitKernel;

/* Unpack tap i to its signed value in [-8, 7]. */
int32_t nhit_tap(const NhitKernel *k, uint32_t i);

/* out[k] = ( sum_j in[j] * w[(k-j) mod N] ) >> shift, saturated to INT8.
 * `in` and `out` are NHIT_N INT8 values. Aliasing is not permitted. */
void nhit_apply(const NhitKernel *k, const int8_t *in, int8_t *out);

/* ReLU in place: negatives to zero. The nonlinearity between stacked rings. */
void nhit_relu(int8_t *v, uint32_t n);

/* Apply `depth` kernels in sequence with ReLU between them. */
void nhit_stack(const NhitKernel *kernels, uint32_t depth,
                const int8_t *in, int8_t *out);

/* Multi-head: `heads` independent rings over the SAME input, summed.
 *
 * This is the scaling direction, and it is not obvious. A ring of size N holds
 * N taps but costs N^2 MACs, so parameters grow linearly while compute grows
 * quadratically. Widening N therefore buys parameters at a terrible rate:
 * at a fixed compute budget, N=64 yields 16x more taps than N=1024.
 *
 * Many small rings is the parameter-efficient shape. Heads are independent --
 * no cross-head dependencies, fixed sizes throughout -- which is also the
 * shape fixed-function accelerators execute best.
 *
 * `kernels` holds heads*depth entries, head-major: head h uses
 * kernels[h*depth .. h*depth+depth-1]. */
void nhit_multihead(const NhitKernel *kernels, uint32_t heads, uint32_t depth,
                    const int8_t *in, int8_t *out);

/* Grouped multi-head: `groups` independent multiheads over the SAME input,
 * each writing NHIT_N values, CONCATENATED into out[groups * NHIT_N].
 *
 * Why this exists, and it is the difference between scaling working and not:
 * nhit_multihead averages every head into one NHIT_N vector, so no matter how
 * many taps feed it the representation reaching the readout is 64 numbers.
 * Heads past the first few then buy nothing. Measured at equal tap count
 * (196,608 taps, 1000 steps): one group of 1024 heads scored +0.0176 over
 * majority-class baseline, sixteen groups of 64 heads scored +0.0261.
 *
 * Averaging is also near-degenerate on its own terms -- absent the ReLU, the
 * mean of H convolutions IS a single convolution with the mean taps, so H heads
 * would be exactly one head. Groups widen; heads only deepen the mixture.
 *
 * `kernels` holds groups*heads*depth entries, group-major then head-major. */
void nhit_grouped(const NhitKernel *kernels, uint32_t groups, uint32_t heads,
                  uint32_t depth, const int8_t *in, int8_t *out);

/* Integer readout: out[r] = ( sum_c in[c] * w[r*in_dim + c] ) >> shift + bias.
 *
 * This is a genuine matrix, not a ring -- the one place a GEMM shape is
 * unavoidable, because the vocabulary decision is not translation-invariant on
 * Z/N and cannot be expressed as a convolution. It is also the dominant weight
 * cost: INT8 and proportional to groups, so at 64 groups it is 256 KB against
 * the taps' 96 KB. Accumulates in INT32; `in` is INT8 and `w` is INT8. */
void nhit_readout(const int8_t *w, const int32_t *bias, uint32_t in_dim,
                  uint32_t out_dim, int32_t shift,
                  const int8_t *in, int32_t *out);

/* Index of the largest element -- the decision at the end of a stack. */
uint32_t nhit_argmax(const int8_t *v, uint32_t n);

/* Same, over INT32 logits as produced by nhit_readout. */
uint32_t nhit_argmax32(const int32_t *v, uint32_t n);

/* Zero the profiling counters. Call once before a measured region: nhit_stack
 * deliberately does not reset, so that multi-head work accumulates. */
void nhit_prof_reset(void);

/* Multiply-accumulate count for one nhit_apply, so the HUD can report real
 * arithmetic performed rather than an invented throughput figure. */
uint32_t nhit_mac_count(void);

/* Per-phase cycle attribution for the most recent nhit_stack call. */
extern uint32_t nhit_unpack_cycles;
extern uint32_t nhit_mac_cycles;

#endif
