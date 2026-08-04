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
 * output scale applied as a right shift. */
typedef struct {
    uint8_t taps[NHIT_PACKED_BYTES];  /* low nibble first, each value +8 */
    int32_t shift;                    /* output >>= shift */
} ALIGN16 NhitKernel;

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

/* Index of the largest element -- the decision at the end of a stack. */
uint32_t nhit_argmax(const int8_t *v, uint32_t n);

/* Multiply-accumulate count for one nhit_apply, so the HUD can report real
 * arithmetic performed rather than an invented throughput figure. */
uint32_t nhit_mac_count(void);

/* Per-phase cycle attribution for the most recent nhit_stack call. */
extern uint32_t nhit_unpack_cycles;
extern uint32_t nhit_mac_cycles;

#endif
