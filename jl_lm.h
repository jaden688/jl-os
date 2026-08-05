/* ============================================================================
 * JL OS :: JL_LM  --  the language model, as a kernel primitive
 * ============================================================================
 * Character prediction running in ring 0 with no runtime underneath it: no
 * libc, no allocator beneath the kernel's own, no FPU, no OS services.
 *
 * The weights are NOT compiled in. They are read from /models/core.nhit on a
 * PEWTER-FS volume at boot, and the model's shape -- groups, heads, depth,
 * vocabulary -- comes from the blob header rather than from #defines. That is
 * the difference rung 14 bought: while the weights were .rodata, the model was
 * bounded by the 576 KB boot image; loaded from disk it is bounded by the
 * volume, and changing model size no longer means recompiling the kernel.
 *
 * The pipeline, all integer:
 *
 *   context chars -> INT8 embedding lookup
 *                 -> rotate by position, sum        (bundling on Z/N)
 *                 -> nhit_grouped                   (the GEMM replacement)
 *                 -> nhit_readout                   (INT8 matrix, INT32 logits)
 *                 -> argmax                         (the predicted character)
 *
 * Position is carried by ROTATION rather than a learned positional vector.
 * Displacing a value on Z/N is a rotation and composing displacements is
 * addition mod N, so order is expressed by the algebra instead of parameters.
 *
 * That bundling is also the model's tightest constraint, and it is worth being
 * honest about: JL_LM_CTX characters are superposed into NHIT_N INT8 values.
 * Capacity does not grow with the number of taps, so past some context length
 * the input is lossy no matter how large the model gets.
 * ============================================================================ */

#ifndef JL_LM_H
#define JL_LM_H

#include "jl_core.h"
#include "nhit.h"

#define JL_LM_CTX 16          /* characters of context; must match training */
#define JL_LM_MAGIC 0x5449484Eu   /* "NHIT" */

/* Parse a model blob and allocate its working state from the kernel heap.
 * Returns 0 and loads nothing if the header is wrong or the declared shape does
 * not account for exactly `len` bytes -- a blob that is close but not exact
 * would run and produce confident garbage. Requires the heap to be up. */
int jl_lm_load(const uint8_t *blob, uint32_t len);

int jl_lm_loaded(void);

/* Predict the next character from `ctx` (JL_LM_CTX character codes, oldest
 * first). Returns 0 if no model is loaded. `logit_out`, if non-null, receives
 * the winning INT32 logit so callers can display confidence rather than
 * asserting it. */
char jl_lm_predict(const char *ctx, int32_t *logit_out);

/* Continue `seed` for `n` characters, writing to `out` (n+1 bytes including
 * the terminator). Greedy argmax -- there is no entropy source in the kernel
 * and a fabricated one would make output irreproducible. */
void jl_lm_generate(const char *seed, char *out, uint32_t n);

/* Model shape, read from the loaded blob so the HUD reports what is actually
 * running rather than what was compiled. */
uint32_t jl_lm_taps(void);
uint32_t jl_lm_macs(void);
uint32_t jl_lm_groups(void);
uint32_t jl_lm_heads(void);
uint32_t jl_lm_depth(void);
uint32_t jl_lm_vocab(void);
uint32_t jl_lm_bytes(void);

#endif
