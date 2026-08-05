/* ============================================================================
 * JL OS :: JL_LM  --  the language model, as a kernel primitive
 * ============================================================================
 * Character prediction running in ring 0 with no runtime underneath it: no
 * libc, no allocator, no FPU, no operating-system services. The entire model
 * is .rodata in the flat kernel image, loaded by our own stage-1 loader.
 *
 * The pipeline, all integer:
 *
 *   context chars -> INT8 embedding lookup
 *                 -> rotate by position, sum        (bundling on Z/N)
 *                 -> nhit_grouped                   (the GEMM replacement)
 *                 -> nhit_readout                   (INT8 matrix, INT32 logits)
 *                 -> argmax                         (the predicted character)
 *
 * Position is carried by ROTATION rather than by a learned positional vector.
 * Displacing a value on the ring Z/N is a rotation and composing displacements
 * is addition mod N, so order is expressed by the algebra instead of by
 * parameters. Summing the rotated embeddings bundles the context into a single
 * ring -- the HRR/VSA superposition, done in INT8.
 *
 * That bundling is also the model's tightest constraint, and it is worth being
 * honest about: JL_LM_CTX characters are superposed into NHIT_N INT8 values.
 * Capacity does not grow with the number of taps, so beyond some context length
 * the input is lossy no matter how large the model gets.
 * ============================================================================ */

#ifndef JL_LM_H
#define JL_LM_H

#include "jl_core.h"
#include "nhit.h"

#define JL_LM_CTX 16          /* characters of context; must match training */

/* Predict the next character from `ctx` (JL_LM_CTX character codes, oldest
 * first). Returns the predicted character. `logit_out`, if non-null, receives
 * the winning INT32 logit so callers can display confidence rather than
 * asserting it. */
char jl_lm_predict(const char *ctx, int32_t *logit_out);

/* Continue `seed` for `n` characters, writing to `out` (n+1 bytes, including
 * the terminator). Greedy argmax -- no sampling, since there is no entropy
 * source in the kernel and a fabricated one would make output irreproducible. */
void jl_lm_generate(const char *seed, char *out, uint32_t n);

/* True if the model header compiled in is self-consistent with nhit.h. Checked
 * at boot rather than assumed: a mismatched NHIT_N or vocabulary would produce
 * confident garbage instead of a visible failure. */
int jl_lm_selftest(void);

/* Compare a model blob read from PEWTER-FS against the weights compiled into
 * this kernel. Non-zero if every byte matches. This is the check that licenses
 * deleting the compiled-in copy: until the on-disk form is proven identical,
 * moving the model to disk is a change of storage AND a change of weights, and
 * a failure could not be attributed to either. */
int jl_lm_verify_blob(const uint8_t *blob, uint32_t len);

/* Model shape, for the HUD to report what is actually running. */
uint32_t jl_lm_taps(void);
uint32_t jl_lm_macs(void);

#endif
