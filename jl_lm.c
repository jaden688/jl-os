/* ============================================================================
 * JL OS :: JL_LM  --  language model inference (implementation)
 * ============================================================================
 * Integer only. Every value crossing this file is INT8, INT32 or a shift; there
 * is no float anywhere and no call outside nhit.c and this translation unit.
 * ============================================================================ */

#include "jl_lm.h"
#include "nhit_weights_mh.h"

/* Working buffers are static rather than automatic. At MH_GROUPS = 64 the
 * concatenated ring output alone is 4096 bytes, and the kernel stack is 16 KiB
 * shared with everything else -- putting these on the stack would make model
 * size silently couple to stack depth. */
static int8_t  lm_bundle[NHIT_N];
static int8_t  lm_rings[MH_GROUPS * NHIT_N];
static int32_t lm_logits[MH_VOCAB];
static NhitKernel lm_kernels[MH_GROUPS * MH_HEADS * MH_DEPTH];
static int lm_ready = 0;

/* Map a character to its vocabulary index, or -1. MH_CHARS is sorted, but at 64
 * entries a linear scan is a handful of comparisons and needs no invariant
 * maintained at export time. */
static int lm_index_of(char c) {
    for (uint32_t i = 0; i < sizeof(MH_CHARS); i++)
        if ((unsigned char)c == MH_CHARS[i]) return (int)i;
    return -1;
}

static int8_t lm_sat8(int32_t v) {
    if (v > 127) return (int8_t)127;
    if (v < -128) return (int8_t)(-128);
    return (int8_t)v;
}

/* Unpack the generated tap/shift tables into the NhitKernel array once. The
 * tables are .rodata in the flat image; this just gives nhit_grouped the struct
 * layout it walks. */
static void lm_init(void) {
    const uint32_t rings = MH_GROUPS * MH_HEADS * MH_DEPTH;
    for (uint32_t r = 0; r < rings; r++) {
        lm_kernels[r].shift = (int8_t)MH_SHIFT[r];
        for (uint32_t b = 0; b < NHIT_PACKED_BYTES; b++)
            lm_kernels[r].taps[b] = MH_TAPS[r * NHIT_PACKED_BYTES + b];
    }
    lm_ready = 1;
}

/* Bundle the context into one ring: each character's INT8 embedding is rotated
 * by its position and the rotated vectors are summed. Accumulation is INT32 so
 * the sum cannot wrap before saturation is applied deliberately. */
static void lm_bundle_ctx(const char *ctx) {
    int32_t acc[NHIT_N];
    for (uint32_t i = 0; i < NHIT_N; i++) acc[i] = 0;

    for (uint32_t p = 0; p < JL_LM_CTX; p++) {
        int idx = lm_index_of(ctx[p]);
        if (idx < 0) continue;              /* out-of-vocabulary: contributes nothing */
        const signed char *e = &MH_EMBED[(uint32_t)idx * NHIT_N];
        for (uint32_t i = 0; i < NHIT_N; i++) {
            /* Rotation by p on Z/N. This is what carries position -- rotating
             * and summing is not commutative across different p, so the
             * bundle distinguishes "ab" from "ba". */
            acc[(i + p) & (NHIT_N - 1)] += (int32_t)e[i];
        }
    }

    /* Scale the bundle back into INT8. Dividing by the context length keeps the
     * range comparable to a single embedding, which is what the ring taps were
     * trained against. */
    for (uint32_t i = 0; i < NHIT_N; i++)
        lm_bundle[i] = lm_sat8(acc[i] / (int32_t)JL_LM_CTX);
}

char jl_lm_predict(const char *ctx, int32_t *logit_out) {
    if (!lm_ready) lm_init();

    lm_bundle_ctx(ctx);
    nhit_grouped(lm_kernels, MH_GROUPS, MH_HEADS, MH_DEPTH,
                 lm_bundle, lm_rings);
    nhit_readout(MH_READ_W, MH_READ_B, MH_READ_DIM, MH_VOCAB,
                 MH_READ_SHIFT, lm_rings, lm_logits);

    uint32_t best = nhit_argmax32(lm_logits, MH_VOCAB);
    if (logit_out) *logit_out = lm_logits[best];
    return (char)MH_CHARS[best];
}

void jl_lm_generate(const char *seed, char *out, uint32_t n) {
    char ctx[JL_LM_CTX];

    /* Right-align the seed in the context window, padding with spaces. A seed
     * shorter than the window is the normal case, and left-aligning it would
     * put the useful characters at the wrong rotations entirely. */
    uint32_t len = 0;
    while (seed[len]) len++;
    for (uint32_t i = 0; i < JL_LM_CTX; i++) {
        uint32_t back = JL_LM_CTX - i;
        ctx[i] = (len >= back) ? seed[len - back] : ' ';
    }

    for (uint32_t k = 0; k < n; k++) {
        char c = jl_lm_predict(ctx, 0);
        out[k] = c;
        /* Slide the window: drop the oldest, append the prediction. */
        for (uint32_t i = 0; i + 1 < JL_LM_CTX; i++) ctx[i] = ctx[i + 1];
        ctx[JL_LM_CTX - 1] = c;
    }
    out[n] = 0;
}

int jl_lm_selftest(void) {
    /* The weight header and nhit.h are generated and hand-written respectively,
     * so nothing guarantees they agree. A mismatch here does not crash -- it
     * produces confident nonsense -- which is exactly the failure worth
     * catching at boot rather than discovering from output that looks plausible. */
    if (sizeof(MH_CHARS) != MH_VOCAB) return 0;
    if (MH_READ_DIM != MH_GROUPS * NHIT_N) return 0;
    if (sizeof(MH_TAPS) !=
        (uint32_t)MH_GROUPS * MH_HEADS * MH_DEPTH * NHIT_PACKED_BYTES) return 0;
    if (sizeof(MH_SHIFT) != (uint32_t)MH_GROUPS * MH_HEADS * MH_DEPTH) return 0;
    if (sizeof(MH_EMBED) != (uint32_t)MH_VOCAB * NHIT_N) return 0;
    if (sizeof(MH_READ_W) != (uint32_t)MH_VOCAB * MH_READ_DIM) return 0;
    return 1;
}

int jl_lm_verify_blob(const uint8_t *blob, uint32_t len) {
    /* Layout must match export_multihead.py's blob writer exactly:
     *   u32 magic 'NHIT', u32 groups, heads, depth, vocab, read_dim
     *   packed taps, shifts, embed, readout weights, bias, chars
     * Comparing against the compiled-in tables is what proves the on-disk
     * form is trustworthy enough to replace them. */
    const uint32_t hdr = 24;
    if (len < hdr) return 0;
    const uint32_t *h = (const uint32_t *)blob;
    if (h[0] != 0x5449484Eu) return 0;                 /* "NHIT" */
    if (h[1] != MH_GROUPS || h[2] != MH_HEADS || h[3] != MH_DEPTH) return 0;
    if (h[4] != MH_VOCAB || h[5] != MH_READ_DIM) return 0;

    const uint32_t rings = (uint32_t)MH_GROUPS * MH_HEADS * MH_DEPTH;
    const uint8_t *p = blob + hdr;

    for (uint32_t i = 0; i < rings * NHIT_PACKED_BYTES; i++)
        if (*p++ != MH_TAPS[i]) return 0;
    for (uint32_t i = 0; i < rings; i++)
        if ((int8_t)*p++ != MH_SHIFT[i]) return 0;
    for (uint32_t i = 0; i < (uint32_t)MH_VOCAB * NHIT_N; i++)
        if ((int8_t)*p++ != MH_EMBED[i]) return 0;
    for (uint32_t i = 0; i < (uint32_t)MH_VOCAB * MH_READ_DIM; i++)
        if ((int8_t)*p++ != MH_READ_W[i]) return 0;

    const int32_t *b = (const int32_t *)p;
    for (uint32_t i = 0; i < MH_VOCAB; i++)
        if (b[i] != MH_READ_B[i]) return 0;
    p += 4u * MH_VOCAB;

    for (uint32_t i = 0; i < MH_VOCAB; i++)
        if (*p++ != MH_CHARS[i]) return 0;

    return (uint32_t)(p - blob) == len;
}

uint32_t jl_lm_taps(void) {
    return (uint32_t)MH_GROUPS * MH_HEADS * MH_DEPTH * NHIT_N;
}

uint32_t jl_lm_macs(void) {
    /* Ring MACs plus the readout matrix. The readout is a real matrix and is
     * not free: at 64 groups it is 4096x64, comparable to the rings. */
    return jl_lm_taps() * NHIT_N + (uint32_t)MH_READ_DIM * MH_VOCAB;
}
