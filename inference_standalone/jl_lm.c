/* ============================================================================
 * JL OS :: JL_LM  --  language model inference (implementation)
 * ============================================================================
 * Integer only. Every value crossing this file is INT8, INT32 or a shift; there
 * is no float anywhere and no call outside nhit.c, the kernel heap, and this
 * translation unit.
 * ============================================================================ */

#include "jl_lm.h"
/* kheap.h not needed — kmalloc/kfree come from jl_core.h via stdlib */

/* Loaded model. All of it points into the blob the caller supplied, except the
 * unpacked kernel array and the scratch buffers, which are heap-allocated.
 * The blob itself is NOT copied -- it is already in memory and copying 171 KiB
 * to own it would double the cost for no benefit. Callers must therefore keep
 * the blob alive, which kmain does by holding it in a static buffer. */
static uint32_t m_groups, m_heads, m_depth, m_vocab, m_read_dim, m_rings;
static uint32_t m_ctx;
static const uint8_t *m_chars;
static const int8_t *m_embed;
static const int8_t *m_read_w;
static const int32_t *m_read_b;

static NhitKernel *m_kernels;
static int8_t *m_bundle;
static int8_t *m_ringbuf;
static int32_t *m_logits;
static uint32_t m_bytes;
static int loaded = 0;

static int8_t lm_sat8(int32_t v) {
    if (v > 127) return (int8_t)127;
    if (v < -128) return (int8_t)(-128);
    return (int8_t)v;
}

/* Map a character to its vocabulary index, or -1. Linear over 64 entries is a
 * handful of comparisons and needs no invariant maintained at export time. */
static int lm_index_of(char c) {
    for (uint32_t i = 0; i < m_vocab; i++)
        if ((unsigned char)c == m_chars[i]) return (int)i;
    return -1;
}

int jl_lm_load(const uint8_t *blob, uint32_t len) {
    loaded = 0;
    if (!blob || len < 28) return 0;

    const uint32_t *h = (const uint32_t *)blob;
    if (h[0] != JL_LM_MAGIC) return 0;

    uint32_t groups = h[1], heads = h[2], depth = h[3];
    uint32_t vocab = h[4], read_dim = h[5], ctx = h[6];

    /* Shape has to be self-consistent AND consistent with this kernel's ring
     * size, which is a compile-time constant in nhit.h. A blob built for a
     * different NHIT_N would index off the end of every buffer below. */
    if (!groups || !heads || !depth || !vocab) return 0;
    if (read_dim != groups * NHIT_N) return 0;
    if (!ctx || ctx > JL_LM_CTX_MAX) return 0;

    uint32_t rings = groups * heads * depth;

    /* The declared shape must account for EXACTLY len bytes. A blob that is
     * close but not exact still parses; it just reads the wrong weights and
     * predicts confidently from them. */
    uint32_t want = 28u
                  + rings * NHIT_PACKED_BYTES     /* packed INT4 taps      */
                  + rings                         /* per-ring shifts       */
                  + vocab * NHIT_N                /* embedding             */
                  + vocab * read_dim              /* readout matrix        */
                  + vocab * 4u                    /* readout bias, INT32   */
                  + vocab;                        /* character table       */
    if (want != len) return 0;

    const uint8_t *p = blob + 28;
    const uint8_t *taps = p;                      p += rings * NHIT_PACKED_BYTES;
    const int8_t  *shifts = (const int8_t *)p;    p += rings;
    m_embed = (const int8_t *)p;                  p += vocab * NHIT_N;
    m_read_w = (const int8_t *)p;                 p += vocab * read_dim;
    m_read_b = (const int32_t *)p;                p += vocab * 4u;
    m_chars = p;

    /* Working state from the kernel heap. Before rung 9 these were static
     * arrays sized by #define, which is exactly what tied model size to the
     * kernel binary. */
    NhitKernel *k = (NhitKernel *)kmalloc(rings * sizeof(NhitKernel));
    int8_t *bundle = (int8_t *)kmalloc(NHIT_N);
    int8_t *ringbuf = (int8_t *)kmalloc(read_dim);
    int32_t *logits = (int32_t *)kmalloc(vocab * sizeof(int32_t));
    if (!k || !bundle || !ringbuf || !logits) {
        kfree(k); kfree(bundle); kfree(ringbuf); kfree(logits);
        return 0;
    }

    for (uint32_t r = 0; r < rings; r++) {
        k[r].shift = shifts[r];
        for (uint32_t b = 0; b < NHIT_PACKED_BYTES; b++)
            k[r].taps[b] = taps[r * NHIT_PACKED_BYTES + b];
    }

    m_groups = groups; m_heads = heads; m_depth = depth;
    m_vocab = vocab; m_read_dim = read_dim; m_rings = rings; m_ctx = ctx;
    m_kernels = k; m_bundle = bundle; m_ringbuf = ringbuf; m_logits = logits;
    m_bytes = len;
    loaded = 1;
    return 1;
}

/* Bundle the context into one ring: each character's INT8 embedding is rotated
 * by its position and the rotated vectors summed. Accumulation is INT32 so the
 * sum cannot wrap before saturation is applied deliberately. */
static void lm_bundle_ctx(const char *ctx) {
    int32_t acc[NHIT_N];
    for (uint32_t i = 0; i < NHIT_N; i++) acc[i] = 0;

    for (uint32_t p = 0; p < m_ctx; p++) {
        int idx = lm_index_of(ctx[p]);
        if (idx < 0) continue;          /* out of vocabulary: contributes nothing */
        const int8_t *e = &m_embed[(uint32_t)idx * NHIT_N];
        for (uint32_t i = 0; i < NHIT_N; i++) {
            /* Rotation by p on Z/N. This carries position -- rotating and
             * summing does not commute across different p, so the bundle
             * distinguishes "ab" from "ba". */
            acc[(i + p) & (NHIT_N - 1)] += (int32_t)e[i];
        }
    }

    /* Scale back into INT8. Dividing by the context length keeps the range
     * comparable to a single embedding, which is what the taps trained against. */
    for (uint32_t i = 0; i < NHIT_N; i++)
        m_bundle[i] = lm_sat8(acc[i] / (int32_t)m_ctx);
}

char jl_lm_predict(const char *ctx, int32_t *logit_out) {
    if (!loaded) return 0;

    lm_bundle_ctx(ctx);
    nhit_grouped(m_kernels, m_groups, m_heads, m_depth, m_bundle, m_ringbuf);
    nhit_readout(m_read_w, m_read_b, m_read_dim, m_vocab, 0,
                 m_ringbuf, m_logits);

    uint32_t best = nhit_argmax32(m_logits, m_vocab);
    if (logit_out) *logit_out = m_logits[best];
    return (char)m_chars[best];
}

void jl_lm_generate(const char *seed, char *out, uint32_t n) {
    char ctx[JL_LM_CTX_MAX];
    if (!loaded) { out[0] = 0; return; }

    /* Right-align the seed in the context window, padding with spaces. A seed
     * shorter than the window is the normal case, and left-aligning it would
     * put the useful characters at the wrong rotations entirely. */
    uint32_t len = 0;
    while (seed[len]) len++;
    for (uint32_t i = 0; i < m_ctx; i++) {
        uint32_t back = m_ctx - i;
        ctx[i] = (len >= back) ? seed[len - back] : ' ';
    }

    for (uint32_t k = 0; k < n; k++) {
        char c = jl_lm_predict(ctx, 0);
        out[k] = c;
        for (uint32_t i = 0; i + 1 < m_ctx; i++) ctx[i] = ctx[i + 1];
        ctx[m_ctx - 1] = c;
    }
    out[n] = 0;
}

int jl_lm_loaded(void) { return loaded; }
uint32_t jl_lm_groups(void) { return m_groups; }
uint32_t jl_lm_heads(void) { return m_heads; }
uint32_t jl_lm_depth(void) { return m_depth; }
uint32_t jl_lm_vocab(void) { return m_vocab; }
uint32_t jl_lm_bytes(void) { return m_bytes; }
uint32_t jl_lm_ctx(void) { return m_ctx; }
uint32_t jl_lm_taps(void) { return loaded ? m_rings * NHIT_N : 0; }

uint32_t jl_lm_macs(void) {
    /* Ring MACs plus the readout matrix. The readout is a real matrix and is
     * not free: at 64 groups it is 4096x64, comparable to the rings. */
    return loaded ? m_rings * NHIT_N * NHIT_N + m_read_dim * m_vocab : 0;
}
