/* ============================================================================
 * JL OS :: host harness for the kernel language model
 * ============================================================================
 * Compiles and runs THE ACTUAL KERNEL SOURCES -- nhit.c and jl_lm.c, unmodified
 * -- against the same validation windows PyTorch scored. This is the only
 * measurement that says what the kernel will really do; a Python simulation of
 * the integer path would be a reimplementation, and a reimplementation is
 * exactly where a discrepancy would hide.
 *
 * It exists because the training-to-kernel gap is where this project has lost
 * accuracy before: post-training quantization once cost 0.2397 -> 0.1200, and
 * the per-ring shift calibration is a heuristic applied at export time that
 * training never sees. If the number this prints does not match the number
 * PyTorch printed, the export is wrong -- not the model.
 *
 * Host-only. Nothing here is linked into the kernel: it uses stdio and malloc,
 * which is precisely why it is a separate translation unit with its own main
 * and is never added to OBJS.
 *
 *   cc -m32 -O2 -DJL_LM_HOST_TEST test_lm_host.c nhit.c jl_lm.c -o test_lm_host
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jl_lm.h"
#include "nhit_weights_mh.h"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "lm_valset.bin";

    if (!jl_lm_selftest()) {
        fprintf(stderr, "FAIL: jl_lm_selftest -- weight header disagrees with "
                        "nhit.h (NHIT_N, vocab or table sizes)\n");
        return 1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FAIL: cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)n + 1);
    if (!text || fread(text, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "FAIL: short read on %s\n", path); return 1;
    }
    text[n] = 0;
    fclose(f);

    /* Majority-class baseline computed on this same slice. Reporting accuracy
     * without it is meaningless -- a model that always predicts ' ' scores
     * around 0.23 on this corpus. */
    long freq[256] = {0};
    for (long i = 0; i < n; i++) freq[(unsigned char)text[i]]++;
    int major = 0;
    for (int i = 1; i < 256; i++) if (freq[i] > freq[major]) major = i;

    long correct = 0, major_correct = 0, total = 0;
    for (long i = 0; i + JL_LM_CTX + 1 <= n; i++) {
        char target = text[i + JL_LM_CTX];
        char pred = jl_lm_predict(&text[i], 0);
        if (pred == target) correct++;
        if ((unsigned char)target == (unsigned char)major) major_correct++;
        total++;
    }

    double acc = (double)correct / (double)total;
    double base = (double)major_correct / (double)total;

    printf("windows scored      %ld\n", total);
    printf("model shape         groups=%d heads=%d depth=%d N=%d vocab=%d\n",
           MH_GROUPS, MH_HEADS, MH_DEPTH, NHIT_N, MH_VOCAB);
    printf("taps                %u\n", jl_lm_taps());
    printf("MACs per inference  %u\n", jl_lm_macs());
    printf("\n");
    printf("majority baseline   %.4f  (char 0x%02x)\n", base, major);
    printf("INTEGER path        %.4f\n", acc);
    printf("margin              %+.4f\n", acc - base);

    /* A model that has collapsed to a constant scores near baseline while
     * looking like it works. Count distinct predictions so that failure is
     * visible rather than inferred. */
    int seen[256] = {0}, distinct = 0;
    for (long i = 0; i + JL_LM_CTX + 1 <= n && i < 4096; i++) {
        unsigned char p = (unsigned char)jl_lm_predict(&text[i], 0);
        if (!seen[p]) { seen[p] = 1; distinct++; }
    }
    printf("distinct predictions %d of %d vocab %s\n", distinct, MH_VOCAB,
           distinct <= 2 ? "  <-- COLLAPSED, accuracy is not meaningful" : "");

    /* Show the model actually continuing text, since an accuracy figure alone
     * has never been convincing evidence that inference works. */
    char out[81];
    jl_lm_generate("the ", out, 80);
    printf("\ngreedy continuation of \"the \":\n  ");
    /* Escaped, not printed raw: a model that emits only newlines produces
     * blank output that reads as "nothing happened" when it is in fact a
     * specific and diagnosable failure. */
    for (int i = 0; out[i]; i++) {
        unsigned char c = (unsigned char)out[i];
        if (c == '\n') printf("\\n");
        else if (c == '\t') printf("\\t");
        else if (c < 32 || c > 126) printf("\\x%02x", c);
        else putchar(c);
    }
    printf("\n");

    free(text);
    return 0;
}
