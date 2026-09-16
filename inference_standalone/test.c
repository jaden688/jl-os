/* Standalone inference test harness */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "jl_lm.h"

/* Load file into memory */
uint8_t *load_file(const char *path, uint32_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = malloc(sz);
    if (!buf) {
        fprintf(stderr, "Failed to allocate %ld bytes\n", sz);
        fclose(f);
        return NULL;
    }

    size_t rd = fread(buf, 1, sz, f);
    fclose(f);

    if (rd != (size_t)sz) {
        fprintf(stderr, "Read %lld bytes, expected %lld\n", (long long)rd, (long long)sz);
        free(buf);
        return NULL;
    }

    *size_out = (uint32_t)sz;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_blob_path> [seed_text]\n", argv[0]);
        fprintf(stderr, "  Example: test mh_model.pt \"the \"\n");
        return 1;
    }

    const char *model_path = argv[1];
    const char *seed = argc > 2 ? argv[2] : "the ";

    printf("=== Standalone Inference Test ===\n");
    printf("Loading model: %s\n", model_path);

    uint32_t blob_size = 0;
    uint8_t *blob = load_file(model_path, &blob_size);
    if (!blob) return 1;

    printf("Blob size: %u bytes\n", blob_size);

    /* Debug: check header */
    const uint32_t *h = (const uint32_t *)blob;
    printf("Header magic: 0x%08x (expect 0x5449484E)\n", h[0]);
    if (blob_size >= 28) {
        printf("  groups=%u, heads=%u, depth=%u, vocab=%u, read_dim=%u, ctx=%u\n",
               h[1], h[2], h[3], h[4], h[5], h[6]);
    }

    if (!jl_lm_load(blob, blob_size)) {
        fprintf(stderr, "Failed to load model\n");
        free(blob);
        return 1;
    }

    printf("\nModel loaded:\n");
    printf("  Groups:  %u\n", jl_lm_groups());
    printf("  Heads:   %u\n", jl_lm_heads());
    printf("  Depth:   %u\n", jl_lm_depth());
    printf("  Vocab:   %u\n", jl_lm_vocab());
    printf("  Context: %u\n", jl_lm_ctx());
    printf("  Taps:    %u\n", jl_lm_taps());
    printf("  MACs:    %u per character\n", jl_lm_macs());

    printf("\nGenerating 32 characters from seed: \"%s\"\n", seed);

    char out[33];
    clock_t t0 = clock();
    jl_lm_generate(seed, out, 32);
    clock_t t1 = clock();

    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("Output: \"%s\"\n", out);
    printf("Time: %.3f seconds (%.1f ms/char)\n", elapsed, elapsed * 1000 / 32);

    free(blob);
    return 0;
}
