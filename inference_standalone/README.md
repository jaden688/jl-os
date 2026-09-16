# Inference Standalone

**Isolated inference library** — all the working parts of the kernel's language model, extracted and ready to test independently.

## What You Have Here

The complete inference pipeline, decoupled from the OS:

- **nhit.c/h** — Ring convolution (the core compute primitive)
  - `nhit_apply()` — single ring kernel
  - `nhit_grouped()` — grouped multi-head rings
  - `nhit_readout()` — vocabulary matrix
  - All INT8 arithmetic, no FPU

- **jl_lm.c/h** — Language model wrapper
  - HRR context bundling (CTX=4)
  - Model blob loading and parsing
  - `jl_lm_generate()` — autoregressive text generation
  - `jl_lm_predict()` — single-step prediction

- **jl_core.h** — Minimal type definitions
  - Redirects kmalloc/kfree to stdlib malloc/free
  - No kernel dependencies

- **test.c** — Standalone test harness
  - Loads model blob from disk
  - Runs inference
  - Measures performance

## Current Model

**core.nhit** (171 KB):
- 16 groups × 64 heads × 3 depth layers
- 196,608 INT4 taps (ring convolution weights)
- Vocab: 64 characters
- Context: 4 characters
- Integer path accuracy: **0.5129** (margin +0.2330 over baseline)
- MACs per character: 12,648,448

## Compile

```bash
gcc -O3 -Wall -Wextra -std=c99 -g -c nhit.c -o nhit.o
gcc -O3 -Wall -Wextra -std=c99 -g -c jl_lm.c -o jl_lm.o
gcc -O3 -Wall -Wextra -std=c99 -g -c test.c -o test.o
gcc -O3 -o test_inference nhit.o jl_lm.o test.o
```

Or use the Makefile:
```bash
make
make run
```

## Run

```bash
./test_inference core.nhit "the "
```

Output:
```
=== Standalone Inference Test ===
Loading model: core.nhit
Blob size: 171356 bytes
Header magic: 0x5449484e (expect 0x5449484E)
  groups=16, heads=64, depth=3, vocab=64, read_dim=1024, ctx=4

Model loaded:
  Groups:  16
  Heads:   64
  Depth:   3
  Vocab:   64
  Context: 4
  Taps:    196608
  MACs:    12648448 per character

Generating 32 characters from seed: "the "
Output: "elements
struct a session       "
Time: 0.137 seconds (4.3 ms/char)
```

## What You Can Do Now

1. **Test the current model**
   - Run inference without recompiling the kernel
   - Measure performance on different inputs
   - Profile the hot paths

2. **Experiment with different architectures**
   - Replace `nhit_grouped()` with LSTM/RNN/Transformer
   - Keep the same blob loading and serialization
   - Keep the context bundling (HRR) or replace it
   - Measure on the same validation set

3. **Iterate on quantization and training**
   - Export new model weights from Python
   - Test immediately without kernel rebuild
   - No boot, no emulation, no ISO rebuild

4. **Benchmark**
   - Performance scales with the algorithm
   - MAC counts are precise
   - Latency measured in standalone mode

## Model Blob Format

Header (28 bytes):
```c
uint32_t magic      // 0x5449484E ("NHIT")
uint32_t groups
uint32_t heads
uint32_t depth
uint32_t vocab
uint32_t read_dim   // groups * NHIT_N (always 64 * groups)
uint32_t ctx        // context window size
```

Then (byte-aligned):
```
rings * NHIT_PACKED_BYTES  // INT4 taps (2 per byte)
rings                       // per-ring shift amounts (INT8)
vocab * NHIT_N              // character embeddings (INT8)
vocab * read_dim            // vocabulary readout matrix (INT8)
vocab * 4                   // readout bias (INT32)
vocab                       // character lookup table
```

Total must match exactly — off-by-one is silently wrong output, not a parse error.

## Next Steps

To run a different model type:
1. Keep the blob loader and serialization
2. Write your inference code (LSTM, Transformer, etc.)
3. Export weights from training
4. Test here before deploying to kernel

To get better accuracy:
1. Train on full 17.9M character corpus (currently using 8M)
2. Close the 0.5129 → 0.3706 quantization gap (fixed-point vs dynamic scaling)
3. Swap in different context encoding or ring convolution variant

## Files Reference

- `nhit.c/h` — Ring convolution primitive (do not modify for testing)
- `jl_lm.c` — Model loader and inference loop (can replace with different architecture)
- `jl_lm.h` — Public API (defines what kernel expects)
- `jl_core.h` — Type definitions (minimal, no kernel deps)
- `test.c` — Test harness (modify as needed for your tests)
- `core.nhit` — Current trained model weights
- `lm_valset.bin` — Validation set (used by test_lm_host.c)
- `Makefile` — Build rules

---

**This is the inference library decoupled from the OS.** The kernel infrastructure (bootloader, filesystem, scheduler, paging) is done. This is where you iterate on the model.
