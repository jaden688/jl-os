# Quick Testing Guide

## Build and Run

```bash
# One command to build and test
gcc -O3 -Wall -std=c99 nhit.c jl_lm.c test.c -o test_inference && ./test_inference core.nhit "the "
```

## Test Different Inputs

```bash
# Test with different seeds
./test_inference core.nhit "hello"
./test_inference core.nhit "function"
./test_inference core.nhit "    "  # spaces
```

## Measure Performance

```bash
# Time it (will show ms/char)
time ./test_inference core.nhit "the "
```

## Benchmark Against Baseline

Use `test_lm_host.c` to score against validation set:

```bash
gcc -m32 -O2 test_lm_host.c nhit.c jl_lm.c -o test_lm_host
./test_lm_host lm_valset.bin core.nhit
```

This gives accuracy metrics:
- Majority-class baseline
- Model accuracy
- Margin over baseline

## Common Tests

### 1. Verify Model Loads
```bash
./test_inference core.nhit "x"
# Should print model shape and generate 32 chars
```

### 2. Test Determinism
```bash
./test_inference core.nhit "seed" > out1.txt
./test_inference core.nhit "seed" > out2.txt
diff out1.txt out2.txt
# Should be identical
```

### 3. Test Context Dependency
```bash
# Same next char but different context
./test_inference core.nhit "the "    # "the " context
./test_inference core.nhit "cat "    # different context
# Should generate different output
```

### 4. Profile Hotspots
```bash
# Compile with profiling
gcc -O3 -pg nhit.c jl_lm.c test.c -o test_inference
./test_inference core.nhit "the "
gprof test_inference gmon.out | head -30
```

## Modify for Your Tests

Edit `test.c` to:
- Change seed text
- Generate more/fewer characters
- Add detailed timing
- Collect statistics
- Test specific code paths

Example: measure just convolution:
```c
// In test.c after jl_lm_load():
nhit_prof_reset();
for (int i = 0; i < 1000; i++) {
    jl_lm_predict(ctx, NULL);
}
uint64_t macs = nhit_mac_count() * 1000;
printf("Total MACs: %llu\n", macs);
```

## Export New Model

Once you train a new model, export it:

```python
# In your training script
import torch

model_state = {...}  # your trained model
groups, heads, depth = ...
vocab, read_dim, ctx = ...

# Serialize to blob format that jl_lm_load() expects
header = struct.pack("<7I",
    0x5449484E,  # magic "NHIT"
    groups, heads, depth, vocab, read_dim, ctx
)

# Then append weights in the right order:
# - kernels (INT4 taps, packed)
# - shifts (INT8)
# - embeddings (INT8)
# - readout matrix (INT8)
# - readout bias (INT32)
# - char table

with open("my_model.nhit", "wb") as f:
    f.write(header)
    f.write(taps_packed)
    f.write(shifts)
    # ... etc
```

Then test immediately:
```bash
./test_inference my_model.nhit "test"
```

## Debug Failed Load

If `jl_lm_load()` fails:

1. Check magic bytes:
   ```c
   hexdump -C core.nhit | head -2
   // Should start with: 4e 48 49 54 (NHIT in little-endian)
   ```

2. Verify size calculation:
   ```c
   // In jl_lm_load, the `want` variable should equal `len`
   // If not, the weight layout is wrong
   ```

3. Add debug output:
   ```c
   // Edit jl_lm.c:
   fprintf(stderr, "want=%u, len=%u\n", want, len);
   ```

---

**You can now iterate on inference without touching the kernel.**
