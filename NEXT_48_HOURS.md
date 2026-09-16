# NEXT 48 HOURS: Scale NHIT

## Status Right Now
- ✅ Inference code isolated and tested
- ✅ Bootloader, filesystem, scheduler working
- ✅ Current model: 196 KB, 51.29% accuracy
- ✅ Full corpus available: 17.9M chars (frozen)
- ❌ Training pipeline not optimized
- ❌ Only using 8M/17.9M corpus
- ❌ Quantization gap not closed (0.5129 vs 0.3706)

---

## HOUR 0-4: UNDERSTAND THE GAP

### The Quantization Problem
```
PyTorch (float): 0.3706 accuracy
Quantized (INT4): 0.5129 accuracy  ← This looks backwards!

Wait, 0.5129 > 0.3706. Let me re-read...

Actually:
- PyTorch QAT: 0.3706
- C integer path (kernel): 0.3228  ← This is what matters
- Gap: 0.3706 - 0.3228 = 0.0478 (13% loss to quantization)

Real baseline: 0.2799
So:
- QAT margin: +0.0907
- Integer margin: +0.0430  ← Only half the margin due to quantization
```

**What changed in CTX=4 commit?**
- Context encoding was breaking the signal
- CTX=4 gave +0.2599 margin in training
- But integer path implementation has fixed-shift that QAT training never saw
- Gap: dynamic scaling (QAT) vs fixed p99 shift (integer)

### The Real Questions
1. **Can we close the 0.0478 gap?** (dynamic vs fixed shift)
   - Train targeting integer path directly
   - Calibrate shifts empirically, not post-hoc

2. **What does "scale" actually mean for NHIT?**
   - More heads? Cost: O(heads) more taps
   - More groups? Cost: O(groups) more output dim (→ readout matrix grows)
   - More depth? Cost: O(depth) more computation
   - Bigger rings (N=128)? Cost: O(N²) compute, defeats parameter efficiency

---

## HOUR 4-16: MEASURE THE CURRENT STATE

### Run these NOW:

**1. Verify what you have:**
```bash
cd inference_standalone
./test_inference core.nhit "function definition"
```

**2. Score on full validation set:**
```bash
cd ..
gcc -O3 -std=c99 -DJL_HOST test_lm_host.c nhit.c jl_lm.c -o test_lm_host
./test_lm_host inference_standalone/lm_valset.bin inference_standalone/core.nhit
```

**3. Profile the hot path:**
```bash
# Add timing to nhit.c to measure per-ring cost
# Print: time per character, which layers are slow
gcc -O3 -g -pg inference_standalone/nhit.c ...
./test_inference core.nhit "test"
gprof test_inference | head -20
```

**Output you need to see:**
- Current accuracy (should be ~0.5129)
- Which part of NHIT is slowest
- How many MACs per character (should be 12.6M)

---

## HOUR 16-32: UNDERSTAND SCALING TRADE-OFFS

### Option 1: More Groups (Current Path)
```
Current: 16 groups × 64 heads × 3 depth × 64 N = 196K taps
Scaled:  128 groups × 64 heads × 3 depth × 64 N = 1.57M taps

Cost:
- Taps: +8×
- Output dim: 64 → 128×64 (each group outputs 64 values)
- Readout matrix: 64 × 1024 → 64 × 8192 = +8× size
- Total model: 171 KB → 1.4 MB

Benefit:
- More independent representations
- Should improve accuracy (measured +0.0261 vs +0.0176 for 16 vs 1 group)
```

### Option 2: More Heads Per Group
```
Current: 16 groups × 64 heads × 3 depth
Scaled:  16 groups × 256 heads × 3 depth = 786K taps

Cost:
- Taps: +4×
- Output dim: same (16 × 64)
- Readout: same size
- Total model: 171 KB → 700 KB

But: Heads averaging collapses representation
- More heads = diminishing returns
- ADR showed: 1024 heads → +0.0176, 16 groups of 64 → +0.0261
- More heads without more groups = wasted taps
```

### Option 3: More Depth
```
Current: depth=3 (3 convolution layers)
Scaled:  depth=7

Cost:
- Taps: +2.3×
- Compute: +2.3× MACs per character
- Model size: ~2× (add more weight storage)

Benefit:
- More nonlinearity = better feature learning
- Could reduce need for more groups/heads
```

### Option 4: Bigger Rings (N=128)
```
Current: N=64
Scaled:  N=128

Cost:
- Taps: same count but 128 per ring instead of 64
- Compute: +4× per ring (N² operation)
- Model size: +2×

Benefit:
- Larger receptive field
- But defeats parameter efficiency premise
- **Probably wrong direction**
```

---

## HOUR 32-48: DECIDE AND IMPLEMENT

### My Recommendation: **Option 1 + Option 3**

**Do this:**
1. Increase groups: 16 → 64 (trade accuracy for model size)
2. Increase depth: 3 → 7 (add nonlinearity)
3. Keep heads: 64 (proven working)
4. Keep N: 64 (proven optimal)

**Result:**
- Groups×heads×depth×N = 64×64×7×64 = 1.8M taps
- Output dim: 64×64 = 4096 (readout: 64 × 4096 = 256 KB)
- Total model: ~1.2 MB
- Estimated accuracy: 65-70% (guess based on scaling trends)
- Fits in kernel memory (disk-loaded)

**Why this combo:**
- Groups scale cleanly (independent representations)
- Depth adds real nonlinearity (not just averaging)
- Both have precedent in your code (ADR measured groups, you use depth)
- Avoids ring resizing (N=64 is proven)

### How to Implement

**Step 1: Train this config**
```
groups=64, heads=64, depth=7, N=64
vocab=64, ctx=4
Run on full 17.9M corpus (not 8M)
```

**Step 2: Export to binary blob**
```
Magic: 0x5449484E ("NHIT")
Header: groups(64), heads(64), depth(7), vocab(64), read_dim(4096), ctx(4)
Weights: packed INT4 taps + shifts + embeddings + readout
```

**Step 3: Test**
```
./test_lm_host lm_valset.bin exported.nhit
# Should show accuracy > 0.5129
```

**Step 4: Deploy to kernel**
```
Copy exported.nhit to PEWTER-FS as /models/core.nhit
Boot kernel
Verify inference runs at startup
```

**Step 5: Iterate**
```
If accuracy still < 65%:
  - Train longer
  - Increase depth more (→ 10 layers)
  - Increase groups more (→ 128)
  - Train on different data subset

If accuracy > 65%:
  - You've won
  - Move to Phase 2: even bigger
```

---

## Tools You Have Ready

```bash
# Compile inference
cd inference_standalone
gcc -O3 nhit.c jl_lm.c test.c -o test_inference

# Test on validation
gcc -O3 -DJL_HOST test_lm_host.c nhit.c jl_lm.c -o test_lm_host

# Train (incomplete, need to finish)
python train_scale.py  # Has skeleton, needs circular convolution fix + export

# Boot kernel
make run-nographic  # QEMU
# or on real hardware after ISO build
```

---

## What Blocks You Right Now

1. **Training script is incomplete**
   - Circular convolution is naive (slow)
   - Export to .nhit format not implemented
   - Could use existing `train_multihead.py` instead

2. **You haven't trained on FULL corpus**
   - 17.9M chars available
   - Currently using 8M
   - Rerun training on all

3. **Quantization calibration is heuristic**
   - p99 shift is post-hoc
   - Could train to target fixed shifts directly
   - Worth trying after you scale

---

## The Move RIGHT NOW

**Don't overthink. Do this order:**

1. **TODAY**: Run `train_multihead.py` with:
   - `groups=64, heads=64, depth=7`
   - Full 17.9M corpus
   - Same training setup as before

2. **TODAY+1**: Export result to .nhit blob

3. **TODAY+2**: Test with `test_lm_host`, measure accuracy

4. **TODAY+3**: If > 55%, load into kernel and boot. If < 55%, increase groups to 128 and retry.

**That's it.** One variable at a time, measure, iterate.

---

## How You Know It's Working

- Accuracy increases from 0.5129 → 0.55+ (even 51.3 → 52% is progress)
- Model is 2-5 MB (not bloated)
- Kernel still boots and runs inference
- No quantization errors (test_lm_host scores correctly)
- Execution time increases but stays reasonable (<10ms/char on CPU)

Once you hit 65%+, you've proven:
- NHIT scales
- Integer-only inference works
- Competes with transformers at lower weight count
- Can run in kernel space efficiently

Then expand further.

---

## Keep This Energy

You have everything you need. The OS is built. The inference is proven. Now just make it **bigger and better**. Not trying something new. Not rethinking. Just: **scale NHIT until it works**.

Go.
