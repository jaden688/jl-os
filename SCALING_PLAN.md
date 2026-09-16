# NHIT-First Inference OS: Full Scale Plan

## Vision
Build a **production-grade operating system designed from the ground up for neural inference**, using NHIT (ring convolution) as the fundamental compute primitive instead of matrix multiply. Not minimal. Not barely-works. **Full expansion**.

---

## Current State → Target State

### Now:
- 196 KB model, 51% accuracy
- 36 KB kernel
- CTX=4 context window
- 8M char training

### Target (Phase 1 - 6 months):
- **10M parameter model**, 70%+ accuracy
- Full OS stack (scheduler ✓, filesystem ✓, memory ✓, I/O ✓)
- Proven on real hardware
- Multiple NHIT variants (not just one architecture)

### Future (Phase 2+):
- 100M+ parameter models
- Multi-model inference
- Real userspace programs
- Network I/O
- Model serving infrastructure
- Competitive with transformer stacks on efficiency metrics

---

## Immediate Work (Next 2-4 weeks)

### 1. Scale NHIT (Right Now)
**Current**: 16 groups × 64 heads × 3 depth = 196K taps
**Target**: 64-256 groups × 128 heads × 5-7 depth = 2-10M taps

Steps:
- [ ] Train NHIT on full 17.9M corpus (not 8M)
- [ ] Sweep: groups, heads, depth for optimal accuracy
- [ ] Close quantization gap (0.5129 → 0.3706)
- [ ] Target: 65%+ accuracy on validation set

**Constraints to solve:**
- Readout matrix grows with groups (64 × groups×64)
- At 256 groups: 64 × 16384 = 1 MB readout
- Total model: ~5MB at that scale
- Still fits in kernel memory (have 576KB available in stage-1, but switched to disk-loading)

**Plan:**
```
NHIT_N = 64 (fixed, ring size)
groups = 32-256 (variable, independent representations)
heads = 64-128 (variable, depth per group)
depth = 5-7 (more nonlinearity)
taps = groups * heads * depth * N = variable
```

### 2. Training Pipeline (Parallel)
- [ ] Run train_multihead.py on full corpus
- [ ] Implement dynamic shift calibration (not fixed p99)
- [ ] Export to binary blob format
- [ ] Validation loop: train → export → test_lm_host → measure

### 3. Standalone Testing (Proven)
- [ ] inference_standalone/ already compiles and runs
- [ ] Add benchmarking suite
- [ ] Profile hot paths in NHIT
- [ ] Measure MACs, latency, energy per character

### 4. Integration to Kernel
- [ ] Load larger models from PEWTER-FS
- [ ] Boot-time inference runs 100 characters (not 32)
- [ ] Time it end-to-end
- [ ] Verify on QEMU

---

## NHIT Expansion Ideas (Not Transformers)

**Current limitation**: CTX=4 (HRR capacity), ring convolution averaging

### Option A: Deeper Stacks
- More depth layers (3 → 7-10)
- Different nonlinearities (ReLU → Gelu, learnable)
- Skip connections between layers

### Option B: Richer Context Encoding
- Replace HRR with learned projection?
- Attention-like weighting over context?
- (NOT transformer attention, something novel)

### Option C: Multiple Ring Sizes
- Not just N=64
- N=32, 64, 128 in parallel
- Different scales for different patterns

### Option D: Grouped Rings with Cross-Group
- Current: groups are independent
- Add sparse connections between groups
- Learn which groups should communicate

### Option E: Hierarchical NHIT
- Rings operating at different context lengths
- Fast path (local patterns) + slow path (long context)
- Route through hierarchy

**Do ONE of these properly**, don't scatter.

---

## Hardware Validation (Phase 1.5)
- [ ] Spare x86 machine (used laptop OK)
- [ ] Disable Secure Boot
- [ ] Boot the ISO
- [ ] Verify: VGA output, serial log, inference runs
- [ ] Time it on real hardware vs QEMU

---

## Scaling Milestones

| Milestone | Model Size | Accuracy | Status |
|-----------|-----------|----------|--------|
| Current | 196 KB | 51.29% | ✓ Works |
| **M1: Scale NHIT** | 2 MB | 65%+ | → Next |
| M2: Expand OS | 5 MB | 70%+ | TBD |
| M3: Multi-Model | 10+ MB | 75%+ | TBD |
| M4: Production | 50-100 MB | 80%+ | TBD |

---

## Why Not Transformers?

Transformers have:
- O(n²) attention (context scales quadratically)
- Requires FPU or careful INT16 (you use INT4)
- 4+ billion parameters for decent quality
- Hardware-specific optimizations (CUDA, TPU)

NHIT has:
- O(n) rings (linear in parameters)
- Integer-only (INT4 taps, INT8 activations)
- Proven to tie LSTM, beat transformer on this corpus at 1/4 the weight
- Fits in kernel space, predictable performance
- **Scalable on your terms, not someone else's research**

You built it. Own it. Make it work at scale.

---

## What You Have Ready

✓ Bootloader + ISO (proven, works)
✓ Filesystem (PEWTER-FS, read/write tested)
✓ Preemptive scheduler (real task switching)
✓ Memory management (PMM, paging, heap)
✓ ATA driver (disk I/O)
✓ Inference pipeline (nhit.c, jl_lm.c, tested)
✓ Standalone testing (inference_standalone/, compiles & runs)
✓ Training data (17.9M chars, frozen)

**Missing:**
✗ Scaled model training (using 8M of 17.9M)
✗ Better quantization (fixed-shift vs dynamic)
✗ Real hardware validation
✗ Userspace/syscalls (not needed for Phase 1)

---

## First Move

**THIS WEEK:**
1. Run `train_multihead.py` on FULL 17.9M corpus
2. Sweep groups={32,64,128}, heads={64,128}, depth={5,7}
3. Export each trained variant to .nhit blob
4. Test each with `test_lm_host` on lm_valset.bin
5. Pick winner (highest accuracy)
6. Load into kernel, verify boots and runs

**Target outcome**: 65%+ accuracy model, ~2-5 MB, proven working in kernel.

Then iterate from there.

---

## You Already Won

The hard part (proving inference works in ring 0 with integer-only NHIT) is done. Everything after this is scaling what you built. Not research. Not "does this concept work". Just **make it bigger and better**.

NHIT is yours. Make it compete.
