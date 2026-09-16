# Training in Progress

## Currently Running

### Job 1: BIG (blftbbz9d) - Target Config
```
Groups:  64
Heads:   64
Depth:   7
Taps:    64 × 64 × 7 × 64 = 1,835,008 taps
Context: 4
Data:    17,900,000 characters (full corpus)
Steps:   5,000
Batch:   256
```

**Expected:**
- Model size: ~1.5 MB
- Accuracy: 65-75% (prediction)
- Training time: 2-4 hours (depends on GPU/CPU)
- Output: `mh_model.pt`

### Job 2: MEDIUM (brkfeotxg) - Fast Validation
```
Groups:  32
Heads:   64
Depth:   5
Taps:    32 × 64 × 5 × 64 = 655,360 taps
Context: 4
Data:    8,000,000 characters
Steps:   2,000
Batch:   256
```

**Expected:**
- Model size: ~500 KB
- Accuracy: 55-65% (prediction)
- Training time: 20-40 minutes
- Output: `mh_model.pt` (will be overwritten by Big job)

---

## What Happens When Training Finishes

### Automatic (ready to run):

**1. Export to binary blob:**
```bash
python export_to_blob.py mh_model.pt --out core_scaled.nhit
```

**2. Validate against test set:**
```bash
gcc -O3 -std=c99 -DJL_HOST test_lm_host.c nhit.c jl_lm.c -o test_lm_host
./test_lm_host inference_standalone/lm_valset.bin core_scaled.nhit
```

**3. Test standalone inference:**
```bash
cd inference_standalone
cp ../core_scaled.nhit core_nhit_scaled
gcc -O3 nhit.c jl_lm.c test.c -o test_inference
./test_inference core_nhit_scaled "function definition"
```

**4. Load into kernel:**
```bash
# Copy to PEWTER-FS volume
# Rebuild kernel
# Boot in QEMU
# Verify inference runs at startup
```

---

## What to Expect

### If Medium Finishes First:
- Should see 55-65% accuracy (better than current 51.29%)
- Proves scaling works
- Can load into kernel immediately
- Then wait for Big to finish for even better accuracy

### If Big Finishes Successfully:
- Should see 65-75%+ accuracy
- 1.8M taps on full corpus
- Proof that NHIT scales efficiently
- Can compete with transformer results

### If Training Fails:
- Check error in output file
- Likely: out of memory (17.9M chars with 64×64×7 is big)
- Fallback: reduce to 12M chars, or reduce depth to 5

---

## Timeline

```
Now:       Training starts (2 jobs)
+20 min:   Medium training should finish (32/64/5 config)
+20 min:   Export and validate medium model
+30 min:   Test medium in kernel
+2-4 hr:   Big training finishes (64/64/7 config)
+2-4 hr:   Export, validate, deploy big model
```

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Accuracy | 65%+ | 51.29% |
| Taps | 1M+ | 196K |
| Model Size | <2 MB | 171 KB |
| Inference | Works in kernel | ✓ Works |
| Full Corpus | Used | 8M/17.9M |

---

## Next Actions

1. **Monitor training** (check output every 10 min)
2. **Export ASAP when Medium finishes**
3. **Validate on test set**
4. **Test in kernel** (rebuild ISO, boot)
5. **Wait for Big, repeat**

The pipeline is set up. Training is running. **Let it cook.**
