# ADR-001: What JL/OS builds next

**Status:** Accepted — outcome recorded below
**Date:** 2026-08-04 (outcome 2026-08-05)
**Deciders:** JL

## Context

JL/OS boots on its own stage-1 loader from a hand-written ISO, runs a 256-gate
IDT with a 100 Hz PIT, and executes NHIT ring convolution — integer-only, no
FPU, no libc — as a kernel primitive. All of that is verified running.

The differentiator is narrow and real: **inference as a kernel primitive with no
runtime underneath it.** Nothing else does that. But the claim is currently weak
in exactly one place — the model barely predicts:

```
integer path 0.1583   vs   majority-class baseline 0.1304   =  +0.028
```

Three quantization hypotheses were tested this session. QAT held (+0.003 →
+0.028). INT8 taps and rescale-alignment were both refuted. The remaining
levers are not in the quantizer.

**The finding that reframes this decision:** the model is small for no
architectural reason. Stage-1 stages the kernel at `0x10000` and real-mode
addressing runs clean to the VGA hole at `0xA0000`:

```
usable staging span   589,824 bytes  (288 sectors)
current kernel flat    20,484 bytes
headroom                     29x     before the loader needs redesign
```

Weights ship as `.rodata` inside the flat image and are already loaded by the
existing DAP read. **We are using roughly 0.03% of the capacity the current
design supports.** N=64 / depth=3 / 192 taps was chosen early as a first
working shape and never revisited.

This also retires an assumption from the last planning pass: a memory manager
is *not* what blocks a larger model.

## Decision

Scale the model into the headroom that already exists, after freezing the
training corpus. Defer OS-depth work until the differentiator is either
demonstrated or falsified.

## Options Considered

### Option A: Scale the model into existing headroom

| Dimension | Assessment |
|---|---|
| Complexity | Low — no kernel or loader change |
| Cost | Hours |
| Risk | Low; reversible, git baseline exists |
| Answers the real question | **Yes** |

**Pros:** Uses capacity already built and paid for. Directly attacks the one
weak claim. Requires nothing new in the kernel — larger `.rodata`, larger `N`,
more depth. Failure is informative: if NHIT does not improve with 10–100× the
parameters, the premise itself is in question and we learn that cheaply.

**Cons:** Requires the corpus freeze first (see Consequences). `nhit.h` has
`NHIT_N` hardcoded at 64 with a power-of-two mask assumption; larger N must stay
a power of two.

### Option B: Memory manager and OS depth

| Dimension | Assessment |
|---|---|
| Complexity | High — parse Multiboot map, physical allocator, then paging |
| Cost | Days |
| Risk | Medium — touches the boot path |
| Answers the real question | No |

**Pros:** Genuine OS milestone. Eventually needed for dynamic model loading,
processes, and anything resembling userspace.

**Cons:** The premise that justified it — "model size is capped by `.bss`" — is
measurably false. It buys nothing for the differentiator until models exceed
~576 KB, which is 29× away. Building it now is infrastructure ahead of demand.

### Option C: Physical hardware validation

| Dimension | Assessment |
|---|---|
| Complexity | Low technically, high in unknowns |
| Cost | Hours, plus spare hardware |
| Risk | Low to the repo; the ISO is unchanged |
| Answers the real question | Partially |

**Pros:** The largest untested claim in the project. QEMU supplies a PS/2
controller, VGA text mode and a 16550 UART; a modern UEFI machine may supply
none. Everything measured so far is under TCG emulation.

**Cons:** Doesn't improve the differentiator. Needs a spare machine, Secure Boot
disabled, ideally a serial capture. Best done once, when there is something
worth demonstrating on it.

### Option D: Consolidate and stop

| Dimension | Assessment |
|---|---|
| Complexity | None |
| Cost | Hours |
| Risk | None |
| Answers the real question | No |

**Pros:** The repo is in a good state — boots, verified, committed, honestly
documented. A clean stopping point.

**Cons:** Leaves the central claim unproven at the exact moment the cheapest
experiment against it is available.

## Trade-off Analysis

The decision hinges on one question: **is NHIT weak because it is small, or
weak because the approach does not carry?**

Nothing built so far distinguishes those. A 192-tap model performing near
baseline is consistent with both. Option A separates them for hours of work and
no new kernel code, and it is the only option that does.

Option B was the previous plan and rests on a claim now measured false. Option C
tests a different axis and is better spent once there is a result worth showing.
Option D forecloses the cheapest available answer.

The asymmetry decides it: if scaling works, the differentiator becomes real and
everything downstream is justified. If scaling does not work, we learn that
before investing days in a memory manager to serve a model that was never going
to carry.

## Consequences

**Easier**
- The differentiator becomes demonstrable, or demonstrably wrong, quickly
- No kernel, loader or ISO change — the verified boot path stays frozen
- `git diff` against the committed baseline bounds any regression

**Harder**
- Larger weight blobs grow the ISO; kernel load time rises (irrelevant in QEMU,
  relevant on real hardware later)
- `NHIT_N` must remain a power of two — `(o-j) & NHIT_MASK` depends on it
- Beyond ~576 KB the stage-1 staging design must change (unreal mode or chunked
  loading). Not near, but it is the next hard ceiling and should be recorded.

**Must revisit**
- **The corpus is not frozen.** `load_text()` reads the project's own
  `*.md`/`*.c`/`*.h`, which changed throughout the session; baselines drifted
  0.1172 → 0.1304 → 0.1191 → 0.1194. No accuracy comparison is trustworthy
  until this is fixed. This is a prerequisite, not a nice-to-have.
- If a 10–100× larger NHIT does not beat baseline convincingly, revisit whether
  ring convolution is the right primitive at all.

## Outcome

Option A was taken. It answered the question it was chosen to answer.

**Scaling works.** The model is 1024× larger (192 → 196,608 taps) and the margin
over majority-class baseline improved roughly 3× on the integer path:

| path | accuracy | baseline | margin |
|---|---|---|---|
| PyTorch QAT | 0.3706 | 0.2799 | +0.0907 |
| C integer (kernel code) | 0.3228 | 0.2799 | **+0.0430** |

The integer figure is produced by `test_lm_host.c`, which compiles the actual
`nhit.c` and `jl_lm.c`. Earlier numbers in this repo came from Python; this one
does not, and it is the number the kernel gets.

**But the hypothesis behind the sweep was wrong in an instructive way.** The plan
was to sweep `NHIT_N` and depth. Neither turned out to be the lever:

1. **Averaging heads was the cap.** `nhit_multihead` collapsed all heads into one
   `NHIT_N` vector, so no matter how many taps fed it, 64 numbers reached the
   readout. Worse, without the ReLU the mean of *H* convolutions *is* a single
   convolution — heads were nearly a no-op by construction. Grouping (average
   within, concatenate across) fixed it. Groups peak at 16 and turn over by 64,
   where the INT8 readout starts to memorise.
2. **Data bound harder than parameters.** The same configuration scored +0.0525
   on 2M chars and +0.0907 on 8M. We have 17.9M and have used 8M.
3. **Shift calibration cost more than architecture.** A closed-form estimate
   measured +0.0113 from weights that empirical p99 calibration lifted to
   +0.0430 — a 4× swing with the model untouched.

**Two claims in this ADR were wrong and are corrected:**

- "usable staging span 589,824 bytes / 29× headroom" ignored that a real-mode
  INT 13h DAP buffer is `segment:offset` and a single transfer cannot cross a
  64 KiB boundary. The true ceiling was **64 KiB per read**, and the 196 KiB
  kernel failed to boot until stage-1 was changed to chunked reads. The 576 KB
  figure is now correct only *because* that change was made.
- The `ALIGN16` on `NhitKernel` was pure loss — 33 useful bytes padded to 48 for
  a vectorized path already rejected by measurement. Removed.

**Still not demonstrated:** the model is weak in absolute terms, has never run
on physical hardware, and the remaining 0.3706 → 0.3228 gap is the fixed-shift
approximation of training's dynamic scaling. MK-2 scheduling is still simulated.

## Action Items

1. [x] Freeze the training corpus — `corpus_frozen.txt`, 974 files, 17,859,972
       chars, sha256 `fb755a4b…`.
2. [x] Trustworthy reference point established on the frozen corpus.
3. [x] Swept — but on the *groups* axis, not `NHIT_N`, once head-averaging was
       identified as the real cap.
4. [x] Best configuration in the kernel. Flat image 196,612 bytes, `make verify`
       passes, boots under QEMU, inference runs before the HUD starts.
5. [x] Recorded above; marked Accepted.
6. [ ] Open items, in the order they now matter:
       - Train on the full 17.9M corpus — data is the measured binding
         constraint and we are using under half of it.
       - Close the 0.3706 → 0.3228 quantization gap by training against the
         kernel's fixed per-ring shift rather than calibrating it at export.
       - Option C (physical hardware). Now worth doing: there is something to
         demonstrate, and every number so far is under TCG emulation.
       - Option B (memory manager) remains unjustified — 196 KB of 576 KB used.
