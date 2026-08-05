# ADR-001: What JL/OS builds next

**Status:** Proposed
**Date:** 2026-08-04
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

## Action Items

1. [ ] Freeze the training corpus to a committed file; make `load_text()` read
       only that. Prerequisite for every measurement below.
2. [ ] Re-measure the current 192-tap model against the frozen corpus to get a
       trustworthy reference point.
3. [ ] Sweep `NHIT_N` ∈ {64, 128, 256} × depth ∈ {3, 6} on the frozen corpus,
       host-side only — no kernel changes while sweeping.
4. [ ] Take the best configuration into the kernel; confirm the flat image stays
       under 576 KB; rebuild, `make verify`, boot, read `argmax` off `0xB8000`.
5. [ ] Record the result in this ADR and mark it Accepted or Superseded.
6. [ ] Only then re-open Option B (memory manager) or Option C (hardware).
