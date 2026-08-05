"""Train a multi-head NHIT model on the frozen corpus.

Scaling direction, and it is counter-intuitive: a ring of size N holds N taps
but costs N^2 MACs. Parameters grow linearly, compute quadratically. So at a
fixed compute budget, MANY SMALL rings beat few large ones -- N=64 yields
roughly 16x the parameters of N=1024 for the same arithmetic.

Heads are independent: each sees the same input, runs its own depth-D stack,
and the outputs are averaged. No cross-head dependencies, fixed shapes
throughout, which is what nhit_multihead() executes in the kernel.

QAT throughout, since post-training quantization was measured to cost most of
the accuracy (0.2397 float -> 0.1200 integer). Dynamic activation scaling is
used deliberately: forcing training to use the kernel's fixed power-of-two
shift was tested and made things WORSE (0.1583 -> 0.1350), so the constraint is
left off during training and paid for at export.

Corpus is frozen (corpus_frozen.txt, sha256 fb755a4b...). Every earlier number
in this project was measured against a corpus that changed between runs; those
comparisons were not sound. These are.

    python train_multihead.py --heads 1024 --depth 3
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "corpus_frozen.txt"
N = 64          # ring size; must match NHIT_N and stay a power of two

# Context length is a VARIABLE, not a constant, because it is the thing under
# test. The bundle superposes CTX characters into N INT8 values by rotation;
# HRR capacity goes roughly as N/(4 ln N), which at N=64 is about 4 items. If
# that bound is what limits the model, then SHORTENING the context should
# IMPROVE accuracy -- a backwards prediction, and the reason this is a real
# test rather than a confirmation.
CTX = 16


class RoundSTE(torch.autograd.Function):
    """round() forward, identity backward -- straight-through estimator."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def fake_quant_taps(taps: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Symmetric per-ring INT4, on the exact grid nhit_tap() decodes."""
    qmax = 2 ** (bits - 1) - 1
    peak = taps.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    step = peak / qmax
    return RoundSTE.apply(taps / step).clamp(-(qmax + 1), qmax) * step


def fake_quant_act(x: torch.Tensor) -> torch.Tensor:
    """INT8 activations with saturation, matching sat8()."""
    step = (x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0)
    return RoundSTE.apply(x / step).clamp(-128, 127) * step


class MultiHeadNhit(nn.Module):
    """Grouped multi-head NHIT.

    Heads within a group are averaged; groups are CONCATENATED. This matters
    more than the head count: averaging every head into one N-vector caps the
    representation at N=64 numbers no matter how many taps feed it, so heads
    past the first few buy nothing. Groups widen the readout to groups*N.

    Kernel mapping is unchanged -- nhit_multihead() is called once per group,
    each writing N bytes into a groups*N output buffer.
    """

    def __init__(self, vocab: int, heads: int, depth: int, groups: int = 1):
        super().__init__()
        self.groups, self.heads, self.depth = groups, heads, depth
        self.embed = nn.Embedding(vocab, N)
        # [groups, heads, depth, N] -- group-major, then head-major, matching
        # the order the kernel walks its NhitKernel array.
        self.taps = nn.Parameter(torch.randn(groups, heads, depth, N) * 0.3)
        self.head = nn.Linear(groups * N, vocab)

    def forward(self, idx):
        # The embedding and readout ship in the kernel too, so they must be
        # quantized during training. Leaving them float would measure an
        # accuracy the integer kernel cannot reproduce -- exactly the gap that
        # cost 0.2397 -> 0.1200 when quantization was applied post-hoc.
        e = F.embedding(idx, fake_quant_act(self.embed.weight))
        # Position enters as a rotation on the ring. Mean-pooling would destroy
        # character order, which is the whole task.
        x = torch.stack([torch.roll(e[:, p], shifts=p, dims=-1)
                         for p in range(e.shape[1])], dim=1).sum(dim=1)
        x = fake_quant_act(x)                       # [B, N]

        # Broadcast one input across every group and head; they never interact.
        h = x[:, None, None, :].expand(-1, self.groups, self.heads, -1)
        for d in range(self.depth):
            t = fake_quant_taps(self.taps[:, :, d, :])         # [G, H, N]
            # Circular convolution as spectral multiplication on Z/N.
            h = torch.fft.ifft(torch.fft.fft(h, dim=-1)
                               * torch.fft.fft(t, dim=-1).unsqueeze(0)).real
            h = fake_quant_act(h)
            if d + 1 < self.depth:
                h = F.relu(h)
        # Mean within group, concat across groups -> [B, G*N].
        z = fake_quant_act(h.mean(dim=2).flatten(1))
        return F.linear(z, fake_quant_act(self.head.weight), self.head.bias)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=64, help="heads per group")
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chars", type=int, default=4_000_000,
                    help="corpus slice; the full 17.9M is slow to window")
    ap.add_argument("--out", default="nhit_weights_mh.h")
    args = ap.parse_args()

    global CTX
    CTX = args.ctx

    if not CORPUS.exists():
        print("corpus_frozen.txt missing -- run freeze_corpus.py first")
        return 1
    full = CORPUS.read_text(encoding="utf-8", errors="ignore")
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()

    # Take evenly spaced BLOCKS, not a prefix. The corpus is ordered by source
    # repository, so text[:4M] is franken-ao plus the head of jl-wave-fabric --
    # heavily indented C, where the majority class (space) alone scores 0.2521.
    # Blocks keep windows locally contiguous while sampling the whole corpus;
    # the ~31 windows straddling block seams are noise at this scale.
    if args.chars >= len(full):
        text = full
    else:
        nblk = 32
        blk = args.chars // nblk
        stride = (len(full) - blk) // (nblk - 1)
        text = "".join(full[i * stride: i * stride + blk] for i in range(nblk))

    chars = sorted(set(text))
    if len(chars) > 64:
        freq = {c: text.count(c) for c in chars}
        chars = sorted(sorted(chars, key=lambda c: -freq[c])[:64])
    stoi = {c: i for i, c in enumerate(chars)}
    text = "".join(c for c in text if c in stoi)
    vocab = len(chars)

    # Keep the corpus FLAT on device and gather windows per batch. Materialising
    # every window as a [len(text), CTX] int64 tensor would cost 512 MB for a 4M
    # slice -- the same data replicated CTX times for no reason.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=dev)
    n_win = len(data) - CTX - 1
    cut = int(0.9 * n_win)
    offs = torch.arange(CTX, device=dev)

    def batch(idx):
        return data[idx.unsqueeze(1) + offs], data[idx + CTX]

    va_idx = torch.arange(cut, min(cut + 8192, n_win), device=dev)
    Xva, Yva = batch(va_idx)

    taps_total = args.groups * args.heads * args.depth * N
    b_taps = taps_total // 2                     # INT4, two per byte
    b_read = args.groups * N * vocab             # INT8 readout
    b_emb = vocab * N                            # INT8 embedding
    total = b_taps + b_read + b_emb

    print(f"corpus     frozen, sha256 {digest[:16]}...")
    print(f"           using {len(text):,} of 17,859,972 chars, vocab {vocab}")
    print(f"model      groups={args.groups} heads={args.heads} "
          f"depth={args.depth} N={N} CTX={CTX}")
    print(f"           {taps_total:,} taps, readout {args.groups*N}->{vocab}")
    print(f"bytes      taps {b_taps/1024:>7.0f} KB   INT4")
    print(f"           read {b_read/1024:>7.0f} KB   INT8  (scales with groups)")
    print(f"           emb  {b_emb/1024:>7.0f} KB   INT8")
    print(f"           ---- {total/1024:>7.0f} KB   "
          f"{'OVER 576 KB' if total > 576*1024 else 'fits'}")
    print(f"compute    {taps_total*N:,} MACs per inference")
    print(f"previous   192 taps -> this is {taps_total//192}x larger\n")

    model = MultiHeadNhit(vocab, args.heads, args.depth, args.groups).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    for step in range(args.steps):
        xb, yb = batch(torch.randint(cut, (args.batch,), device=dev))
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(model(Xva[:2048]), Yva[:2048])
                va = float((model(Xva[:2048]).argmax(-1) == Yva[:2048]).float().mean())
            model.train()
            print(f"  step {step:>5}  train {loss.item():.3f}  "
                  f"val {vl.item():.3f}  acc {va:.4f}", flush=True)

    # Chunked: one forward over all 8192 val windows would build a
    # [8192, G, H, N] complex transform -- 537M elements at G=16 -- and cuFFT
    # fails outright rather than degrading.
    def predict(x, chunk=512):
        return torch.cat([model(x[i:i + chunk])
                          for i in range(0, len(x), chunk)])

    model.eval()
    with torch.no_grad():
        acc = float((predict(Xva).argmax(-1) == Yva).float().mean())
        # Majority class taken from TRAIN, scored on VAL -- taking it from val
        # would leak the answer and inflate the bar we are trying to clear.
        major = torch.bincount(data[:cut], minlength=vocab).argmax()
        baseline = float((Yva == major).float().mean())

    print(f"\nmajority-class baseline  {baseline:.4f}")
    print(f"QAT model accuracy       {acc:.4f}")
    print(f"margin over baseline     {acc - baseline:+.4f}")
    print(f"\nreference (192 taps, unfrozen corpus): +0.028")

    torch.save({"taps": model.taps.detach().cpu(),
                "embed": model.embed.weight.detach().cpu(),
                "head_w": model.head.weight.detach().cpu(),
                "head_b": model.head.bias.detach().cpu(),
                "chars": "".join(chars), "heads": args.heads,
                "groups": args.groups, "depth": args.depth, "ctx": CTX,
                # The exact validation slice, so the C integer path can be
                # scored on the SAME windows rather than on a re-derived split.
                # Reproducing the split in C would be duplicated logic, and any
                # drift between the two would show up as a quantization loss
                # that is not real.
                "val_text": text[cut:cut + 8192 + CTX + 1],
                "val_acc": acc, "baseline": baseline},
               HERE / "mh_model.pt")
    print(f"saved mh_model.pt for export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
