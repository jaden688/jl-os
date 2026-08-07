"""Expanding NHIT: the ring gets WIDER as it goes up.

The measured problem: a bundle superposes items into N INT8 values by rotation,
and HRR capacity is about N/(4 ln N) -- roughly 4 items at N=64. Flat bundling
peaks at CTX=4 (+0.2599) and collapses to +0.0009 at CTX=32. A parameter-matched
LSTM is flat across that range, so the bound is the encoding's, not the task's.

Capacity is a function of WIDTH, so widen:

    N=64    3.9 items      level 0: local detail, cheap, many positions
    N=256   11.5 items     level 1: wider, fewer positions
    N=1024  37.0 items     level 2: widest, one position

The join is the important part. When the expansion multiplier EQUALS the branch
factor, `mult` children of width N exactly fill a parent of width mult*N, so
they are CONCATENATED rather than bundled. Concatenation is lossless: no
rotation, no superposition, no interference. Capacity grows exactly as fast as
the content it has to hold, which is why this does not hit the bound again one
level up.

Only level 0 bundles, and it bundles `branch` characters into N=64 -- at the
capacity bound, not past it.

Compute is dominated by the top: level l has branch^(L-1-l) positions each
costing N_l^2 MACs, and N_l grows geometrically while the position count
shrinks geometrically. Widening is not free -- a ring holds N taps but costs
N^2 MACs -- so the point of keeping the BOTTOM narrow is that most positions
live there.

    python train_expand.py --branch 4 --levels 2      # ctx 16, top width 256
    python train_expand.py --branch 4 --levels 3      # ctx 64, top width 1024
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE / "corpus_frozen.txt"
N0 = 64          # base ring width; the kernel's NHIT_N


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def fq_taps(t: torch.Tensor, bits: int = 4) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    peak = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    step = peak / qmax
    return RoundSTE.apply(t / step).clamp(-(qmax + 1), qmax) * step


def fq_act(x: torch.Tensor) -> torch.Tensor:
    step = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    return RoundSTE.apply(x / step).clamp(-128, 127) * step


def ring_stack(h: torch.Tensor, taps: torch.Tensor, depth: int) -> torch.Tensor:
    """`depth` circular convolutions with ReLU between -- nhit_stack, spectrally.

    Width is taken from the last dimension, so the same code serves every level.
    On the kernel side this is the one change required: NHIT_N stops being a
    compile-time constant. The mask (o-j) & (N-1) is already correct for any
    power of two.
    """
    for d in range(depth):
        ft = torch.fft.fft(fq_taps(taps[:, d, :]), dim=-1)
        while ft.dim() < h.dim():
            ft = ft.unsqueeze(0)
        h = fq_act(torch.fft.ifft(torch.fft.fft(h, dim=-1) * ft).real)
        if d + 1 < depth:
            h = F.relu(h)
    return h


class ExpandNhit(nn.Module):
    """ctx = branch^levels, ring width multiplied by `branch` at every level.

    Level 0 bundles characters (the only lossy join, and it stays at the
    capacity bound). Every level above concatenates its children, which is
    lossless, then mixes them with rings at the wider size.
    """

    def __init__(self, vocab, branch, levels, heads, depth):
        super().__init__()
        self.branch, self.levels, self.heads, self.depth = \
            branch, levels, heads, depth
        self.ctx = branch ** levels
        self.widths = [N0 * (branch ** l) for l in range(levels)]

        self.embed = nn.Embedding(vocab, N0)
        # One ring bank per level, shared across that level's positions -- the
        # weight sharing a convolution uses, so parameter count depends on
        # width and depth rather than on context length.
        self.taps = nn.ParameterList([
            nn.Parameter(torch.randn(heads, depth, w) * 0.3) for w in self.widths
        ])
        self.head = nn.Linear(self.widths[-1], vocab)

    def forward(self, idx):
        B = idx.shape[0]
        e = F.embedding(idx, fq_act(self.embed.weight))          # [B, ctx, N0]

        # Level 0: bundle `branch` characters into one N0 ring. Rotation
        # carries position, so "ab" and "ba" differ. This is the ONLY place
        # superposition happens.
        g = e.reshape(B, -1, self.branch, N0)
        h = fq_act(torch.stack(
            [torch.roll(g[..., p, :], shifts=p, dims=-1)
             for p in range(self.branch)], dim=-2).sum(dim=-2))  # [B, k, N0]

        for lvl in range(self.levels):
            k = h.shape[1]
            hh = h[:, :, None, :].expand(B, k, self.heads, h.shape[-1])
            hh = ring_stack(hh, self.taps[lvl], self.depth)
            h = fq_act(hh.mean(dim=2))                           # [B, k, W]
            if lvl + 1 < self.levels:
                # CONCATENATE children into the wider parent. Lossless: no
                # rotation and no summing, so nothing interferes.
                h = h.reshape(B, k // self.branch, self.branch * h.shape[-1])

        z = h.reshape(B, self.widths[-1])
        return F.linear(z, fq_act(self.head.weight), self.head.bias)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", type=int, default=4)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chars", type=int, default=4_000_000)
    args = ap.parse_args()

    full = CORPUS.read_text(encoding="utf-8", errors="ignore")
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()
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

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ExpandNhit(vocab, args.branch, args.levels,
                       args.heads, args.depth).to(dev)
    ctx = model.ctx

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=dev)
    n_win = len(data) - ctx - 1
    cut = int(0.9 * n_win)
    offs = torch.arange(ctx, device=dev)

    def batch(i):
        return data[i.unsqueeze(1) + offs], data[i + ctx]

    va = torch.arange(cut, min(cut + 8192, n_win), device=dev)
    Xva, Yva = batch(va)
    major = torch.bincount(data[:cut], minlength=vocab).argmax()
    baseline = float((Yva == major).float().mean())

    taps = sum(p.numel() for n_, p in model.named_parameters() if "taps" in n_)
    total = sum(p.numel() for p in model.parameters())
    macs = sum((args.branch ** (args.levels - 1 - l)) * args.heads
               * args.depth * w * w for l, w in enumerate(model.widths))

    print(f"corpus     frozen, sha256 {digest[:16]}...")
    print(f"model      branch={args.branch} levels={args.levels} -> ctx {ctx}")
    # HRR capacity per level, N/(4 ln N). Printed because the whole design is
    # an argument about whether capacity keeps up with content: level l has to
    # hold branch^(l+1) characters, so capacity must grow at least that fast.
    cap = [w / (4 * math.log(w)) for w in model.widths]
    print(f"           widths   {model.widths}")
    print(f"           capacity {[f'{c:.1f}' for c in cap]} items")
    print(f"           must hold {[args.branch ** (l + 1) for l in range(args.levels)]} chars")
    print(f"           {taps:,} taps, {total:,} params, {macs:,} MACs")
    print(f"baseline   {baseline:.4f}")
    print(f"reference  flat CTX=4 +0.2599   CTX=16 +0.0481   CTX=32 +0.0009\n")

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
        if step % 250 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                a = float((model(Xva[:1024]).argmax(-1)
                           == Yva[:1024]).float().mean())
            model.train()
            print(f"  step {step:>5}  loss {loss.item():.3f}  acc {a:.4f}",
                  flush=True)

    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(Xva[i:i + 256])
                          for i in range(0, len(Xva), 256)])
        acc = float((pred.argmax(-1) == Yva).float().mean())

    print(f"\nbaseline            {baseline:.4f}")
    print(f"expand ctx {ctx:<8}  {acc:.4f}")
    print(f"margin              {acc - baseline:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
