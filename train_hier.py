"""Get past the ~4-item bundle bound: SHARD vs HIERARCHY, head to head.

The measured problem: the context bundle superposes CTX characters into N INT8
values by rotation, and HRR capacity is about N/(4 ln N) -- roughly 4 items at
N=64. Flat bundling peaks at CTX=4 (+0.2599) and collapses to +0.0009 at
CTX=32, where the model learns nothing. A parameter-matched LSTM is flat across
that range, so the bound belongs to the encoding, not the task.

Two ways past it, and they fail differently, which is why both are here:

  shard  [4][4][4][4] -> concat        no bundle exceeds 4, NOTHING is
                                       compressed. Readout width grows with
                                       context, so it dies past ~64 chars.

  hier   [4][4][4][4] -> fold to 1     no bundle exceeds 4, but each level
                                       squeezes `branch` vectors into one of
                                       the same width. Scales as branch^levels;
                                       loss can compound with depth.

Shard is the control. If shard works and hier does not, the problem is
compression, not capacity -- and that distinction decides which one is worth
putting in the kernel.

    python train_hier.py --mode shard --branch 4 --levels 2
    python train_hier.py --mode hier  --branch 4 --levels 2
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
N = 64


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

    h: [..., H, N]   taps: [H, depth, N]
    """
    for d in range(depth):
        ft = torch.fft.fft(fq_taps(taps[:, d, :]), dim=-1)
        while ft.dim() < h.dim():
            ft = ft.unsqueeze(0)
        h = fq_act(torch.fft.ifft(torch.fft.fft(h, dim=-1) * ft).real)
        if d + 1 < depth:
            h = F.relu(h)
    return h


def bundle(x: torch.Tensor) -> torch.Tensor:
    """[..., k, N] -> [..., N]: rotate by position and sum, on Z/N.

    Position is carried by the rotation; rotate-and-sum does not commute across
    different positions, so the bundle distinguishes "ab" from "ba".
    """
    parts = [torch.roll(x[..., p, :], shifts=p, dims=-1)
             for p in range(x.shape[-2])]
    return fq_act(torch.stack(parts, dim=-2).sum(dim=-2))


class ShardNhit(nn.Module):
    """ctx = branch * shards. Each shard bundles `branch` chars; shards concat.

    Nothing is compressed: every bundle stays at the capacity bound and all of
    them reach the readout intact. The price is that readout width scales with
    context, which is why this is a control rather than the destination.
    """

    def __init__(self, vocab, branch, shards, heads, depth):
        super().__init__()
        self.branch, self.shards, self.heads, self.depth = \
            branch, shards, heads, depth
        self.ctx = branch * shards
        self.embed = nn.Embedding(vocab, N)
        # One ring bank per shard -- shards see different parts of the context
        # and should be free to specialise, unlike heads within a shard.
        self.taps = nn.Parameter(torch.randn(shards, heads, depth, N) * 0.3)
        self.head = nn.Linear(shards * N, vocab)

    def forward(self, idx):
        B = idx.shape[0]
        e = F.embedding(idx, fq_act(self.embed.weight))
        b = bundle(e.reshape(B, self.shards, self.branch, N))   # [B, S, N]
        outs = []
        for s in range(self.shards):
            h = b[:, s, None, :].expand(B, self.heads, N)
            outs.append(ring_stack(h, self.taps[s], self.depth).mean(dim=1))
        z = fq_act(torch.stack(outs, dim=1).reshape(B, self.shards * N))
        return F.linear(z, fq_act(self.head.weight), self.head.bias)


class HierNhit(nn.Module):
    """ctx = branch^levels. Each level folds `branch` vectors into one.

    Lower levels average their heads back to width N, because that vector must
    itself be bundle-able by the level above; concatenating would change the
    width and break the recursion. Only the top level concatenates groups,
    where the output feeds a readout instead of another bundle.
    """

    def __init__(self, vocab, branch, levels, heads, depth, groups):
        super().__init__()
        self.branch, self.levels = branch, levels
        self.heads, self.depth, self.groups = heads, depth, groups
        self.ctx = branch ** levels
        self.embed = nn.Embedding(vocab, N)
        self.level_taps = nn.ParameterList([
            nn.Parameter(torch.randn(heads, depth, N) * 0.3)
            for _ in range(levels - 1)
        ])
        self.top_taps = nn.Parameter(torch.randn(groups, heads, depth, N) * 0.3)
        self.head = nn.Linear(groups * N, vocab)

    def forward(self, idx):
        B = idx.shape[0]
        h = F.embedding(idx, fq_act(self.embed.weight))         # [B, ctx, N]

        for lvl in range(self.levels - 1):
            h = bundle(h.reshape(B, -1, self.branch, N))        # [B, k, N]
            k = h.shape[1]
            hh = h[:, :, None, :].expand(B, k, self.heads, N)
            # Shared ring bank across positions at this level -- the same
            # weight sharing a convolution uses, and what keeps parameter count
            # independent of context length.
            hh = ring_stack(hh, self.level_taps[lvl], self.depth)
            h = fq_act(hh.mean(dim=2))                          # [B, k, N]

        h = bundle(h.reshape(B, -1, self.branch, N)).reshape(B, N)
        outs = []
        for g in range(self.groups):
            hg = h[:, None, :].expand(B, self.heads, N)
            outs.append(ring_stack(hg, self.top_taps[g], self.depth).mean(dim=1))
        z = fq_act(torch.stack(outs, dim=1).reshape(B, self.groups * N))
        return F.linear(z, fq_act(self.head.weight), self.head.bias)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shard", "hier"], default="shard")
    ap.add_argument("--branch", type=int, default=4)
    ap.add_argument("--levels", type=int, default=2, help="hier only")
    ap.add_argument("--shards", type=int, default=4, help="shard only")
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--groups", type=int, default=16, help="hier only")
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
    if args.mode == "shard":
        model = ShardNhit(vocab, args.branch, args.shards, args.heads, args.depth)
    else:
        model = HierNhit(vocab, args.branch, args.levels,
                         args.heads, args.depth, args.groups)
    model = model.to(dev)
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

    print(f"corpus     frozen, sha256 {digest[:16]}...")
    print(f"mode       {args.mode.upper()}  ctx {ctx}")
    print(f"           {taps:,} taps, {total:,} params total")
    print(f"baseline   {baseline:.4f}")
    print(f"reference  flat CTX=4 +0.2599   flat CTX=16 +0.0481   "
          f"flat CTX=32 +0.0009\n")

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

    print(f"\nbaseline           {baseline:.4f}")
    print(f"{args.mode} ctx {ctx:<10} {acc:.4f}")
    print(f"margin             {acc - baseline:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
