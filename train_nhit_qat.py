"""Quantization-aware training for NHIT ring kernels.

Post-training quantization currently costs most of the model's accuracy:

    float 0.2397  ->  integer path 0.1200   (baseline 0.1172)

The float model never sees the quantizer, so training walks to a minimum that
happens to be fragile under rounding. QAT puts the quantizer INSIDE the forward
pass, so the loss being minimised is the loss the kernel will actually incur.
Gradients flow through the rounding with a straight-through estimator: rounding
has zero derivative almost everywhere, so the backward pass treats it as the
identity while the forward pass keeps the hard quantized value.

Everything the kernel does is simulated during training, not just tap rounding:

    taps         symmetric integer, `bits` wide
    activations  INT8, saturating
    rescale      arithmetic right shift (power-of-two only)
    ReLU         between layers, matching nhit_stack

Two knobs so their contributions can be separated rather than conflated:

    --bits 4     what ships today
    --bits 8     wider taps; needs the INT8 tap path in nhit.c

    python train_nhit_qat.py --bits 4
    python train_nhit_qat.py --bits 8
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
N = 64
DEPTH = 3
CTX = 16


def load_text() -> str:
    parts = []
    for pat in ("*.md", "*.c", "*.h"):
        for p in sorted(HERE.glob(pat)):
            try:
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    extra = pathlib.Path("C:/Users/J_lin/Desktop/jl-wave-fabric")
    if extra.exists():
        for p in sorted(extra.rglob("*.md"))[:15]:
            if ".git" not in p.as_posix():
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


class RoundSTE(torch.autograd.Function):
    """round() forward, identity backward -- the straight-through estimator."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def fake_quant_taps(taps: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-kernel quantization, differentiable via STE.

    Returns values on the integer grid the kernel stores, rescaled back so the
    rest of the graph stays in float. The kernel sees exactly these levels.
    """
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    peak = taps.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    step = peak / qmax
    q = RoundSTE.apply(taps / step).clamp(qmin, qmax)
    return q * step


def fake_quant_act(x: torch.Tensor) -> torch.Tensor:
    """INT8 activations with saturation, matching sat8() in nhit.c."""
    peak = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    step = peak / 127.0
    q = RoundSTE.apply(x / step).clamp(-128, 127)
    return q * step


def circular_conv(x, taps):
    return torch.fft.ifft(torch.fft.fft(x, dim=-1)
                          * torch.fft.fft(taps, dim=-1)).real


class QatNhit(nn.Module):
    def __init__(self, vocab: int, bits: int):
        super().__init__()
        self.bits = bits
        self.embed = nn.Embedding(vocab, N)
        self.taps = nn.Parameter(torch.randn(DEPTH, N) * 0.3)
        self.head = nn.Linear(N, vocab)

    def forward(self, idx):
        e = self.embed(idx)
        # Position as rotation on the ring, matching to_int8_ring().
        x = torch.stack([torch.roll(e[:, p], shifts=p, dims=-1)
                         for p in range(e.shape[1])], dim=1).sum(dim=1)
        x = fake_quant_act(x)                       # the INT8 input the kernel gets
        for d in range(DEPTH):
            t = fake_quant_taps(self.taps[d].unsqueeze(0), self.bits)
            x = circular_conv(x, t)
            x = fake_quant_act(x)                   # INT8 after the shift rescale
            if d + 1 < DEPTH:
                x = F.relu(x)
        return self.head(x)


def quantise_int(taps: np.ndarray, bits: int, calib):
    """Final export quantization, identical in form to what QAT simulated."""
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q = np.zeros_like(taps, dtype=np.int16)
    for d in range(taps.shape[0]):
        peak = np.abs(taps[d]).max()
        if peak >= 1e-12:
            q[d] = np.clip(np.round(taps[d] / (peak / qmax)), qmin, qmax)

    shifts = np.zeros(taps.shape[0], dtype=np.int32)
    acts = [np.array(x, dtype=np.int32) for x in calib]
    for d in range(taps.shape[0]):
        accs, mags = [], []
        for a in acts:
            acc = np.zeros(N, dtype=np.int64)
            for o in range(N):
                idx = (o - np.arange(N)) & (N - 1)
                acc[o] = int(np.dot(a, q[d][idx].astype(np.int64)))
            accs.append(acc)
            mags.append(np.abs(acc))
        p99 = np.percentile(np.concatenate(mags), 99.0)
        sh = 0
        while p99 / (2.0 ** sh) > 110.0 and sh < 24:
            sh += 1
        shifts[d] = sh
        nxt = []
        for acc in accs:
            v = np.clip(acc >> int(sh), -128, 127).astype(np.int32)
            if d + 1 < taps.shape[0]:
                v = np.maximum(v, 0)
            nxt.append(v)
        acts = nxt
    return q, shifts


def int_stack(x, q, shifts):
    a = x.astype(np.int32)
    for d in range(DEPTH):
        acc = np.zeros(N, dtype=np.int64)
        for o in range(N):
            idx = (o - np.arange(N)) & (N - 1)
            acc[o] = int(np.dot(a, q[d][idx].astype(np.int64)))
        a = np.clip(acc >> int(shifts[d]), -128, 127).astype(np.int32)
        if d + 1 < DEPTH:
            a = np.maximum(a, 0)
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4, choices=(4, 8))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    text = load_text()
    chars = sorted(set(text))
    if len(chars) > 64:
        freq = {c: text.count(c) for c in chars}
        chars = sorted(sorted(chars, key=lambda c: -freq[c])[:64])
    stoi = {c: i for i, c in enumerate(chars)}
    text = "".join(c for c in text if c in stoi)
    vocab = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    X = torch.stack([data[i:i + CTX] for i in range(len(data) - CTX - 1)])
    Y = data[CTX:len(data) - 1]
    cut = int(0.9 * len(X))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr = X[:cut].to(dev), Y[:cut].to(dev)
    Xva, Yva = X[cut:].to(dev), Y[cut:].to(dev)

    print(f"QAT  bits={args.bits}  vocab={vocab}  ring N={N}  depth={DEPTH}")
    print(f"taps={DEPTH * N}  (a GEMM stack: {DEPTH * N * N:,})")

    model = QatNhit(vocab, args.bits).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for step in range(args.steps):
        i = torch.randint(len(Xtr), (512,), device=dev)
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr[i]), Ytr[i])
        loss.backward()
        opt.step()
        sched.step()
        if step % 800 == 0:
            with torch.no_grad():
                v = F.cross_entropy(model(Xva[:4096]), Yva[:4096])
            print(f"  step {step:>4}  train {loss.item():.3f}  val {v.item():.3f}")

    with torch.no_grad():
        qat_acc = float((model(Xva[:4096]).argmax(-1) == Yva[:4096]).float().mean())
        baseline = float((Yva[:4096] == torch.mode(Ytr).values).float().mean())

    taps = model.taps.detach().cpu().numpy()
    emb = model.embed.weight.detach().cpu().numpy()
    hw = model.head.weight.detach().cpu().numpy()
    hb = model.head.bias.detach().cpu().numpy()

    def to_ring(ctx):
        vec = np.zeros(N)
        for pos, ch in enumerate(ctx):
            vec += np.roll(emb[ch], pos)
        scale = (np.abs(vec).max() / 127.0) or 1.0
        return np.clip(np.round(vec / scale), -128, 127).astype(np.int32)

    calib = [to_ring(c) for c in Xva[:256].cpu().numpy()]
    q, shifts = quantise_int(taps, args.bits, calib)
    print(f"calibrated shifts: {list(map(int, shifts))}")

    hits = 0
    n_check = 600
    for c, t in zip(Xva[:n_check].cpu().numpy(), Yva[:n_check].cpu().numpy()):
        if int(np.argmax(hw @ int_stack(to_ring(c), q, shifts) + hb)) == int(t):
            hits += 1
    int_acc = hits / n_check

    print(f"\nmajority-class baseline    {baseline:.4f}")
    print(f"QAT model (quantizer live) {qat_acc:.4f}")
    print(f"INT{args.bits} kernel path        {int_acc:.4f}   <- what boots")
    print(f"\nprevious post-training quantization: float 0.2397 -> integer 0.1200")

    out = pathlib.Path(args.out or (HERE / "nhit_weights.h"))
    lines = [
        "/* AUTO-GENERATED by train_nhit_qat.py -- do not edit.",
        " * Quantization-aware trained NHIT ring kernels.",
        f" * ring N={N} depth={DEPTH} taps={DEPTH * N} bits={args.bits}",
        f" * QAT {qat_acc:.4f} / integer path {int_acc:.4f} / baseline {baseline:.4f}",
        " */",
        "#ifndef JL_NHIT_WEIGHTS_H",
        "#define JL_NHIT_WEIGHTS_H",
        f"#define NHIT_TRAINED_DEPTH {DEPTH}",
        f"#define NHIT_TAP_BITS {args.bits}",
    ]
    if args.bits == 4:
        lines.append(f"static const unsigned char NHIT_TRAINED_TAPS[{DEPTH}][{N // 2}] = {{")
        for d in range(DEPTH):
            packed = [((int(q[d][i + 1]) + 8) << 4) | (int(q[d][i]) + 8)
                      for i in range(0, N, 2)]
            lines.append("  {" + ",".join(str(b) for b in packed) + "},")
    else:
        lines.append(f"static const signed char NHIT_TRAINED_TAPS8[{DEPTH}][{N}] = {{")
        for d in range(DEPTH):
            lines.append("  {" + ",".join(str(int(v)) for v in q[d]) + "},")
    lines.append("};")
    lines.append(f"static const int NHIT_TRAINED_SHIFT[{DEPTH}] = {{"
                 + ",".join(str(int(s)) for s in shifts) + "};")
    lines.append("#endif")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
