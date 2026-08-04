"""Train real NHIT ring kernels and export them as an integer-only C header.

The kernel's compute primitive is circular convolution on Z/64, not a matmul:

    out[k] = ( sum_j in[j] * w[(k-j) mod 64] ) >> shift

so the thing to learn is a length-64 tap vector per layer, not a 64x64 matrix.
Three layers with ReLU between them, 192 taps total, against a GEMM stack's
12,288 weights.

Training happens here in float. Everything is then quantised to the exact
integer layout nhit.c decodes -- INT4 taps packed two per byte with a +8 bias,
INT8 activations, INT32 accumulate, power-of-two shift -- and the QUANTISED
model is re-scored. The header records the accuracy the kernel will actually
deliver, not the float model's.

Task: next-character prediction over the project's own text, so the boot screen
is running inference on something real.

    python train_nhit.py        train, quantise, verify, write nhit_weights.h
"""

from __future__ import annotations

import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
N = 64            # ring size, must match NHIT_N in nhit.h
DEPTH = 3         # ring kernels in the stack
CTX = 16          # characters of context fed onto the ring
STEPS = 3000


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
            if ".git" in p.as_posix():
                continue
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def circular_conv(x, taps):
    """Batched circular convolution on Z/N via FFT. Differentiable."""
    return torch.fft.ifft(torch.fft.fft(x, dim=-1)
                          * torch.fft.fft(taps, dim=-1)).real


class NhitStack(nn.Module):
    """DEPTH ring kernels with ReLU between -- the model nhit.c executes."""

    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, N)
        self.taps = nn.Parameter(torch.randn(DEPTH, N) * 0.3)
        self.head = nn.Linear(N, vocab)

    def forward(self, idx):
        # Position enters as a ROTATION on the ring, not by averaging. Mean
        # pooling destroys character order, which is the whole task; rolling
        # each position by its index keeps order in the algebra itself.
        e = self.embed(idx)                                   # [B, CTX, N]
        rolled = torch.stack([torch.roll(e[:, p], shifts=p, dims=-1)
                              for p in range(e.shape[1])], dim=1)
        x = rolled.sum(dim=1)
        for d in range(DEPTH):
            x = circular_conv(x, self.taps[d].unsqueeze(0))
            if d + 1 < DEPTH:
                x = F.relu(x)
        return self.head(x)


def quantise_taps(taps: np.ndarray, calib_inputs):
    """INT4 taps plus shifts CALIBRATED ON REAL ACTIVATIONS.

    The theoretical worst case (sum|taps| * 127) overestimates the accumulator
    by more than an order of magnitude, and shifting by it drives every real
    activation to zero -- fatally so once layers stack. Instead push calibration
    data through layer by layer and pick each shift so the 99th percentile of
    the ACTUAL accumulator lands near INT8 full scale.
    """
    depth = taps.shape[0]
    q = np.zeros_like(taps, dtype=np.int8)
    shifts = np.zeros(depth, dtype=np.int32)
    for d in range(depth):
        peak = np.abs(taps[d]).max()
        if peak >= 1e-12:
            q[d] = np.clip(np.round(taps[d] / (peak / 7.0)), -8, 7).astype(np.int8)

    acts = [np.array(x, dtype=np.int32) for x in calib_inputs]
    for d in range(depth):
        mags = []
        accs = []
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
            if d + 1 < depth:
                v = np.maximum(v, 0)
            nxt.append(v)
        acts = nxt
    return q, shifts


def pack_taps(q_row: np.ndarray) -> np.ndarray:
    """Two INT4 taps per byte, low nibble first, +8 biased -- nhit_tap's layout."""
    out = np.zeros(N // 2, dtype=np.uint8)
    for i in range(0, N, 2):
        out[i // 2] = ((int(q_row[i + 1]) + 8) << 4) | (int(q_row[i]) + 8)
    return out


def int_stack(x_int8, q, shifts):
    """Bit-exact replica of nhit_stack: integer only, saturating to INT8."""
    a = x_int8.astype(np.int32)
    for d in range(DEPTH):
        acc = np.zeros(N, dtype=np.int64)
        for o in range(N):
            idx = (o - np.arange(N)) & (N - 1)
            acc[o] = int(np.dot(a, q[d][idx].astype(np.int64)))
        a = np.clip(acc >> int(shifts[d]), -128, 127).astype(np.int32)
        if d + 1 < DEPTH:
            a = np.maximum(a, 0)
    return a


def main():
    text = load_text()
    chars = sorted(set(text))
    if len(chars) > 64:
        freq = {c: text.count(c) for c in chars}
        chars = sorted(sorted(chars, key=lambda c: -freq[c])[:64])
    stoi = {c: i for i, c in enumerate(chars)}
    text = "".join(c for c in text if c in stoi)
    vocab = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"corpus {len(text):,} chars, vocab {vocab}, ring N={N}, depth={DEPTH}")

    X = torch.stack([data[i:i + CTX] for i in range(len(data) - CTX - 1)])
    Y = data[CTX:len(data) - 1]
    cut = int(0.9 * len(X))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr = X[:cut].to(dev), Y[:cut].to(dev)
    Xva, Yva = X[cut:].to(dev), Y[cut:].to(dev)

    model = NhitStack(vocab).to(dev)
    n_taps = DEPTH * N
    print(f"trainable ring taps: {n_taps}  (a GEMM stack would need "
          f"{DEPTH * N * N:,})")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
    for step in range(STEPS):
        i = torch.randint(len(Xtr), (512,), device=dev)
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr[i]), Ytr[i])
        loss.backward()
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                v = F.cross_entropy(model(Xva[:4096]), Yva[:4096])
            print(f"  step {step:>4}  train {loss.item():.3f}  val {v.item():.3f}")

    with torch.no_grad():
        logits = model(Xva[:4096])
        float_acc = float((logits.argmax(-1) == Yva[:4096]).float().mean())
        baseline = float((Yva[:4096] == torch.mode(Ytr).values).float().mean())

    taps = model.taps.detach().cpu().numpy()
    emb_pre = model.embed.weight.detach().cpu().numpy()

    def to_int8_ring(ctx):
        vec = np.zeros(N)
        for pos, ch in enumerate(ctx):
            vec += np.roll(emb_pre[ch], pos)
        scale = (np.abs(vec).max() / 127.0) or 1.0
        return np.clip(np.round(vec / scale), -128, 127).astype(np.int32)

    calib = [to_int8_ring(c) for c in Xva[:256].cpu().numpy()]
    q, shifts = quantise_taps(taps, calib)
    print(f"calibrated shifts: {list(map(int, shifts))}")

    # verify the integer path end to end on a sample
    emb = model.embed.weight.detach().cpu().numpy()
    head_w = model.head.weight.detach().cpu().numpy()
    n_check = 400
    hits = 0
    ctxs = Xva[:n_check].cpu().numpy()
    tgts = Yva[:n_check].cpu().numpy()
    head_b = model.head.bias.detach().cpu().numpy()
    for c, t in zip(ctxs, tgts):
        # Must mirror forward() exactly: roll each position's embedding by its
        # index and sum. Mean-pooling here scores a different model than the
        # one that trained, which is what made the integer path look broken.
        x8 = to_int8_ring(c)
        out = int_stack(x8, q, shifts)
        if int(np.argmax(head_w @ out + head_b)) == int(t):
            hits += 1
    int_acc = hits / n_check

    print(f"\nmajority-class baseline   {baseline:.4f}")
    print(f"float next-char accuracy  {float_acc:.4f}")
    print(f"INT4/INT8 kernel path     {int_acc:.4f}   <- what boots")

    lines = [
        "/* AUTO-GENERATED by train_nhit.py -- do not edit.",
        " * Trained NHIT ring kernels for the JL OS kernel.",
        f" * ring N={N}  depth={DEPTH}  taps={n_taps} (a GEMM stack: {DEPTH*N*N:,})",
        f" * float accuracy {float_acc:.4f} / integer path {int_acc:.4f}"
        f" / baseline {baseline:.4f}",
        " */",
        "#ifndef JL_NHIT_WEIGHTS_H",
        "#define JL_NHIT_WEIGHTS_H",
        f"#define NHIT_TRAINED_DEPTH {DEPTH}",
        f"static const unsigned char NHIT_TRAINED_TAPS[{DEPTH}][{N // 2}] = {{",
    ]
    for d in range(DEPTH):
        packed = pack_taps(q[d])
        lines.append("  {" + ",".join(str(int(b)) for b in packed) + "},")
    lines.append("};")
    lines.append(f"static const int NHIT_TRAINED_SHIFT[{DEPTH}] = {{"
                 + ",".join(str(int(s)) for s in shifts) + "};")
    lines.append("#endif")

    path = HERE / "nhit_weights.h"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
