"""Train a small char-level model and export it as an integer-only C header.

The kernel has no libc, no FPU guarantees and no runtime, so the exported model
is built to run with integers alone:

    weights      INT4, packed two per byte (the format franken_core.c already
                 uses for gemm_quantized_int4)
    activations  INT8
    accumulate   INT32
    rescale      arithmetic right shift by a power-of-two exponent

No float appears anywhere on the kernel side. Training happens here in float,
then everything is quantised and the quantised model is re-scored so the header
carries the accuracy the kernel will actually deliver -- not the float model's.

    python train_export.py            train, quantise, verify, write model.h
"""

from __future__ import annotations

import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
CTX = 8          # characters of context
EMB = 16         # embedding width
HID = 256        # hidden units
STEPS = 4000


def load_text() -> str:
    """Train on the project's own prose so generations are in its voice."""
    parts = []
    for pattern in ("*.md", "*.c", "*.h"):
        for path in sorted(HERE.glob(pattern)):
            try:
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    extra = pathlib.Path("C:/Users/J_lin/Desktop/jl-wave-fabric")
    if extra.exists():
        for path in sorted(extra.rglob("*.md"))[:20]:
            if ".git" in path.as_posix():
                continue
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


class CharMLP(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB)
        self.fc1 = nn.Linear(CTX * EMB, HID, bias=False)
        self.fc2 = nn.Linear(HID, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x).flatten(1)
        return self.fc2(F.relu(self.fc1(h)))


def quantise_int4(w: np.ndarray):
    """Symmetric per-row INT4 in [-8, 7], with a power-of-two scale exponent."""
    rows, cols = w.shape
    q = np.zeros((rows, cols), dtype=np.int8)
    shifts = np.zeros(rows, dtype=np.int32)
    for r in range(rows):
        peak = np.abs(w[r]).max()
        if peak < 1e-12:
            continue
        # step = peak / 7, rounded to a power of two so rescaling is a shift.
        exp = int(np.ceil(np.log2(peak / 7.0)))
        shifts[r] = exp
        q[r] = np.clip(np.round(w[r] / (2.0 ** exp)), -8, 7).astype(np.int8)
    return q, shifts


def pack_int4(q: np.ndarray) -> np.ndarray:
    """Two INT4 values per byte, low nibble first, biased by +8 to fit 0..15."""
    rows, cols = q.shape
    packed = np.zeros((rows, cols // 2), dtype=np.uint8)
    for r in range(rows):
        for c in range(0, cols, 2):
            lo = int(q[r, c]) + 8
            hi = int(q[r, c + 1]) + 8
            packed[r, c // 2] = (hi << 4) | lo
    return packed


def int_forward(ctx_ids, emb_q, w1_q, w1_s, w2_q, w2_s):
    """Bit-exact model of the kernel's integer path, used to verify the export."""
    x = emb_q[ctx_ids].reshape(-1).astype(np.int32)          # INT8 activations
    acc1 = (w1_q.astype(np.int32) @ x)                        # INT32 accumulate
    h = np.maximum(acc1 >> 7, 0)                              # ReLU + rescale
    h = np.clip(h, 0, 127).astype(np.int32)
    acc2 = (w2_q.astype(np.int32) @ h)
    return acc2


def main():
    text = load_text()
    chars = sorted(set(text))
    if len(chars) > 96:                     # keep the table small for the kernel
        freq = {c: text.count(c) for c in chars}
        chars = sorted(sorted(chars, key=lambda c: -freq[c])[:96])
    stoi = {c: i for i, c in enumerate(chars)}
    text = "".join(c for c in text if c in stoi)
    vocab = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"corpus {len(text):,} chars, vocab {vocab}")

    X = torch.stack([data[i:i + CTX] for i in range(len(data) - CTX - 1)])
    Y = data[CTX:len(data) - 1]
    split = int(0.9 * len(X))
    Xtr, Ytr, Xva, Yva = X[:split], Y[:split], X[split:], Y[split:]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharMLP(vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    Xtr, Ytr, Xva, Yva = Xtr.to(dev), Ytr.to(dev), Xva.to(dev), Yva.to(dev)
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
        float_acc = float((model(Xva[:4096]).argmax(-1) == Yva[:4096]).float().mean())

    # ---- quantise -----------------------------------------------------------
    emb = model.emb.weight.detach().cpu().numpy()
    emb_scale = np.abs(emb).max() / 127.0
    emb_q = np.clip(np.round(emb / emb_scale), -127, 127).astype(np.int8)

    w1 = model.fc1.weight.detach().cpu().numpy()
    w2 = model.fc2.weight.detach().cpu().numpy()
    w1_q, w1_s = quantise_int4(w1)
    w2_q, w2_s = quantise_int4(w2)

    # ---- verify the INTEGER path, not the float one -------------------------
    n_check = 2000
    ctxs = Xva[:n_check].cpu().numpy()
    tgts = Yva[:n_check].cpu().numpy()
    hits = 0
    for c, t in zip(ctxs, tgts):
        if int(np.argmax(int_forward(c, emb_q, w1_q, w1_s, w2_q, w2_s))) == int(t):
            hits += 1
    int_acc = hits / n_check
    print(f"\nfloat next-char accuracy {float_acc:.4f}")
    print(f"INT4/INT8 kernel path     {int_acc:.4f}   <- what the kernel delivers")

    # ---- emit the header ----------------------------------------------------
    w1_p, w2_p = pack_int4(w1_q), pack_int4(w2_q)

    def carr(name, arr, ctype):
        flat = arr.reshape(-1)
        body = ",".join(str(int(v)) for v in flat)
        return f"static const {ctype} {name}[{flat.size}] = {{{body}}};\n"

    esc = "".join(
        f"\\x{ord(c):02x}" if (c < ' ' or c > '~' or c in '"\\') else c
        for c in chars)

    out = [
        "/* AUTO-GENERATED by train_export.py -- do not edit.",
        " * Integer-only char model for the bare-metal kernel.",
        f" * vocab={vocab} ctx={CTX} emb={EMB} hid={HID}",
        f" * float next-char accuracy {float_acc:.4f}",
        f" * INT4/INT8 integer path   {int_acc:.4f}  (what the kernel runs)",
        " */",
        "#ifndef FRANKEN_MODEL_H",
        "#define FRANKEN_MODEL_H",
        f"#define M_VOCAB {vocab}",
        f"#define M_CTX   {CTX}",
        f"#define M_EMB   {EMB}",
        f"#define M_HID   {HID}",
        f"#define M_IN    {CTX * EMB}",
        f'static const char M_CHARS[{vocab} + 1] = "{esc}";',
        carr("M_EMB_Q", emb_q, "signed char"),
        carr("M_W1_PACKED", w1_p, "unsigned char"),
        carr("M_W1_SHIFT", w1_s, "int"),
        carr("M_W2_PACKED", w2_p, "unsigned char"),
        carr("M_W2_SHIFT", w2_s, "int"),
        "#endif",
    ]
    path = HERE / "franken_model.h"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"\nwrote {path.name}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
