"""Parameter-matched conventional models on the same task as NHIT.

This is the control the project has never run, and its absence is the largest
open question in every accuracy claim made so far.

NHIT has only ever been scored against a majority-class baseline. Clearing that
bar shows the model is FUNCTIONAL; it says nothing about whether the primitive
is COMPETITIVE. The one result where NHIT beat a transformer badly (0.9431 vs
0.0733) was compositional binding -- a task circular convolution is naturally
suited to, where the transformer had no inductive bias at all. Next-character
prediction is not that task, and assuming the win transfers would be exactly
the kind of unexamined claim this project keeps having to retract.

So: same frozen corpus, same block sampling, same 90/10 split, same context
length, same vocabulary, and parameter counts matched to NHIT's ~262K. Whatever
the answer is, it is measured rather than assumed.

    python control_baseline.py --ctx 16 --steps 1000
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

# NHIT at groups=16 heads=64 depth=3: 196,608 INT4 taps + a 1024x64 readout
# + a 64x64 embedding = 266,240 parameters. The controls are sized to match
# that total, not to match layer widths.
NHIT_PARAMS = 196_608 + 1024 * 64 + 64 * 64


class LstmLM(nn.Module):
    def __init__(self, vocab: int, hidden: int, emb: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)
        self.rnn = nn.LSTM(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        e = self.embed(x)
        out, _ = self.rnn(e)
        return self.head(out[:, -1])          # predict from the last step


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, d: int, heads: int, layers: int, ctx: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, ctx, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4,
            batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.embed(x) + self.pos[:, :x.shape[1]]
        return self.head(self.enc(h)[:, -1])


def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chars", type=int, default=4_000_000)
    args = ap.parse_args()

    full = CORPUS.read_text(encoding="utf-8", errors="ignore")
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()

    # Identical sampling to train_multihead.py. If this differed even slightly
    # the comparison would be between two tasks, not two models.
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

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=dev)
    ctx = args.ctx
    n_win = len(data) - ctx - 1
    cut = int(0.9 * n_win)
    offs = torch.arange(ctx, device=dev)

    def batch(idx):
        return data[idx.unsqueeze(1) + offs], data[idx + ctx]

    va_idx = torch.arange(cut, min(cut + 8192, n_win), device=dev)
    Xva, Yva = batch(va_idx)
    major = torch.bincount(data[:cut], minlength=vocab).argmax()
    baseline = float((Yva == major).float().mean())

    print(f"corpus     frozen, sha256 {digest[:16]}...")
    print(f"           {len(text):,} chars, vocab {vocab}, ctx {ctx}")
    print(f"target     NHIT parameter count {NHIT_PARAMS:,}")
    print(f"baseline   majority-class {baseline:.4f}\n")

    # Hidden size solved so the LSTM lands near NHIT's parameter count:
    # 4*(H^2 + H*E + H) + H*V + V*E  ~=  266,240 at E=64, V=64.
    models = {
        "LSTM": LstmLM(vocab, hidden=224),
        "Transformer": TinyTransformer(vocab, d=128, heads=4, layers=2, ctx=ctx),
    }

    results = {}
    for name, model in models.items():
        model = model.to(dev)
        n_par = count(model)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

        print(f"{name}  {n_par:,} params "
              f"({n_par / NHIT_PARAMS:.2f}x NHIT)")

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
                    acc = float((model(Xva[:2048]).argmax(-1)
                                 == Yva[:2048]).float().mean())
                model.train()
                print(f"  step {step:>5}  loss {loss.item():.3f}  acc {acc:.4f}",
                      flush=True)

        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xva[i:i + 512])
                              for i in range(0, len(Xva), 512)])
            acc = float((pred.argmax(-1) == Yva).float().mean())
        results[name] = (acc, n_par)
        print(f"  final {acc:.4f}   margin {acc - baseline:+.4f}\n")

    print("=" * 58)
    print(f"{'model':<16}{'params':>10}{'acc':>10}{'margin':>10}")
    print(f"{'majority class':<16}{'-':>10}{baseline:>10.4f}{0.0:>+10.4f}")
    for name, (acc, n_par) in results.items():
        print(f"{name:<16}{n_par:>10,}{acc:>10.4f}{acc - baseline:>+10.4f}")
    print(f"{'NHIT (float)':<16}{NHIT_PARAMS:>10,}{'see':>10}{'train_mh':>10}")
    print("=" * 58)
    print("\nNHIT is scored in FLOAT terms here only if compared against its QAT"
          "\nnumber; the kernel's integer path is lower still. A control that"
          "\nbeats the float model beats the integer one by more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
