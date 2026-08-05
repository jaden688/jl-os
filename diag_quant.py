"""Isolate WHERE the QAT -> integer accuracy is lost.

At CTX=16 the integer path kept 87% of QAT accuracy (0.3706 -> 0.3228).
At CTX=4 it kept 20% (0.5708 -> 0.1128), which is below majority-class. Between
training and the kernel there are three separate places that can lose it:

  1. the export's quantization of embed / readout (global vs per-row scales)
  2. the per-ring shift calibration (dynamic scaling -> one fixed shift)
  3. a divergence between this exporter's integer model and jl_lm.c

This runs the exporter's own integer simulation end to end and scores it. If it
matches the C harness, (3) is excluded and the fault is in the export. If it
matches QAT, the export is fine and jl_lm.c disagrees with it.

Guessing has cost this project four wrong hypotheses in a row. This measures.
"""

from __future__ import annotations

import pathlib

import torch

import export_multihead as ex

HERE = pathlib.Path(__file__).resolve().parent
N = 64


def main() -> int:
    ck = torch.load(HERE / "mh_model.pt", map_location="cpu", weights_only=False)
    G, H, D = ck["groups"], ck["heads"], ck["depth"]
    ctxlen = int(ck.get("ctx", 16))
    chars = ck["chars"]
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}

    q = ex.quant_taps_int4(ck["taps"])
    emb = ex.quant_int8(ck["embed"])
    rw = ex.quant_int8(ck["head_w"])

    text = ck["val_text"]
    idx = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    n = len(idx) - ctxlen - 1
    ctx = torch.stack([idx[i:i + ctxlen] for i in range(n)])
    target = idx[ctxlen:ctxlen + n]

    shifts, _ = ex.calibrate_shifts(ck, q, emb, 2048)
    sh = torch.tensor(shifts).reshape(G, H, D)

    print(f"ctx {ctxlen}  groups {G} heads {H} depth {D} vocab {vocab}")
    print(f"windows {n}")
    print(f"shifts: min {sh.min().item()} max {sh.max().item()} "
          f"mean {sh.float().mean().item():.2f}")
    for d in range(D):
        print(f"  depth {d}: mean {sh[:, :, d].float().mean().item():.2f} "
              f"min {sh[:, :, d].min().item()} max {sh[:, :, d].max().item()}")

    # --- exporter's integer forward pass, in full ---------------------------
    x = ex.bundle_ctx(ctx, emb, ctxlen)
    print(f"\nbundle: absmax {int(x.abs().max())}  "
          f"absmean {x.abs().float().mean():.1f}  "
          f"saturated {(x.abs() >= 127).float().mean() * 100:.2f}%")

    h = x.unsqueeze(1).unsqueeze(1).expand(-1, G, H, -1)
    for d in range(D):
        acc = ex.circ_conv(h, q[:, :, d, :].unsqueeze(0))
        s = sh[:, :, d]
        h = ex.sat8(acc >> s[None, :, :, None])
        print(f"depth {d}: post-shift absmax {int(h.abs().max())} "
              f"absmean {h.abs().float().mean():.1f} "
              f"zeros {(h == 0).float().mean() * 100:.1f}% "
              f"saturated {(h.abs() >= 127).float().mean() * 100:.2f}%")
        if d + 1 < D:
            h = h.clamp_min(0)

    rings = ex.sat8(torch.div(h.sum(dim=2), H, rounding_mode="trunc"))
    rings = rings.reshape(-1, G * N)
    print(f"rings:  absmax {int(rings.abs().max())} "
          f"absmean {rings.abs().float().mean():.1f} "
          f"zeros {(rings == 0).float().mean() * 100:.1f}%")

    logits = rings.to(torch.float64) @ rw.to(torch.float64).T
    acc_sd = float(logits.std().clamp_min(1.0))
    b_sd = float(ck["head_b"].std())
    k = acc_sd / b_sd if b_sd > 1e-6 else 0.0
    bias = torch.round(ck["head_b"] * k).to(torch.float64)

    pred_nb = logits.argmax(-1)
    pred_b = (logits + bias).argmax(-1)
    major = torch.bincount(target, minlength=vocab).argmax()
    base = float((target == major).float().mean())

    print(f"\nmajority baseline        {base:.4f}")
    print(f"integer sim, no bias     {float((pred_nb == target).float().mean()):.4f}")
    print(f"integer sim, with bias   {float((pred_b == target).float().mean()):.4f}")
    print(f"QAT reported             {ck.get('val_acc', float('nan')):.4f}")
    print(f"C harness measured       0.1128")
    print("\nIf 'with bias' ~ the C harness, the loss is in this export.")
    print("If 'with bias' ~ QAT, the export is fine and jl_lm.c diverges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
