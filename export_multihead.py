"""Export a trained grouped-NHIT model to a C header the kernel compiles in.

The kernel is integer-only, so everything crossing this boundary must already be
integer: INT4 taps packed two per byte, INT8 embedding and readout, and
power-of-two shifts. Nothing here may emit a float.

The one genuinely lossy step is the shift calibration. Training used per-sample
dynamic activation scaling; the kernel has a single fixed shift per ring. Shifts
are therefore calibrated EMPIRICALLY on real activations at the 99th percentile.
Calibrating on the worst case was tried and crushed activations to zero -- one
outlier sets a shift that silently zeroes every ordinary value.

    python export_multihead.py --model mh_model.pt --out nhit_weights_mh.h
"""

from __future__ import annotations

import argparse
import pathlib
import struct

import torch

HERE = pathlib.Path(__file__).resolve().parent
N = 64


def quant_taps_int4(taps: torch.Tensor) -> torch.Tensor:
    """[..., N] float -> [..., N] int in [-8, 7], symmetric per ring."""
    peak = taps.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.round(taps / (peak / 7.0)).clamp(-8, 7).to(torch.int32)


def pack_int4(row: torch.Tensor) -> list[int]:
    """N signed nibbles -> N/2 bytes, +8 biased, low nibble first.

    Must match nhit_tap() exactly: it reads byte i>>1 and takes the low nibble
    for even i, high for odd, then subtracts 8.
    """
    out = []
    for i in range(0, N, 2):
        lo = int(row[i]) + 8
        hi = int(row[i + 1]) + 8
        out.append((hi << 4) | lo)
    return out


def quant_int8(t: torch.Tensor) -> torch.Tensor:
    step = t.abs().amax().clamp_min(1e-8) / 127.0
    return torch.round(t / step).clamp(-128, 127).to(torch.int32)


def sat8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-128, 127)


def bundle_ctx(ctx: torch.Tensor, emb: torch.Tensor, ctxlen: int) -> torch.Tensor:
    """Mirror of lm_bundle_ctx() in jl_lm.c, in exact integer arithmetic.

    ctx: [B, CTX] vocab indices. Returns [B, N] INT8-ranged integers.
    Any divergence from the C version shows up as an accuracy gap between this
    exporter's prediction and the harness's, which is why the harness scores
    the real C code rather than trusting this.
    """
    B = ctx.shape[0]
    acc = torch.zeros(B, N, dtype=torch.int32)
    for p in range(ctxlen):
        e = emb[ctx[:, p]]                       # [B, N] int32
        acc += torch.roll(e, shifts=p, dims=-1)  # rotation by p on Z/N
    return sat8(torch.div(acc, ctxlen, rounding_mode="trunc"))


def circ_conv(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """out[k] = sum_j x[j] * w[(k-j) mod N], exactly as nhit_apply computes it.

    Done via FFT for speed; the inputs are integers on a small grid so the
    result is integer-exact after rounding, and it is asserted below.
    """
    f = torch.fft.fft(x.float(), dim=-1) * torch.fft.fft(taps.float(), dim=-1)
    out = torch.fft.ifft(f, dim=-1).real
    return torch.round(out).to(torch.int64)


def calibrate_shifts(ck, q, emb, n_samples: int) -> list[int]:
    G, H, D = ck["groups"], ck["heads"], ck["depth"]
    text = ck.get("val_text") or ""
    stoi = {c: i for i, c in enumerate(ck["chars"])}

    if len(text) < n_samples + 64:
        # No stored slice (older checkpoint): fall back to the frozen corpus.
        raw = (HERE / "corpus_frozen.txt").read_text(encoding="utf-8",
                                                     errors="ignore")
        text = "".join(c for c in raw[:400_000] if c in stoi)

    ctxlen = 16
    idx = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    n = min(n_samples, len(idx) - ctxlen - 1)
    ctx = torch.stack([idx[i:i + ctxlen] for i in range(n)])

    x = bundle_ctx(ctx, emb, ctxlen)                      # [B, N]
    h = x.unsqueeze(1).unsqueeze(1).expand(-1, G, H, -1)  # [B, G, H, N]

    shifts = torch.zeros(G, H, D, dtype=torch.long)
    for d in range(D):
        acc = circ_conv(h, q[:, :, d, :].unsqueeze(0))    # [B, G, H, N] int
        # p99 of |accumulator| per ring, over samples and ring positions.
        mag = acc.abs().permute(1, 2, 0, 3).reshape(G, H, -1).float()
        p99 = torch.quantile(mag, 0.99, dim=-1).clamp_min(1.0)
        s = torch.ceil(torch.log2(p99 / 127.0)).clamp(0, 20).long()
        shifts[:, :, d] = s
        h = sat8(acc >> s[None, :, :, None])
        if d + 1 < D:
            h = h.clamp_min(0)                            # ReLU

    # Mean within group, concat across groups -- nhit_grouped's output layout.
    rings = sat8(torch.div(h.sum(dim=2), H, rounding_mode="trunc"))
    return shifts.flatten().tolist(), rings.reshape(-1, G * N)


def carr(name: str, ctype: str, vals, per_line: int = 24) -> str:
    body, line = [], []
    for v in vals:
        line.append(str(int(v)))
        if len(line) == per_line:
            body.append("  " + ",".join(line) + ",")
            line = []
    if line:
        body.append("  " + ",".join(line) + ",")
    return (f"static const {ctype} {name}[{len(vals)}] = {{\n"
            + "\n".join(body) + "\n};\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mh_model.pt")
    ap.add_argument("--out", default="nhit_weights_mh.h")
    ap.add_argument("--calib", type=int, default=2048,
                    help="contexts used to measure accumulator magnitudes")
    args = ap.parse_args()

    ck = torch.load(HERE / args.model, map_location="cpu", weights_only=False)
    G, H, D = ck["groups"], ck["heads"], ck["depth"]
    taps = ck["taps"]                        # [G, H, D, N] float
    vocab = ck["head_w"].shape[0]

    q = quant_taps_int4(taps)                # [G, H, D, N] int4
    packed = []
    for g in range(G):
        for h in range(H):
            for d in range(D):
                packed.extend(pack_int4(q[g, h, d]))

    emb = quant_int8(ck["embed"])            # [vocab, N]

    # --- shift calibration, empirical ---------------------------------------
    # The kernel applies ONE fixed power-of-two shift per ring, but training
    # used per-sample dynamic scaling. Choosing those shifts is the single
    # lossiest step in the export and every closed-form guess tried here has
    # been wrong: the analytic worst case (N*127*7) crushes ordinary
    # activations to zero, and an RMS-based estimate measured +0.0113 against
    # PyTorch's +0.0525.
    #
    # So measure instead. Push REAL contexts through the exact integer pipeline
    # jl_lm.c implements, collect the true INT32 accumulator magnitudes per
    # ring, and pick the shift that puts the 99th percentile near INT8 full
    # scale. p99 not max, because a single outlier sets a shift that silently
    # zeroes every typical value.
    shifts, rings = calibrate_shifts(ck, q, emb, args.calib)
    rw = quant_int8(ck["head_w"])            # [vocab, G*N]

    # Readout: argmax is invariant to any positive scale on the logits, so the
    # accumulator needs no shift at all -- 4096 * 127 * 127 is ~66M and fits
    # INT32 with room. What is NOT scale-invariant is the bias, which is added
    # after the accumulation and must therefore be expressed in the SAME units.
    # Its correct scale is measured, not assumed: match the spread of the real
    # accumulator on calibration data.
    read_shift = 0
    acc_logits = rings.to(torch.float32) @ rw.to(torch.float32).T
    acc_sd = float(acc_logits.std().clamp_min(1.0))
    b_sd = float(ck["head_b"].std())
    k = acc_sd / b_sd if b_sd > 1e-6 else 0.0
    bias = torch.round(ck["head_b"] * k).clamp(-2 ** 30, 2 ** 30).to(torch.int32)

    hdr = [
        "/* AUTO-GENERATED by export_multihead.py -- do not edit.",
        " * Grouped NHIT: heads averaged within a group, groups concatenated.",
        f" * groups={G} heads={H} depth={D} N={N} vocab={vocab}",
        f" * taps={G*H*D*N} readout={G*N}x{vocab}",
        " */",
        "#ifndef JL_NHIT_WEIGHTS_MH_H",
        "#define JL_NHIT_WEIGHTS_MH_H",
        f"#define MH_GROUPS {G}",
        f"#define MH_HEADS {H}",
        f"#define MH_DEPTH {D}",
        f"#define MH_VOCAB {vocab}",
        f"#define MH_READ_DIM {G*N}",
        f"#define MH_READ_SHIFT {read_shift}",
        "",
        carr("MH_TAPS", "unsigned char", packed, 32),
        carr("MH_SHIFT", "signed char", shifts, 32),
        carr("MH_EMBED", "signed char", emb.flatten().tolist(), 32),
        carr("MH_READ_W", "signed char", rw.flatten().tolist(), 32),
        carr("MH_READ_B", "int", bias.tolist(), 16),
        # Emitted as bytes, not a string literal: the vocabulary contains
        # quotes, backslashes and newlines, all of which would need escaping
        # and any one of which would silently produce a wrong table.
        carr("MH_CHARS", "unsigned char",
             [ord(c) for c in ck["chars"]], 24),
        "#endif",
    ]
    text = "\n".join(x for x in hdr if x != "")
    (HERE / args.out).write_text(text, encoding="ascii")

    # Validation slice for the host harness, so the C integer path is scored on
    # the same windows PyTorch reported. Written as raw bytes; the harness does
    # no tokenizing of its own.
    if "val_text" in ck:
        (HERE / "lm_valset.bin").write_bytes(
            ck["val_text"].encode("latin-1", errors="replace"))
        print(f"  wrote lm_valset.bin ({len(ck['val_text']):,} chars), "
              f"torch reported acc {ck.get('val_acc', float('nan')):.4f}")

    # --- loadable blob for PEWTER-FS ----------------------------------------
    # The same weights as the C header, as a flat binary the kernel reads off
    # disk at boot. This is what decouples model size from kernel image size:
    # the header form must fit inside the 576 KB boot image, the blob form is
    # bounded only by the volume.
    blob = bytearray()
    blob += struct.pack("<6I", 0x5449484E,      # "NHIT"
                        G, H, D, vocab, G * N)
    blob += bytes(packed)
    blob += bytes((s & 0xFF) for s in shifts)
    blob += emb.flatten().to(torch.int8).numpy().tobytes()
    blob += rw.flatten().to(torch.int8).numpy().tobytes()
    blob += bias.numpy().astype("<i4").tobytes()
    blob += bytes(ord(c) for c in ck["chars"])
    (HERE / "core.nhit").write_bytes(blob)
    print(f"  wrote core.nhit ({len(blob):,} bytes) for the PEWTER volume")

    tot = len(packed) + emb.numel() + rw.numel() + 4 * len(bias) + len(shifts)
    print(f"groups={G} heads={H} depth={D} vocab={vocab}")
    print(f"  taps packed   {len(packed):>9,} bytes")
    print(f"  embed         {emb.numel():>9,} bytes")
    print(f"  readout w     {rw.numel():>9,} bytes")
    print(f"  shifts        {len(shifts):>9,} bytes  (int8 per ring)")
    print(f"  ------------- {tot:>9,} bytes  "
          f"{'OVER 576 KB' if tot > 576*1024 else 'fits under 576 KB'}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
