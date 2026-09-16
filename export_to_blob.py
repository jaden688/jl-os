#!/usr/bin/env python3
"""
Export trained NHIT model to binary blob format (.nhit).

Format matches jl_lm_load() expectations:
- Header (28 bytes): magic, groups, heads, depth, vocab, read_dim, ctx
- Weights: packed INT4 taps + shifts + embeddings + readout + bias + chars

Usage:
    python export_to_blob.py mh_model.pt --out core.nhit
"""

import argparse
import struct
import torch
import pathlib

NHIT_N = 64
NHIT_MAGIC = 0x5449484E  # "NHIT" in little-endian

def export_model(model_pt, out_path):
    """Export PyTorch model to binary blob"""

    print(f"Loading model from {model_pt}...")
    state = torch.load(model_pt, map_location='cpu')

    taps = state['taps']  # [G, H, D, N]
    embed = state['embed']  # [vocab, N]
    head_w = state['head_w']  # [vocab, G*N]
    head_b = state['head_b']  # [vocab]
    chars_str = state['chars']

    groups = taps.shape[0]
    heads = taps.shape[1]
    depth = taps.shape[2]
    vocab = embed.shape[0]
    read_dim = groups * NHIT_N
    ctx = state.get('ctx', 4)

    print(f"Model config:")
    print(f"  Groups: {groups}")
    print(f"  Heads: {heads}")
    print(f"  Depth: {depth}")
    print(f"  Vocab: {vocab}")
    print(f"  Context: {ctx}")
    print(f"  Taps: {groups * heads * depth * NHIT_N:,}")

    # Quantize
    print(f"Quantizing...")

    # INT4 taps
    def quant_int4(t):
        peak = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.round(t / (peak / 7.0)).clamp(-8, 7).to(torch.int32)

    taps_q = quant_int4(taps)  # [G, H, D, N]

    # INT8 embeddings
    def quant_int8(t):
        peak = t.abs().amax().clamp_min(1e-8)
        step = peak / 127.0
        return torch.round(t / step).clamp(-128, 127).to(torch.int32)

    embed_q = quant_int8(embed)  # [vocab, N]
    head_w_q = quant_int8(head_w)  # [vocab, read_dim]
    head_b_i = head_b.to(torch.int32)  # [vocab]

    # Compute per-ring shifts (p99 calibration)
    # This matches export_multihead.py's approach
    shifts = []

    # Bundle and run through model to get activations for shift calibration
    # For now, use a simple heuristic: shift that keeps max at ~100
    for g in range(groups):
        for h in range(heads):
            for d in range(depth):
                kernel = taps_q[g, h, d]
                peak = kernel.abs().amax().float()
                if peak > 0:
                    # Shift such that peak * N ≈ 127
                    shift = max(0, int((peak * NHIT_N / 127.0).log2().ceil()))
                else:
                    shift = 0
                shifts.append(shift)

    shifts = torch.tensor(shifts, dtype=torch.uint8)
    print(f"  Computed {len(shifts)} per-ring shifts")

    # Write blob
    print(f"Writing to {out_path}...")
    with open(out_path, 'wb') as f:
        # Header
        header = struct.pack("<7I",
            NHIT_MAGIC,
            groups, heads, depth, vocab, read_dim, ctx
        )
        f.write(header)

        # Packed INT4 taps: [G*H*D, NHIT_PACKED_BYTES]
        NHIT_PACKED_BYTES = NHIT_N // 2
        taps_flat = taps_q.reshape(-1, NHIT_N)  # [G*H*D, N]
        for i, tap_row in enumerate(taps_flat):
            packed = []
            for j in range(0, NHIT_N, 2):
                lo = int(tap_row[j]) + 8
                hi = int(tap_row[j+1]) + 8
                packed.append((hi << 4) | lo)
            f.write(bytes(packed))

        # Per-ring shifts: [G*H*D]
        f.write(shifts.to(torch.uint8).numpy().tobytes())

        # Embeddings: [vocab, N] INT8
        f.write(embed_q.to(torch.uint8).numpy().tobytes())

        # Readout matrix: [vocab, read_dim] INT8
        f.write(head_w_q.to(torch.uint8).numpy().tobytes())

        # Readout bias: [vocab] INT32
        f.write(head_b_i.numpy().tobytes())

        # Character table: [vocab]
        chars_bytes = chars_str.encode('utf-8')[:vocab]
        f.write(chars_bytes)
        if len(chars_bytes) < vocab:
            f.write(b'\x00' * (vocab - len(chars_bytes)))

    file_size = out_path.stat().st_size if pathlib.Path(out_path).exists() else 0
    print(f"Exported: {file_size:,} bytes")

    # Verify structure
    expected_size = (
        28 +  # header
        (groups * heads * depth * NHIT_PACKED_BYTES) +  # taps
        (groups * heads * depth) +  # shifts
        (vocab * NHIT_N) +  # embeddings
        (vocab * read_dim) +  # readout matrix
        (vocab * 4) +  # readout bias
        vocab  # character table
    )
    print(f"Expected size: {expected_size:,} bytes")

    if file_size == expected_size:
        print("✓ Size matches exactly")
    else:
        print(f"✗ Size mismatch: {file_size} vs {expected_size}")

    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="PyTorch model file (mh_model.pt)")
    ap.add_argument("--out", default="core.nhit", help="Output blob file")
    args = ap.parse_args()

    out = export_model(args.model, args.out)
    print(f"\nReady to test: ./test_lm_host lm_valset.bin {out}")
