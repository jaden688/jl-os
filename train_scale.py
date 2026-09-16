#!/usr/bin/env python3
"""
NHIT Scaling: Train NHIT on full corpus with hyperparameter sweep.
Target: 2-10M taps, 65%+ accuracy

Runs parallel training across groups/heads/depth configurations.
Exports winners to .nhit blobs ready for kernel.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import json
import os
from pathlib import Path

print("=" * 80)
print("NHIT SCALING TRAINER")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

# Sweep these values
GROUPS_OPTIONS = [32, 64, 128]  # Independent representations
HEADS_OPTIONS = [64, 128]        # Taps per group
DEPTH_OPTIONS = [5, 7]           # Nonlinearity layers
RING_N = 64                       # Fixed ring size
VOCAB = 64
CTX = 4

# Training
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 5  # Per config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Ring size: {RING_N}")
print(f"Vocab: {VOCAB}")
print(f"Context: {CTX}")
print()

# ============================================================================
# Load corpus
# ============================================================================

print("[1/5] Loading corpus...")
with open('corpus_frozen.txt', 'r') as f:
    text = f.read()
print(f"Corpus size: {len(text):,} characters")

# Build vocabulary
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

if len(chars) > VOCAB:
    print(f"WARNING: Corpus has {len(chars)} unique chars, vocab is {VOCAB}")
    # Trim to top VOCAB chars by frequency
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    top_chars = sorted(freq.items(), key=lambda x: -x[1])[:VOCAB]
    chars = [c for c, _ in top_chars]
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    print(f"Using top {VOCAB} chars by frequency")

# Create windows: [CTX characters] -> next character
windows = []
for i in range(len(text) - CTX):
    ctx_chars = text[i:i+CTX]
    next_char = text[i+CTX]

    # Skip if any char not in vocab
    if any(c not in char_to_idx for c in ctx_chars + next_char):
        continue

    ctx_idx = [char_to_idx[c] for c in ctx_chars]
    next_idx = char_to_idx[next_char]
    windows.append((ctx_idx, next_idx))

print(f"Windows created: {len(windows):,}")

# Split: 90% train, 10% val
split = int(0.9 * len(windows))
train_windows = windows[:split]
val_windows = windows[split:]
print(f"Train: {len(train_windows):,}, Val: {len(val_windows):,}")

# ============================================================================
# Dataset
# ============================================================================

class ContextDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ctx_idx, next_idx = self.windows[idx]
        return torch.tensor(ctx_idx, dtype=torch.long), torch.tensor(next_idx, dtype=torch.long)

# ============================================================================
# NHIT Model
# ============================================================================

class NHITModel(nn.Module):
    def __init__(self, groups, heads, depth, ring_n, vocab_size, ctx_len):
        super().__init__()
        self.groups = groups
        self.heads = heads
        self.depth = depth
        self.ring_n = ring_n
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len

        # Character embeddings: [vocab_size, ring_n]
        self.embed = nn.Embedding(vocab_size, ring_n)

        # NHIT kernels: [groups * heads * depth, ring_n]
        self.kernels = nn.Parameter(
            torch.randn(groups * heads * depth, ring_n) * 0.1
        )

        # Shift per ring: [groups * heads * depth]
        self.shifts = nn.Parameter(
            torch.zeros(groups * heads * depth)
        )

        # Readout: [groups * ring_n, vocab_size]
        self.readout_w = nn.Parameter(
            torch.randn(vocab_size, groups * ring_n) * 0.1
        )
        self.readout_b = nn.Parameter(torch.zeros(vocab_size))

    def context_bundle(self, ctx_idx):
        """HRR context bundling: rotate embeddings by position, sum"""
        # ctx_idx: [batch, ctx_len]
        embed = self.embed(ctx_idx)  # [batch, ctx_len, ring_n]

        bundle = torch.zeros(
            embed.shape[0], self.ring_n,
            device=embed.device, dtype=embed.dtype
        )

        for pos in range(self.ctx_len):
            # Rotate by pos
            rotated = torch.roll(embed[:, pos, :], shifts=pos, dims=-1)
            bundle = bundle + rotated

        # Normalize
        bundle = bundle / self.ctx_len
        return bundle  # [batch, ring_n]

    def nhit_grouped(self, bundle):
        """Apply grouped ring convolutions"""
        batch_size = bundle.shape[0]

        # For each group: apply heads * depth rings
        outputs = []

        for g in range(self.groups):
            group_out = torch.zeros(
                batch_size, self.ring_n,
                device=bundle.device, dtype=bundle.dtype
            )

            for h in range(self.heads):
                head_in = bundle

                for d in range(self.depth):
                    kernel_idx = g * self.heads * self.depth + h * self.depth + d
                    kernel = self.kernels[kernel_idx]  # [ring_n]

                    # Circular convolution: FFT-based
                    # For simplicity, approximate with matrix mult on rotations
                    out = torch.zeros_like(head_in)
                    for k in range(self.ring_n):
                        for j in range(self.ring_n):
                            rot_idx = (k - j) % self.ring_n
                            out[:, k] += head_in[:, j] * kernel[rot_idx]

                    head_in = out

                    # ReLU between layers
                    if d < self.depth - 1:
                        head_in = torch.relu(head_in)

                # Sum heads
                group_out = group_out + head_in

            outputs.append(group_out)

        # Concatenate groups
        return torch.cat(outputs, dim=-1)  # [batch, groups * ring_n]

    def forward(self, ctx_idx):
        bundle = self.context_bundle(ctx_idx)
        ring_out = self.nhit_grouped(bundle)
        logits = torch.matmul(ring_out, self.readout_w.t()) + self.readout_b
        return logits

# ============================================================================
# Training loop
# ============================================================================

print()
print("[2/5] Training NHIT variants...")
print()

results = []

for groups in GROUPS_OPTIONS:
    for heads in HEADS_OPTIONS:
        for depth in DEPTH_OPTIONS:
            taps = groups * heads * depth * RING_N

            print(f"Training: groups={groups}, heads={heads}, depth={depth} ({taps:,} taps)")

            # Create model
            model = NHITModel(groups, heads, depth, RING_N, VOCAB, CTX).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            # Train
            train_loader = DataLoader(
                ContextDataset(train_windows),
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            val_loader = DataLoader(
                ContextDataset(val_windows),
                batch_size=BATCH_SIZE
            )

            best_val_acc = 0

            for epoch in range(EPOCHS):
                # Train
                model.train()
                train_loss = 0
                for ctx_idx, next_idx in train_loader:
                    ctx_idx, next_idx = ctx_idx.to(DEVICE), next_idx.to(DEVICE)

                    logits = model(ctx_idx)
                    loss = criterion(logits, next_idx)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # Validate
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for ctx_idx, next_idx in val_loader:
                        ctx_idx, next_idx = ctx_idx.to(DEVICE), next_idx.to(DEVICE)

                        logits = model(ctx_idx)
                        loss = criterion(logits, next_idx)
                        val_loss += loss.item()

                        pred = logits.argmax(dim=-1)
                        val_correct += (pred == next_idx).sum().item()
                        val_total += next_idx.shape[0]

                val_acc = val_correct / val_total
                best_val_acc = max(best_val_acc, val_acc)

                print(f"  Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss/len(train_loader):.4f}, "
                      f"val_acc={val_acc:.4f}")

            results.append({
                'groups': groups,
                'heads': heads,
                'depth': depth,
                'taps': taps,
                'accuracy': best_val_acc,
                'model': model
            })

            print(f"  Best val accuracy: {best_val_acc:.4f}")
            print()

# ============================================================================
# Results
# ============================================================================

print("[3/5] Results:")
print()

# Sort by accuracy
results.sort(key=lambda x: -x['accuracy'])

print(f"{'Taps':>10} {'Groups':>8} {'Heads':>8} {'Depth':>8} {'Accuracy':>10}")
print("-" * 54)
for r in results:
    print(f"{r['taps']:>10,} {r['groups']:>8} {r['heads']:>8} {r['depth']:>8} {r['accuracy']:>10.4f}")

best = results[0]
print()
print(f"WINNER: {best['taps']:,} taps, {best['accuracy']:.4f} accuracy")
print(f"  groups={best['groups']}, heads={best['heads']}, depth={best['depth']}")

# ============================================================================
# Export winner
# ============================================================================

print()
print("[4/5] Exporting...")

# TODO: Implement binary export to .nhit format
# For now, save PyTorch model
export_path = f"nhit_scale_{best['taps']}_taps.pt"
torch.save(best['model'].state_dict(), export_path)
print(f"Saved: {export_path}")

print()
print("[5/5] Next steps:")
print(f"  1. Implement binary export from {export_path}")
print(f"  2. Run: test_lm_host lm_valset.bin exported.nhit")
print(f"  3. Load into kernel and boot")

print()
print("=" * 80)
