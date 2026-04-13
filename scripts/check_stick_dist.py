#!/usr/bin/env python3
"""check_stick_dist.py — Print stick bin distribution from a training label file.

Usage:
    python3 check_stick_dist.py <y_path>

Example:
    python3 check_stick_dist.py /home/ubuntu/transformer_v1_y.npy
"""

import sys
import numpy as np

STICK_BINS = 21
BUTTON_DIM = 6
CHUNK      = 2_000_000
NAMES      = ["stick_x", "stick_y", "cstick_x", "cstick_y"]

y_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/transformer_v1_y.npy"
y      = np.lib.format.open_memmap(y_path, mode="r")
N      = len(y)
print(f"Total frames: {N:,}  Label dim: {y.shape[1]}")

counts = np.zeros((4, STICK_BINS), dtype=np.int64)

pos = 0
while pos < N:
    end   = min(pos + CHUNK, N)
    chunk = y[pos:end, BUTTON_DIM:].astype(np.float32)
    bins  = np.round((chunk.clip(-1, 1) + 1) / 2 * (STICK_BINS - 1)).astype(int)
    for axis in range(4):
        counts[axis] += np.bincount(bins[:, axis], minlength=STICK_BINS)
    pos = end
    print(f"  scanned {pos:,} / {N:,}", end="\r", flush=True)

print()

bin_vals = [f"{b / (STICK_BINS - 1) * 2 - 1:.2f}" for b in range(STICK_BINS)]

for axis, name in enumerate(NAMES):
    c     = counts[axis]
    total = c.sum()
    top5  = np.argsort(c)[::-1][:5]
    print(f"\n{name}:")
    print(f"  bin10 (neutral  0.00): {c[10] / total:.1%}  ({c[10]:,} frames)")
    print(f"  bins 8-12 (within ±0.20): {c[8:13].sum() / total:.1%}")
    print(f"  non-neutral (bins <8 or >12): {(c[:8].sum() + c[13:].sum()) / total:.1%}")
    print(f"  top 5 bins: " +
          "  ".join(f"bin{b}({bin_vals[b]})={c[b]/total:.1%}" for b in top5))
    print(f"  full distribution:")
    for b in range(STICK_BINS):
        bar = "#" * int(c[b] / total * 60)
        print(f"    bin{b:2d} ({bin_vals[b]}): {c[b]/total:5.1%}  {bar}")
