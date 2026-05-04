#!/usr/bin/env python3
"""Analyze button combo distribution in training labels to build action vocabulary.

Usage:
    python analyze_button_combos.py "/home/ubuntu/SMS AI/scripts/transformer_v2_y.npy"
"""

import sys
import numpy as np
from collections import Counter

BUTTON_NAMES = ["A", "B", "X", "Y", "lob_pass", "chip_shot", "R"]
BUTTON_DIM = 7

y_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/SMS AI/scripts/transformer_v2_y.npy"

print(f"Loading {y_path} ...")
y = np.lib.format.open_memmap(y_path, mode="r")
print(f"  Shape: {y.shape}  dtype: {y.dtype}")

# Extract button columns (first 7 of 11 labels), round to 0/1
btns = (y[:, :BUTTON_DIM] > 0.5).astype(np.uint8)
print(f"  Total frames: {len(btns):,}")

# Convert each row to a tuple for counting
print("Counting combos ...")
combo_counts = Counter(map(tuple, btns))

print(f"\n{'='*70}")
print(f"Unique button combos: {len(combo_counts)}")
print(f"{'='*70}\n")

# Sort by frequency
sorted_combos = sorted(combo_counts.items(), key=lambda x: -x[1])

print(f"{'Rank':>4}  {'Count':>12}  {'%':>7}  {'Cumul%':>7}  Combo")
print("-" * 70)

cumul = 0.0
for rank, (combo, count) in enumerate(sorted_combos, 1):
    pct = 100.0 * count / len(btns)
    cumul += pct
    label = "+".join(BUTTON_NAMES[i] for i, v in enumerate(combo) if v) or "(none)"
    print(f"{rank:4d}  {count:12,}  {pct:6.2f}%  {cumul:6.2f}%  {list(combo)}  {label}")

print(f"\n{'='*70}")
print("Frequency cutoffs:")
for min_pct in [0.01, 0.05, 0.1, 0.5, 1.0]:
    n = sum(1 for _, c in sorted_combos if 100.0 * c / len(btns) >= min_pct)
    cov = sum(c for _, c in sorted_combos if 100.0 * c / len(btns) >= min_pct) / len(btns) * 100
    print(f"  >= {min_pct:5.2f}% of frames: {n:3d} combos, covering {cov:.2f}% of data")
