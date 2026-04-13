#!/usr/bin/env python3
"""recover_merge.py — Disk-safe merge of build_dataset checkpoints.

Reads the partial _manifest.json produced by build_dataset.py and merges all
checkpoint files into the final output arrays, deleting each checkpoint file
immediately after it is copied into the output.

On Linux, output memmaps start as sparse files (no disk blocks allocated until
written), so peak extra disk usage is ~one checkpoint size (~2.5 GB), not N GB.
This lets you safely merge when free space equals roughly the checkpoint total
rather than twice it.

Usage:
    python recover_merge.py <manifest_json> [--delete-citfs <citf_dir>]

Arguments:
    manifest_json       Path to the *_manifest.json written by build_dataset.py
    --delete-citfs DIR  Delete all *.citframes files in DIR before merging
                        (call this to free input data you no longer need)

Output base path is inferred from the manifest filename:
    foo_manifest.json  ->  foo_X.npy / foo_y.npy / foo_ms.npy / foo_seg.npy
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


def _delete_citfs(citf_dir: str) -> None:
    pattern = os.path.join(citf_dir, "**", "*.citframes")
    files = glob.glob(pattern, recursive=True)
    total_bytes = 0
    for f in files:
        try:
            total_bytes += os.path.getsize(f)
            os.remove(f)
        except OSError as e:
            print(f"  WARN: could not delete {f}: {e}", flush=True)
    print(f"  Deleted {len(files)} .citframes files "
          f"({total_bytes / 1e9:.2f} GB freed)", flush=True)


def merge(manifest_path: str, delete_citfs_dir: str | None = None) -> None:
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    ckpt_paths: list[str] = manifest["checkpoints"]
    if not ckpt_paths:
        print("No checkpoints in manifest — nothing to merge.", flush=True)
        sys.exit(1)

    # Infer output base by stripping _manifest.json suffix
    base = manifest_path
    for suffix in ("_manifest.json", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    x_path   = base + "_X.npy"
    y_path   = base + "_y.npy"
    ms_path  = base + "_ms.npy"
    seg_path = base + "_seg.npy"

    for p in (x_path, y_path, ms_path, seg_path):
        if os.path.exists(p):
            print(f"ERROR: output file already exists: {p}", flush=True)
            print("Delete it first, or rename the output base.", flush=True)
            sys.exit(1)

    # --- Scan checkpoints for total rows and dims ---------------------------
    print("Scanning checkpoints...", flush=True)
    ckpt_rows: list[int] = []
    feature_dim = label_dim = None
    all_ms_arrays:  list[np.ndarray] = []
    all_seg_arrays: list[np.ndarray] = []

    for p in ckpt_paths:
        x_ck = np.load(p + "_X.npy", mmap_mode="r")
        ckpt_rows.append(x_ck.shape[0])
        if feature_dim is None:
            feature_dim = x_ck.shape[1]
            y_ck = np.load(p + "_y.npy", mmap_mode="r")
            label_dim = y_ck.shape[1]
            del y_ck
        all_ms_arrays.append(np.load(p + "_ms.npy"))
        all_seg_arrays.append(np.load(p + "_seg.npy"))
        del x_ck
        print(f"  {Path(p).name}: {ckpt_rows[-1]:,} rows", flush=True)

    total_rows = sum(ckpt_rows)
    x_gb  = total_rows * feature_dim * 2 / 1e9  # float16
    y_gb  = total_rows * label_dim   * 4 / 1e9  # float32
    ckpt_size_gb = (total_rows * feature_dim * 2 + total_rows * label_dim * 4) / 1e9
    per_ckpt_gb  = ckpt_size_gb / len(ckpt_paths)

    print(f"\n  Checkpoints : {len(ckpt_paths)}", flush=True)
    print(f"  Total rows  : {total_rows:,}", flush=True)
    print(f"  Feature dim : {feature_dim}  Label dim: {label_dim}", flush=True)
    print(f"  X.npy size  : {x_gb:.1f} GB  y.npy size: {y_gb:.1f} GB", flush=True)
    print(f"  Peak extra disk needed (sparse): ~{per_ckpt_gb:.1f} GB per step", flush=True)

    # --- Optionally delete CITF inputs first --------------------------------
    if delete_citfs_dir:
        print(f"\nDeleting input CITF files from {delete_citfs_dir} ...", flush=True)
        _delete_citfs(delete_citfs_dir)

    # --- Allocate sparse output memmaps (0 actual disk until written) -------
    print(f"\nAllocating output memmaps (sparse on Linux):", flush=True)
    X_mm = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float16, shape=(total_rows, feature_dim))
    print(f"  X   → {x_path}  ({x_gb:.1f} GB)", flush=True)
    y_mm = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.float32, shape=(total_rows, label_dim))
    print(f"  y   → {y_path}  ({y_gb:.1f} GB)", flush=True)

    # --- Copy each checkpoint then immediately delete it --------------------
    offset = 0
    for i, p in enumerate(ckpt_paths):
        x_ck = np.load(p + "_X.npy", mmap_mode="r")
        y_ck = np.load(p + "_y.npy", mmap_mode="r")
        n = ckpt_rows[i]
        X_mm[offset: offset + n] = x_ck
        y_mm[offset: offset + n] = y_ck
        offset += n
        del x_ck, y_ck

        freed = 0
        for suffix in ("_X.npy", "_y.npy", "_seg.npy", "_ms.npy"):
            fp = p + suffix
            try:
                freed += os.path.getsize(fp)
                os.remove(fp)
            except OSError as e:
                print(f"  WARN: could not delete {fp}: {e}", flush=True)

        print(f"  [{i + 1:>3}/{len(ckpt_paths)}] {Path(p).name}  "
              f"{n:,} rows  freed {freed / 1e9:.2f} GB  "
              f"offset={offset:,}", flush=True)

    del X_mm, y_mm  # flush memmaps to disk

    # --- Save ms and seg (small arrays, fine to concatenate in memory) ------
    ms_combined  = np.concatenate(all_ms_arrays)
    seg_combined = np.concatenate(all_seg_arrays).astype(np.int32)
    np.save(ms_path,  ms_combined)
    np.save(seg_path, seg_combined)
    print(f"\n  ms  → {ms_path}  ({len(ms_combined):,} entries)", flush=True)
    print(f"  seg → {seg_path}  ({len(seg_combined):,} entries)", flush=True)

    # --- Mark manifest complete ---------------------------------------------
    manifest["partial"] = False
    manifest["output_files"] = {
        "X": x_path, "y": y_path, "ms": ms_path, "seg": seg_path,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone.  {total_rows:,} frames × {feature_dim} features.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("manifest_json",
                    help="Path to *_manifest.json from build_dataset.py")
    ap.add_argument("--delete-citfs", metavar="DIR",
                    help="Delete *.citframes files in DIR before merging")
    args = ap.parse_args()

    if not os.path.exists(args.manifest_json):
        print(f"Error: manifest not found: {args.manifest_json}", file=sys.stderr)
        sys.exit(1)

    merge(args.manifest_json, delete_citfs_dir=args.delete_citfs)


if __name__ == "__main__":
    main()
