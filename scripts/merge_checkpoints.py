"""
merge_checkpoints.py — Merge two or more build_dataset.py output datasets into one.

Each dataset is identified by its base path (the part before _X.npy / _y.npy /
_ms.npy / _seg.npy).  Datasets are written sequentially into memory-mapped output
files so RAM usage stays flat regardless of total size.

Two offsets are applied automatically so the merged dataset is internally consistent:
  _ms.npy  — match-start frame indices are shifted by the cumulative frame count of
              all preceding datasets, keeping them valid as absolute indices into the
              merged X / y arrays.
  _seg.npy — segment IDs are shifted by the cumulative max segment ID of all
              preceding datasets, ensuring every segment ID is globally unique.
              train.py relies on unique IDs to build non-overlapping LSTM windows.

Usage:
  python merge_checkpoints.py <output_base> <dataset_A> <dataset_B> [<dataset_C> ...]

Example:
  python merge_checkpoints.py "C:/SMS AI/datasets/sweetieman_merged" \
      "C:/SMS AI/datasets/sweetieman_v3" \
      "C:/SMS AI/datasets/sweetieman_v4"

Output files (all next to <output_base>):
  <output_base>_X.npy
  <output_base>_y.npy
  <output_base>_ms.npy
  <output_base>_seg.npy
"""

import argparse
import numpy as np
from pathlib import Path


def _load_info(base: str) -> dict:
    """Return shape / size metadata for a dataset without loading arrays."""
    X   = np.load(base + "_X.npy",   mmap_mode="r")
    seg = np.load(base + "_seg.npy", mmap_mode="r")
    ms  = np.load(base + "_ms.npy",  mmap_mode="r")
    info = {
        "n_frames":   X.shape[0],
        "feature_dim": X.shape[1],
        "label_dim":  np.load(base + "_y.npy", mmap_mode="r").shape[1],
        "n_matches":  len(ms),
        "max_seg":    int(seg.max()) if len(seg) else 0,
    }
    return info


def merge_datasets(output_base: str, input_bases: list[str]) -> None:
    # ---- Pass 1: validate shapes and collect totals -------------------------
    print("Pass 1: scanning datasets...", flush=True)
    infos = []
    feature_dim = None
    label_dim   = None
    for base in input_bases:
        info = _load_info(base)
        print(f"  {Path(base).name}: {info['n_frames']:,} frames, "
              f"{info['n_matches']} matches, "
              f"seg_max={info['max_seg']:,}", flush=True)
        if feature_dim is None:
            feature_dim = info["feature_dim"]
            label_dim   = info["label_dim"]
        else:
            if info["feature_dim"] != feature_dim or info["label_dim"] != label_dim:
                raise ValueError(
                    f"Shape mismatch in {base}: "
                    f"expected X={feature_dim} y={label_dim}, "
                    f"got X={info['feature_dim']} y={info['label_dim']}"
                )
        infos.append(info)

    total_frames  = sum(i["n_frames"]  for i in infos)
    total_matches = sum(i["n_matches"] for i in infos)
    print(f"\nTotal: {total_frames:,} frames across {total_matches:,} matches",
          flush=True)

    # ---- Pre-allocate output memmaps ----------------------------------------
    x_path   = output_base + "_X.npy"
    y_path   = output_base + "_y.npy"
    ms_path  = output_base + "_ms.npy"
    seg_path = output_base + "_seg.npy"

    print(f"\nPre-allocating output files...", flush=True)
    X_out   = np.lib.format.open_memmap(
        x_path,   mode="w+", dtype=np.float32, shape=(total_frames, feature_dim))
    y_out   = np.lib.format.open_memmap(
        y_path,   mode="w+", dtype=np.float32, shape=(total_frames, label_dim))
    seg_out = np.lib.format.open_memmap(
        seg_path, mode="w+", dtype=np.int32,   shape=(total_frames,))
    print(f"  X   {total_frames:,} × {feature_dim}  "
          f"({total_frames * feature_dim * 4 / 1e9:.1f} GB)", flush=True)
    print(f"  y   {total_frames:,} × {label_dim}", flush=True)
    print(f"  seg {total_frames:,}", flush=True)

    # ---- Pass 2: copy each dataset in with offsets --------------------------
    print("\nPass 2: merging datasets...", flush=True)
    frame_offset = 0
    seg_offset   = 0
    all_ms       = []

    for base, info in zip(input_bases, infos):
        n = info["n_frames"]

        X_src   = np.load(base + "_X.npy",   mmap_mode="r")
        y_src   = np.load(base + "_y.npy",   mmap_mode="r")
        seg_src = np.load(base + "_seg.npy", mmap_mode="r")
        ms_src  = np.load(base + "_ms.npy",  mmap_mode="r")

        X_out  [frame_offset:frame_offset + n] = X_src
        y_out  [frame_offset:frame_offset + n] = y_src
        seg_out[frame_offset:frame_offset + n] = seg_src + seg_offset

        all_ms.append(ms_src + frame_offset)

        print(f"  {Path(base).name}: rows {frame_offset:,}–{frame_offset + n - 1:,}  "
              f"seg offset +{seg_offset:,}  ms offset +{frame_offset:,}", flush=True)

        frame_offset += n
        seg_offset   += info["max_seg"]

    del X_out, y_out, seg_out  # flush to disk

    ms_combined = np.concatenate(all_ms)
    np.save(ms_path, ms_combined)

    # ---- Summary ------------------------------------------------------------
    print(f"\nOutput files written:")
    for p in (x_path, y_path, ms_path, seg_path):
        size = Path(p).stat().st_size
        label = f"{size / 1e9:.2f} GB" if size > 1e8 else f"{size:,} bytes"
        print(f"  {p}  ({label})", flush=True)

    avg = total_frames // max(total_matches, 1)
    print(f"\nSummary: {total_frames:,} frames, {total_matches:,} matches "
          f"(avg {avg:,} frames/match)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge two or more LSTM build_dataset.py outputs into one dataset"
    )
    parser.add_argument("output_base",
                        help="Output base path (suffixes _X/_y/_ms/_seg.npy added)")
    parser.add_argument("input_bases", nargs="+",
                        help="Two or more dataset base paths to merge (in order)")
    args = parser.parse_args()

    if len(args.input_bases) < 2:
        parser.error("Provide at least two input dataset base paths to merge.")

    merge_datasets(args.output_base, args.input_bases)
