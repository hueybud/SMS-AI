"""
generate_manifest.py — One-off tool to create a _manifest.json for a dataset
that was built before manifest support was added.

Scans all *.citframes files, reads their v11 headers, and records every
(epoch, discord_id) pair where discord_id matches the training subject.
This exactly reproduces the seen_keys that build_dataset.py would have written.

Usage:
    python generate_manifest.py <citf_dir> <discord_id> <output_base>

Example:
    python generate_manifest.py "C:/Users/Brian/Downloads/TheSweetieMan CIT Replays" \
        774682411052040202 \
        "C:/Users/Brian/Downloads/sweetieman_dataset"
"""

import sys, json, struct
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from analyze_citf import load_citf_bytes, parse_header


def generate_manifest(citf_dir: str, discord_id: int, output_base: str) -> None:
    citf_paths = sorted(Path(citf_dir).rglob("*.citframes"))
    total = len(citf_paths)
    print(f"Found {total} .citframes files", flush=True)

    seen_keys: list[list[int]] = []
    n_matched = 0
    n_absent  = 0
    n_error   = 0

    for i, path in enumerate(citf_paths):
        if i % 200 == 0 and i > 0:
            print(f"  [{i}/{total}]  matched={n_matched}", flush=True)
        try:
            data = load_citf_bytes(path)
            header = parse_header(data)
        except Exception as e:
            n_error += 1
            continue

        # Check if training subject appears in this CITF's port players
        subject_discord_id = None
        for pp in header.port_players:
            if pp is not None and pp.discord_id == discord_id:
                subject_discord_id = pp.discord_id
                break

        if subject_discord_id is None:
            n_absent += 1
            continue

        seen_keys.append([header.epoch, discord_id])
        n_matched += 1

    seen_keys.sort()

    x_path  = output_base + "_X.npy"
    y_path  = output_base + "_y.npy"
    ms_path = output_base + "_ms.npy"

    manifest = {
        "discord_id": discord_id,
        "build_date": datetime.now(timezone.utc).isoformat(),
        "matched_files": n_matched,
        "total_frames": None,   # unknown — dataset was built before manifest support
        "output_files": {
            "X":  x_path,
            "y":  y_path,
            "ms": ms_path,
        },
        "seen_keys": seen_keys,
    }

    manifest_path = output_base + "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone.")
    print(f"  Matched : {n_matched:,}")
    print(f"  Absent  : {n_absent:,}")
    print(f"  Errors  : {n_error:,}")
    print(f"  Written : {manifest_path}  ({len(seen_keys):,} keys)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    generate_manifest(sys.argv[1], int(sys.argv[2]), sys.argv[3])
