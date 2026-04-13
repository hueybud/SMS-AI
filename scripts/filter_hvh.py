#!/usr/bin/env python3
"""filter_hvh.py -- Filter CITF files to human-vs-human matches only.

A human player is identified by a non-zero discord_id in the v11 header.
A valid HvH match has exactly one human per team (two humans total, on different teams).
Pre-v11 CITFs lack player metadata and are always rejected.

Usage:
    # Audit a folder (summary only)
    python filter_hvh.py <citf_dir>

    # Print each failing file and reason
    python filter_hvh.py <citf_dir> --print-failures

    # Copy passing files to a new directory
    python filter_hvh.py <citf_dir> --copy-to <out_dir>
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_citf import load_citf_bytes, parse_header


def check_hvh(path: Path) -> tuple:
    """Return (passes: bool, reason: str) for a single CITF file."""
    try:
        data = load_citf_bytes(str(path))
    except Exception as e:
        return False, f"read error: {e}"

    try:
        hdr = parse_header(data)
    except Exception as e:
        return False, f"parse error: {e}"

    if hdr.version < 11:
        return False, f"v{hdr.version} header has no player metadata (need v11+)"

    humans = []
    for port_idx, (player, team) in enumerate(zip(hdr.port_players, hdr.port_teams)):
        if player.discord_id != 0:
            humans.append((port_idx, player.discord_id, team))

    if len(humans) != 2:
        return False, f"{len(humans)} human(s) found (expected 2)"

    _, _, team0 = humans[0]
    _, _, team1 = humans[1]

    if team0 == team1:
        ids = [h[1] for h in humans]
        return False, f"both humans ({ids[0]}, {ids[1]}) on same team ({team0})"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Filter CITF files to human-vs-human matches only"
    )
    parser.add_argument("citf_dir", help="Directory to scan recursively for *.citframes files")
    parser.add_argument("--copy-to", metavar="OUT_DIR",
                        help="Copy passing files to this directory")
    parser.add_argument("--print-failures", action="store_true",
                        help="Print each failing file and reason")
    args = parser.parse_args()

    citf_dir = Path(args.citf_dir)
    files = sorted(citf_dir.rglob("*.citframes"))
    if not files:
        print(f"No .citframes files found in {citf_dir}")
        sys.exit(1)

    out_dir = None
    if args.copy_to:
        out_dir = Path(args.copy_to)
        out_dir.mkdir(parents=True, exist_ok=True)

    passed = []
    failed = []

    for i, path in enumerate(files, 1):
        ok, reason = check_hvh(path)
        if ok:
            passed.append(path)
            if out_dir:
                shutil.copy2(path, out_dir / path.name)
        else:
            failed.append((path, reason))

        if i % 50 == 0 or i == len(files):
            print(f"\r  Checked {i}/{len(files)}...", end="", flush=True)

    print()

    print(f"\nResults: {len(passed)}/{len(files)} files pass HvH filter")
    if failed:
        print(f"  Rejected: {len(failed)}")
        if args.print_failures:
            print()
            for path, reason in failed:
                print(f"  FAIL  {path.name}  --  {reason}")

    if out_dir:
        print(f"\nCopied {len(passed)} files to: {out_dir}")


if __name__ == "__main__":
    main()
