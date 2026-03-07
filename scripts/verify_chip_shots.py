#!/usr/bin/env python3
"""verify_chip_shots.py — Test model chip shot (L+B) prediction on expert replay frames.

For each CITF file, finds frames where the subject pressed L+B (chip shot) while
in the chip zone with the ball. Builds the WINDOW_SIZE=8 temporal window used by
the model (with warmup replication on the first active frame, matching AIController
behavior), runs ONNX inference, and reports how often the model correctly predicts
both L and B.

Usage:
    python verify_chip_shots.py <citf_dir> <model.onnx>
    python verify_chip_shots.py <citf_dir> <model.onnx> --discord-id <id>
    python verify_chip_shots.py <citf_dir> <model.onnx> --limit 200
"""

import sys
import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_citf import load_citf_bytes, parse_header, parse_frame
from build_dataset import extract_features, LEFT_TEAM, RIGHT_TEAM

WINDOW_SIZE = 8
FEATURE_DIM = 430
INPUT_DIM   = WINDOW_SIZE * FEATURE_DIM

PAD_B = 0x0200
PAD_L = 0x0040

# Chip zone geometry — matches check_chip_rate.py
CHIP_X_MIN = 11.5
CHIP_X_MAX = 14.5
CHIP_Y_MAX = 6.5

THRESHOLD = 0.5

# Model output indices: A=0, B=1, X=2, Y=3, L=4, R=5
IDX_B = 1
IDX_L = 4


def _autodetect_discord_id(paths):
    id_counter: Counter = Counter()
    for p in paths[:50]:
        try:
            data = load_citf_bytes(str(p))
            hdr  = parse_header(data)
            if hdr.version >= 11 and hdr.submitted_by_discord_id:
                id_counter[hdr.submitted_by_discord_id] += 1
        except Exception:
            pass
    if id_counter:
        discord_id, count = id_counter.most_common(1)[0]
        print(f"Auto-detected Discord ID: {discord_id} (from {count}/50 sampled files)")
        return discord_id
    return None


def _find_port(hdr, discord_id):
    for i, p in enumerate(hdr.port_players):
        if p.discord_id == discord_id:
            return i
    return None


def _in_chip_zone(frame, mirror):
    """True if ball is in chip zone in canonical space (subject always attacks +X)."""
    bpx = (-1.0 if mirror else 1.0) * frame.ball_pos_x
    return CHIP_X_MIN <= bpx <= CHIP_X_MAX and abs(frame.ball_pos_y) <= CHIP_Y_MAX


def main():
    parser = argparse.ArgumentParser(
        description="Verify model chip shot (L+B) prediction on expert replay frames"
    )
    parser.add_argument("citf_dir",   help="Directory containing .citframes files")
    parser.add_argument("model_onnx", help="Path to exported .onnx model file")
    parser.add_argument("--discord-id", type=int, default=None,
                        help="Subject's Discord ID (auto-detected if omitted)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N files (for quick testing)")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed.  pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    sess = ort.InferenceSession(args.model_onnx, providers=["CPUExecutionProvider"])
    print(f"Loaded model: {args.model_onnx}")

    paths = sorted(Path(args.citf_dir).rglob("*.citframes"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"Found {len(paths)} .citframes files")

    discord_id = args.discord_id or _autodetect_discord_id(paths)
    if discord_id is None:
        print("ERROR: could not auto-detect Discord ID. Pass --discord-id explicitly.", file=sys.stderr)
        sys.exit(1)

    # Accumulated per-frame inference results: list of (l_prob, b_prob)
    results = []
    skipped_no_window  = 0  # chip shot frames hit before the window filled (shouldn't happen with warmup)
    skipped_no_subject = 0
    skipped_version    = 0
    errors             = 0
    matched            = 0

    for path in paths:
        try:
            data = load_citf_bytes(str(path))
        except Exception:
            errors += 1
            continue

        try:
            hdr = parse_header(data)
        except Exception:
            errors += 1
            continue

        if hdr.version < 11:
            skipped_version += 1
            continue

        port = _find_port(hdr, discord_id)
        if port is None:
            skipped_no_subject += 1
            continue

        team   = int(hdr.port_teams[port])
        mirror = (team == RIGHT_TEAM)
        matched += 1

        # Circular frame buffer — deque of flat float32 arrays, each FEATURE_DIM long.
        # First active frame replicates to all slots (warmup), matching AIController behavior.
        frame_buf: deque = deque(maxlen=WINDOW_SIZE)

        offset = hdr.header_size
        for _ in range(hdr.frame_count):
            try:
                frame, consumed = parse_frame(data, offset, hdr.fixed_frame_size)
                offset += consumed
            except Exception:
                break

            if frame.game_phase not in (4, 5) or frame.is_paused:
                # Reset buffer at match boundaries (goal celebration, kickoff, etc.)
                frame_buf.clear()
                continue

            try:
                feat = np.array(extract_features(hdr, frame, team), dtype=np.float32)
            except Exception:
                continue

            # Warmup: fill all slots on the first active frame of a sequence
            if len(frame_buf) == 0:
                for _ in range(WINDOW_SIZE):
                    frame_buf.append(feat)
            else:
                frame_buf.append(feat)

            # Only care about L+B chip shot frames where subject has the ball
            btn    = frame.controllers[port].buttons
            l_held = bool(btn & PAD_L)
            b_held = bool(btn & PAD_B)
            if not (l_held and b_held):
                continue

            # Subject must have the ball
            subject_char_ptr = None
            own_slots = [0, 1, 2, 3] if team == LEFT_TEAM else [5, 6, 7, 8]
            for slot in own_slots:
                if frame.characters[slot].is_user_controlled:
                    subject_char_ptr = frame.character_pointers[slot]
                    break
            if subject_char_ptr is None or frame.ball_owner_ptr != subject_char_ptr:
                continue

            if not _in_chip_zone(frame, mirror):
                continue

            # Build [1, INPUT_DIM] input: oldest frame first (left of tensor)
            if len(frame_buf) < WINDOW_SIZE:
                skipped_no_window += 1
                continue

            window = np.concatenate(list(frame_buf)).reshape(1, INPUT_DIM)
            btn_probs, _ = sess.run(None, {"features": window})
            l_prob = float(btn_probs[0, IDX_L])
            b_prob = float(btn_probs[0, IDX_B])
            results.append((l_prob, b_prob))

    # --- Report -----------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"CHIP SHOT INFERENCE VERIFICATION  (Discord ID: {discord_id})")
    print(f"{'='*62}")
    print(f"Files matched:              {matched}")
    print(f"Files skipped (v<11):       {skipped_version}")
    print(f"Files skipped (no subject): {skipped_no_subject}")
    print(f"Parse errors:               {errors}")

    total_found = len(results) + skipped_no_window
    print(f"\nChip shot frames (L+B in chip zone, ball carrier):")
    print(f"  Total found:         {total_found}")
    print(f"  With full window:    {len(results)}")
    print(f"  Skipped (pre-warmup): {skipped_no_window}")

    if not results:
        print("\nNo chip shot frames with full window found.")
        return

    l_probs = np.array([r[0] for r in results])
    b_probs = np.array([r[1] for r in results])
    l_pred  = l_probs >= THRESHOLD
    b_pred  = b_probs >= THRESHOLD
    n = len(results)

    both    = (l_pred & b_pred).sum()
    b_only  = (~l_pred & b_pred).sum()
    l_only  = (l_pred & ~b_pred).sum()
    neither = (~l_pred & ~b_pred).sum()

    print(f"\nPrediction breakdown  (threshold={THRESHOLD}):")
    print(f"  L+B  — chip shot (correct):      {both:>5}  ({100*both/n:.1f}%)")
    print(f"  B only — would fire regular shot: {b_only:>5}  ({100*b_only/n:.1f}%)")
    print(f"  L only — lob held, no shot:       {l_only:>5}  ({100*l_only/n:.1f}%)")
    print(f"  Neither — no shot at all:         {neither:>5}  ({100*neither/n:.1f}%)")

    print(f"\nMean probabilities on chip shot frames:")
    print(f"  L:  {l_probs.mean():.3f}  "
          f"(p10={np.percentile(l_probs,10):.3f}  "
          f"p50={np.percentile(l_probs,50):.3f}  "
          f"p90={np.percentile(l_probs,90):.3f})")
    print(f"  B:  {b_probs.mean():.3f}  "
          f"(p10={np.percentile(b_probs,10):.3f}  "
          f"p50={np.percentile(b_probs,50):.3f}  "
          f"p90={np.percentile(b_probs,90):.3f})")

    print(f"\nL probability distribution on chip shot frames:")
    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        count = int(((l_probs >= lo) & (l_probs < hi)).sum())
        bar   = "#" * int(count / n * 40)
        label = f"[{lo:.1f}-{hi:.1f})"
        print(f"  {label}  {count:>5}  {bar}")


if __name__ == "__main__":
    main()
