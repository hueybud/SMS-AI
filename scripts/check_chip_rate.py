#!/usr/bin/env python3
"""
check_chip_rate.py — Analyze chip shot rate from chip-eligible positions.

For each CITF where the subject player has the ball inside the box chip zone
(abs(ball_pos_x) in 11.5-14.5, |ball_pos_y| <= 6.5, attacking opponent goal),
counts how often they press L+B (chip shot) vs. other actions.

Usage:
    python check_chip_rate.py <citf_dir>
    python check_chip_rate.py <citf_dir> --discord-id <id>
    python check_chip_rate.py <citf_dir> --limit 200
"""

import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_citf import load_citf_bytes, parse_header, parse_frame, CaptureHeader

# Button bits — from Dolphin GCPadStatus (matches build_dataset.py)
PAD_B = 0x0200   # shoot / slide
PAD_L = 0x0040   # modifier: L+B = chip shot, L+A = lob pass

# Box chip zone geometry (from classify_shot in analyze_citf.py)
CHIP_X_MIN = 11.5
CHIP_X_MAX = 14.5
CHIP_Y_MAX = 6.5


def find_subject_port(header: CaptureHeader, discord_id: int):
    for i, p in enumerate(header.port_players):
        if p.discord_id == discord_id:
            return i
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Measure chip shot rate from chip-eligible positions in CITFs"
    )
    parser.add_argument("citf_dir", help="Directory containing .citframes files")
    parser.add_argument("--discord-id", type=int, default=None,
                        help="Subject's Discord ID (auto-detected if omitted)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N files (for quick testing)")
    args = parser.parse_args()

    citf_dir = Path(args.citf_dir)
    paths = sorted(citf_dir.rglob("*.citframes"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"Found {len(paths)} .citframes files", flush=True)

    # --- Auto-detect Discord ID -------------------------------------------------
    discord_id = args.discord_id
    if discord_id is None:
        id_counter: Counter = Counter()
        for p in paths[:50]:
            try:
                data = load_citf_bytes(str(p))
                hdr = parse_header(data)
                if hdr.version >= 11 and hdr.submitted_by_discord_id:
                    id_counter[hdr.submitted_by_discord_id] += 1
            except Exception:
                pass
        if id_counter:
            discord_id, count = id_counter.most_common(1)[0]
            print(f"Auto-detected Discord ID: {discord_id} (from {count}/50 sampled files)")
        else:
            print("ERROR: could not auto-detect Discord ID", file=sys.stderr)
            sys.exit(1)

    # --- Scan frames ------------------------------------------------------------
    # Chip zone = subject has the ball, is in the box chip X/Y range, attacking opponent goal.

    total_chip_zone_frames = 0
    action_counts: Counter = Counter()   # what button combo was pressed

    # Breakdown of what happens on B-press frames specifically
    b_press_with_l  = 0   # L+B: chip shot
    b_press_no_l    = 0   # B only: regular/charged shot

    matched = 0
    skipped_version = 0
    skipped_no_subject = 0
    errors = 0

    for path_idx, path in enumerate(paths):
        try:
            data = load_citf_bytes(str(path))
        except Exception as e:
            errors += 1
            continue

        try:
            hdr = parse_header(data)
        except Exception as e:
            errors += 1
            continue

        if hdr.version < 11:
            skipped_version += 1
            continue

        port = find_subject_port(hdr, discord_id)
        if port is None:
            skipped_no_subject += 1
            continue

        team = hdr.port_teams[port]  # 0=left, 1=right
        if team not in (0, 1):
            skipped_no_subject += 1
            continue

        matched += 1
        if matched % 200 == 0:
            print(f"  {matched} matched files, {total_chip_zone_frames:,} chip-zone frames so far...",
                  flush=True)

        offset = hdr.header_size
        for _ in range(hdr.frame_count):
            try:
                frame, consumed = parse_frame(data, offset, hdr.fixed_frame_size)
                offset += consumed
            except Exception:
                break

            if frame.game_phase not in (4, 5) or frame.is_paused:
                continue

            # Find which character slot the subject controls
            own_slots = list(range(0, 4)) if team == 0 else list(range(5, 9))
            subject_char_ptr = None
            for slot in own_slots:
                if frame.characters[slot].is_user_controlled:
                    subject_char_ptr = frame.character_pointers[slot]
                    break

            if subject_char_ptr is None:
                continue

            # Subject must be holding the ball
            if frame.ball_owner_ptr == 0 or frame.ball_owner_ptr != subject_char_ptr:
                continue

            bx = frame.ball_pos_x
            by = frame.ball_pos_y

            # Ball must be in box chip X range (near OPPONENT goal, not own goal)
            if not (CHIP_X_MIN <= abs(bx) <= CHIP_X_MAX):
                continue

            # Ball must be in Y (width) range
            if abs(by) > CHIP_Y_MAX:
                continue

            # Must be attacking the correct goal:
            #   left team (0) attacks positive X; right team (1) attacks negative X
            if team == 0 and bx < 0:
                continue
            if team == 1 and bx > 0:
                continue

            # In chip zone with ball — classify input
            btn = frame.controllers[port].buttons
            l_held = bool(btn & PAD_L)
            b_held = bool(btn & PAD_B)

            total_chip_zone_frames += 1

            if l_held and b_held:
                action_counts["L+B  (chip shot)"] += 1
                b_press_with_l += 1
            elif b_held:
                action_counts["B    (regular shot)"] += 1
                b_press_no_l += 1
            elif l_held:
                action_counts["L    (lob modifier, no shot)"] += 1
            else:
                action_counts["---- (dribbling / no action)"] += 1

    # --- Results ----------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"CHIP RATE ANALYSIS  (Discord ID: {discord_id})")
    print(f"{'='*55}")
    print(f"Files matched (subject found):  {matched}")
    print(f"Files skipped (no subject):     {skipped_no_subject}")
    print(f"Files skipped (pre-v11):        {skipped_version}")
    print(f"Parse errors:                   {errors}")
    print(f"\nChip-zone frames total:         {total_chip_zone_frames:,}")
    print(f"  (ball carrier, abs(x) {CHIP_X_MIN}-{CHIP_X_MAX}, |y| <= {CHIP_Y_MAX}, attacking goal)")

    if total_chip_zone_frames == 0:
        print("\nNo chip-zone frames found. Check discord ID and file version.")
        return

    print(f"\nInput breakdown while in chip zone:")
    print(f"  {'Action':<35} {'Frames':>8}  {'%':>6}")
    print(f"  {'-'*35} {'-'*8}  {'-'*6}")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / total_chip_zone_frames * 100
        print(f"  {action:<35} {count:>8,}  {pct:>5.1f}%")

    print()
    chip_zone_chip_rate  = action_counts["L+B  (chip shot)"] / total_chip_zone_frames * 100
    chip_zone_shot_rate  = (b_press_with_l + b_press_no_l) / total_chip_zone_frames * 100

    print(f"Chip rate (L+B / all chip-zone frames):    {chip_zone_chip_rate:.1f}%")
    print(f"Shot rate (any B / all chip-zone frames):  {chip_zone_shot_rate:.1f}%")
    if b_press_with_l + b_press_no_l > 0:
        chip_of_shots = b_press_with_l / (b_press_with_l + b_press_no_l) * 100
        print(f"Chip%% of shots fired from chip zone:       {chip_of_shots:.1f}%")


if __name__ == "__main__":
    main()
