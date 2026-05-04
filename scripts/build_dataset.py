#!/usr/bin/env python3
"""build_dataset.py — Build ML training dataset from v11 CITF files.

Usage:
    python build_dataset.py <citf_dir> <output> --experts 774682411052040202,9876543210
    python build_dataset.py <citf_dir> <output> --experts-file experts.txt [--stats] [--resume <manifest.json>]

Arguments:
    citf_dir          Directory to search recursively for *.citframes files
    output            Output base path (suffixes _X.npy / _y.npy / _ms.npy / _seg.npy /
                      _manifest.json added)
    --experts         Comma-separated list of allowlisted expert discord IDs
    --experts-file    Path to a text file with one discord ID per line (alternative to --experts)
    --stats           Print feature and label statistics after building
    --resume          Path to an existing _manifest.json to skip already-processed games

Output files:
    <output>_X.npy            float16 (N, FEATURE_DIM) — input features per frame
    <output>_y.npy            float32 (N, LABEL_DIM)   — controller labels per frame
    <output>_ms.npy           int64   (M,)              — match start frame indices
    <output>_seg.npy          int32   (N,)              — segment ID per frame
                                                          (monotonically increasing;
                                                          new ID on match start, frame gap,
                                                          or kickoff↔active transition)
    <output>_manifest.json    JSON record of all processed game keys and expert IDs

Deduplication: keyed on (room_id, uuidv5(sorted goal timestamps)).
    If the same game is submitted by multiple players, it is processed exactly once.
    From that one game, inputs are extracted for EVERY allowlisted expert port found.
    Games with no allowlisted expert on either side are skipped entirely.

Frame filter:  gamePhase in {1, 4, 5} (kickoff + active play) and not isPaused.
               Goalie-controlled frames are included via a goalie-compatible feature encoding.
Canonical mirror: if expert's team attacks left, all X coords and velocities
                  are negated so the model always sees "attacks right".
"""

import os, sys, math, argparse, struct, json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

# ---- Import shared parsing layer from analyze_citf.py ----------------------
sys.path.insert(0, str(Path(__file__).parent))
from analyze_citf import (
    load_citf_bytes,
    parse_header,
    parse_frame,
    CaptureHeader,
    GameStateFrame,
)

# ---- Action state vocabs ----------------------------------------------------
# Each vocab defines known states in a fixed order. Unknown states map to the
# last bucket ("other"). Vocab length + 1 = one-hot dimension.

_STRIKER_VOCAB_LIST = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0xFFFFFFFF,
]  # 29 known
_GOALIE_VOCAB_LIST = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19,
]  # 26 known

STRIKER_STATE_IDX = {s: i for i, s in enumerate(_STRIKER_VOCAB_LIST)}
GOALIE_STATE_IDX  = {s: i for i, s in enumerate(_GOALIE_VOCAB_LIST)}
STRIKER_STATE_DIM = len(_STRIKER_VOCAB_LIST) + 1  # 30 (29 known + 1 other)
GOALIE_STATE_DIM  = len(_GOALIE_VOCAB_LIST)  + 1  # 27 (26 known + 1 other)
# v6: Action states stored as integer indices (embedded in model, not one-hot in features).
STRIKER_OTHER_IDX = STRIKER_STATE_DIM - 1  # 29
GOALIE_OTHER_IDX  = GOALIE_STATE_DIM  - 1  # 26

# ---- Other encoding constants -----------------------------------------------

EFFECT_DIM    = 5   # none(0) frozen(1) on_fire(2) star(3) electrocuted(4)
SPEED_ITEM_DIM = 3  # none(0) mushroom(1) star(2)
_SPEED_ITEM_MAP = {0: 0, 7: 1, 8: 2}

POWERUP_DIM = 10    # types 0-8 plus empty(-1 → index 9)
_POWERUP_MAP = {-1: 9, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}

# v6: Field item type indices (for embedding in model). 9 types + 1 padding = 10.
FIELD_ITEM_TYPES = 10  # types 0-8 + padding index 9
FIELD_ITEM_K     = 4   # top-K nearest items as entity tokens
FIELD_ITEM_DIM   = 8   # per-item: type_idx(1) + pos_delta(3) + vel(3) + strength(1)

PAD_A = 0x0100
PAD_B = 0x0200
PAD_X = 0x0400
PAD_Y = 0x0800
PAD_L = 0x0040  # PAD_TRIGGER_L — lob pass / chip shot
PAD_R = 0x0020  # PAD_TRIGGER_R — turbo / sprint

LEFT_TEAM  = 0
RIGHT_TEAM = 1

_LEFT_STRIKER_SLOTS  = [0, 1, 2, 3]
_RIGHT_STRIKER_SLOTS = [5, 6, 7, 8]
_LEFT_GOALIE_SLOT    = 4
_RIGHT_GOALIE_SLOT   = 9

# ---- Feature / label dimensions (v6) -----------------------------------------
#
# v6 changes from v5:
#   - Action states: one-hot → integer index (1 float per character, embedded in model)
#   - Ball: +3 (ball-to-goal distance/angle)
#   - Context: -2 (removed score_diff, time_fraction), +2 (is_kickoff, goalie_has_ball)
#   - Labels: L replaced by lob_pass (L∧A) and chip_shot (L∧B) → 7 buttons
#   - Field items: 4 nearest items as features (K=4, 8 dims each)
#   - prev_labels: 11 (7 buttons + 4 sticks)
#
# Ball (absolute, post-mirror):                 11
#   pos(3) + vel(3) + charge(1) + perfect_pass(1) + ball_to_goal(3)
# Ball owner one-hot (slots 0-9 + none=10):     11
# Self character:                                19
#   pos_delta(3) + state_idx(1) + heading(2) + goal_dist_angle(3)
#   + effect(5) + speed_item(3) + item_timer(1) + is_ball_carrier(1)
#   (when goalie: state_idx uses goalie vocab; speed_item/timer = 0)
# 3 other friendly strikers × 7:                21
#   pos_delta(3) + state_idx(1) + heading(2) + is_ball_carrier(1)
# Friendly goalie:                                4
#   pos_delta(3) + state_idx(1)
# 4 enemy strikers × 7:                         28
# Enemy goalie:                                   4
# Own inventory 2 slots × 11:                   22
#   powerup_type_oh(10) + charge_count(1)
# Enemy inventory 2 slots × 11:                 22
# 4 nearest field items × 8:                    32
#   type_idx(1) + pos_delta(3) + vel(3) + strength(1)
# Tactical summary:                               5
#   self_dist/40 + self_rank/3 + is_nearest + nearest_enemy_dist/40 + enemy_in_path
# Possession booleans:                            2
#   friendly_has_ball + enemy_has_ball
# Phase booleans:                                 2
#   is_kickoff + goalie_has_ball
# Previous frame action (buttons + sticks):      11
#   A, B, X, Y, lob_pass, chip_shot, R (0/1) + stick_x, stick_y, cstick_x, cstick_y
#   Zeroed at segment boundaries
# ──────────────────────────────────────────────────
# Total:                                        194

BUTTON_DIM      = 7    # A, B, X, Y, lob_pass, chip_shot, R
STICK_DIM       = 4    # stick_x, stick_y, cstick_x, cstick_y
PREV_ACTION_DIM = BUTTON_DIM + STICK_DIM  # 11
CORE_FEATURE_DIM = 183  # everything except prev_labels
FEATURE_DIM     = CORE_FEATURE_DIM + PREV_ACTION_DIM  # 194
LABEL_DIM       = BUTTON_DIM + STICK_DIM  # 11


# ---- One-hot helpers ---------------------------------------------------------

def _one_hot(idx: int, dim: int) -> List[float]:
    v = [0.0] * dim
    if 0 <= idx < dim:
        v[idx] = 1.0
    return v


def _striker_state_oh(state: int) -> List[float]:
    idx = STRIKER_STATE_IDX.get(state, len(_STRIKER_VOCAB_LIST))  # unknown → last bucket
    return _one_hot(idx, STRIKER_STATE_DIM)


def _goalie_state_oh(state: int) -> List[float]:
    idx = GOALIE_STATE_IDX.get(state, len(_GOALIE_VOCAB_LIST))
    return _one_hot(idx, GOALIE_STATE_DIM)


def _striker_state_idx(state: int) -> float:
    """v6: Return integer index as float (embedded in model, not one-hot)."""
    return float(STRIKER_STATE_IDX.get(state, STRIKER_OTHER_IDX))


def _goalie_state_idx(state: int) -> float:
    """v6: Return integer index as float (embedded in model, not one-hot)."""
    return float(GOALIE_STATE_IDX.get(state, GOALIE_OTHER_IDX))


def _effect_oh(effect: int) -> List[float]:
    return _one_hot(min(max(effect, 0), EFFECT_DIM - 1), EFFECT_DIM)


def _speed_item_oh(item_type: int) -> List[float]:
    return _one_hot(_SPEED_ITEM_MAP.get(item_type, 0), SPEED_ITEM_DIM)


def _powerup_oh(ptype: int) -> List[float]:
    return _one_hot(_POWERUP_MAP.get(ptype, 9), POWERUP_DIM)


def _heading_sincos(heading_u16: int) -> Tuple[float, float]:
    angle = (heading_u16 / 65536.0) * 2.0 * math.pi
    return math.sin(angle), math.cos(angle)


# ---- Subject / team identification ------------------------------------------

def _is_human_vs_human(header: CaptureHeader) -> Tuple[bool, str]:
    """Return (ok, reason). Requires v11+ header.
    Passes if exactly one human (non-zero discord_id) is on each team.
    """
    humans = []
    for port_idx, (player, team) in enumerate(zip(header.port_players, header.port_teams)):
        if player.discord_id != 0:
            humans.append((port_idx, player.discord_id, team))
    if len(humans) != 2:
        return False, f"{len(humans)} human(s) found (expected 2)"
    if humans[0][2] == humans[1][2]:
        return False, f"both humans on same team ({humans[0][2]})"
    return True, "ok"


def _find_subject_port(header: CaptureHeader, discord_id: int) -> Optional[int]:
    """Return GC port index (0-3) for the training subject, or None if absent."""
    for i, player in enumerate(header.port_players):
        if player.discord_id == discord_id:
            return i
    return None


def _get_subject_team(header: CaptureHeader, port: int) -> int:
    """Return LEFT_TEAM (0) or RIGHT_TEAM (1) for the given port."""
    return int(header.port_teams[port])


def _striker_slots(team: int) -> List[int]:
    return _LEFT_STRIKER_SLOTS if team == LEFT_TEAM else _RIGHT_STRIKER_SLOTS


def _goalie_slot(team: int) -> int:
    return _LEFT_GOALIE_SLOT if team == LEFT_TEAM else _RIGHT_GOALIE_SLOT


def _enemy_striker_slots(team: int) -> List[int]:
    return _RIGHT_STRIKER_SLOTS if team == LEFT_TEAM else _LEFT_STRIKER_SLOTS


def _enemy_goalie_slot(team: int) -> int:
    return _RIGHT_GOALIE_SLOT if team == LEFT_TEAM else _LEFT_GOALIE_SLOT


def _find_self_slot(frame: GameStateFrame, team: int) -> Tuple[int, bool]:
    """Return (slot, is_goalie) for the character the subject currently controls.

    is_goalie=True when the player has taken over the goalie slot (e.g. after a
    diving save). Features are built with a goalie-compatible self-block in that case.
    Falls back to the first striker slot only if no character is flagged controlled,
    which can briefly occur on kickoff transitions.
    """
    for slot in _striker_slots(team):
        if frame.characters[slot].is_user_controlled:
            return slot, False
    gs = _goalie_slot(team)
    if frame.characters[gs].is_user_controlled:
        return gs, True
    return _striker_slots(team)[0], False


def _resolve_owner_slot(frame: GameStateFrame) -> int:
    """Return the character slot that owns the ball, or 10 (no owner)."""
    if frame.ball_owner_ptr == 0:
        return 10
    for i, ptr in enumerate(frame.character_pointers):
        if ptr == frame.ball_owner_ptr:
            return i
    return 10


# ---- Game-level deduplication -----------------------------------------------

def _compute_game_key(header: CaptureHeader,
                      frames: List[GameStateFrame]) -> str:
    """Compute a canonical game key stable across duplicate submissions.

    Key: "<room_id_hex>:<t1:.3f>,<t2:.3f>,..."

    Goal timestamps are derived from score-counter increments — the same
    method used in validate_goals.py.  Using score increments (rather than
    phase transitions) keeps the detection logic identical to what the
    validation pipeline already verified, and game_time values are formatted
    to 3 decimal places for a stable string representation.

    Two submissions of the same game will have the same room_id and the same
    sorted goal times, so their keys are guaranteed equal.
    """
    times = []
    for i in range(1, len(frames)):
        if frames[i].left_score > frames[i - 1].left_score:
            times.append(frames[i].game_time)
        if frames[i].right_score > frames[i - 1].right_score:
            times.append(frames[i].game_time)
    times.sort()
    goals_str = ",".join(f"{t:.3f}" for t in times)
    return f"{header.room_id:08X}:{goals_str}"


def _find_expert_ports(header: CaptureHeader,
                       expert_ids: Set[int]) -> List[Tuple[int, int]]:
    """Return [(port_index, discord_id), ...] for each allowlisted expert in this game."""
    result = []
    for port, player in enumerate(header.port_players):
        if player.discord_id in expert_ids:
            result.append((port, player.discord_id))
    return result


# ---- Feature extraction ------------------------------------------------------

def extract_features(header: CaptureHeader,
                     frame: GameStateFrame,
                     subject_team: int,
                     self_slot: int,
                     is_goalie: bool) -> List[float]:
    """Build the feature vector for one active-play frame (v6).

    v6 changes from v5:
      - Action states: integer index (1 float) instead of one-hot (27-30 floats)
      - Ball: +3 ball-to-goal distance/angle
      - Removed: score_diff, time_fraction
      - Added: is_kickoff, goalie_has_ball, 4 nearest field items
      - Labels: 7 buttons (lob_pass/chip_shot replace standalone L)

    Canonical mirror is applied when subject_team == RIGHT_TEAM.
    """
    mirror = (subject_team == RIGHT_TEAM)
    sign   = -1.0 if mirror else 1.0

    def mx(x: float) -> float:
        return sign * x

    # Ball state (mirrored X)
    bpx = mx(frame.ball_pos_x)
    bpy = frame.ball_pos_y
    bpz = frame.ball_pos_z
    bvx = mx(frame.ball_vel_x)
    bvy = frame.ball_vel_y
    bvz = frame.ball_vel_z

    # Attacking goal X (always positive after canonical mirror)
    goal_x = abs(header.goal_line_x)

    owner_slot = _resolve_owner_slot(frame)

    feat: List[float] = []

    # --- 1. Ball (11) ---
    feat += [bpx, bpy, bpz, bvx, bvy, bvz,
             frame.ball_charge_amount / 35.0,
             float(frame.is_perfect_pass)]
    # Ball-to-goal geometry (v6)
    b2g_dx   = goal_x - bpx
    b2g_dy   = 0.0 - bpy  # goal center is at y=0
    b2g_dist = math.sqrt(b2g_dx * b2g_dx + b2g_dy * b2g_dy) + 1e-6
    feat += [b2g_dist / 40.0, b2g_dx / b2g_dist, b2g_dy / b2g_dist]

    # --- 2. Ball owner one-hot (11: slots 0-9 + none=10) ---
    feat += _one_hot(owner_slot, 11)

    # --- 3. Self character (19) ---
    sc    = frame.characters[self_slot]
    sc_px = mx(sc.pos_x)
    sc_py = sc.pos_y
    dx    = goal_x - sc_px
    dy    = sc_py
    dist  = math.sqrt(dx * dx + dy * dy) + 1e-6
    sin_h, cos_h = _heading_sincos(sc.heading)
    if mirror:
        cos_h = -cos_h

    feat += [sc_px - bpx, sc_py - bpy, sc.pos_z]            # Δpos to ball (3)
    if is_goalie:
        feat += [_goalie_state_idx(sc.action_state)]          # state index (1)
    else:
        feat += [_striker_state_idx(sc.action_state)]         # state index (1)
    feat += [sin_h, cos_h]                                    # heading (2)
    feat += [dist / 40.0, dx / dist, dy / dist]               # goal dist+angle (3)
    feat += _effect_oh(sc.effect_type)                        # effect (5)
    if is_goalie:
        feat += [0.0, 0.0, 0.0]                               # speed_item (3, zero for goalie)
        feat += [0.0]                                          # item_timer (1, zero for goalie)
    else:
        feat += _speed_item_oh(sc.speed_item_type)            # speed item (3)
        feat += [sc.speed_item_timer / 10.0]                  # item timer (1)
    feat += [float(self_slot == owner_slot)]                   # is ball carrier (1)

    # --- 4. 3 other friendly strikers, sorted by distance to ball (7 × 3 = 21) ---
    all_friendly = _striker_slots(subject_team)
    others = all_friendly if is_goalie else [s for s in all_friendly if s != self_slot]
    others = sorted(others, key=lambda s: (mx(frame.characters[s].pos_x) - bpx) ** 2
                                          + (frame.characters[s].pos_y - bpy) ** 2)
    for slot in others[:3]:
        ch    = frame.characters[slot]
        ch_px = mx(ch.pos_x)
        sh, ch_ = _heading_sincos(ch.heading)
        if mirror:
            ch_ = -ch_
        feat += [ch_px - bpx, ch.pos_y - bpy, ch.pos_z]    # Δpos (3)
        feat += [_striker_state_idx(ch.action_state)]        # state index (1)
        feat += [sh, ch_]                                    # heading (2)
        feat += [float(slot == owner_slot)]                  # is ball carrier (1)
    for _ in range(3 - len(others[:3])):
        feat += [0.0] * 7

    # --- 5. Friendly goalie (4) ---
    gs = frame.characters[_goalie_slot(subject_team)]
    feat += [mx(gs.pos_x) - bpx, gs.pos_y - bpy, gs.pos_z]  # Δpos (3)
    feat += [_goalie_state_idx(gs.action_state)]               # state index (1)

    # --- 6. 4 enemy strikers, sorted by distance to ball (7 × 4 = 28) ---
    enemies = sorted(_enemy_striker_slots(subject_team),
                     key=lambda s: (mx(frame.characters[s].pos_x) - bpx) ** 2
                                   + (frame.characters[s].pos_y - bpy) ** 2)
    for slot in enemies[:4]:
        ch    = frame.characters[slot]
        ch_px = mx(ch.pos_x)
        sh, ch_ = _heading_sincos(ch.heading)
        if mirror:
            ch_ = -ch_
        feat += [ch_px - bpx, ch.pos_y - bpy, ch.pos_z]    # Δpos (3)
        feat += [_striker_state_idx(ch.action_state)]        # state index (1)
        feat += [sh, ch_]                                    # heading (2)
        feat += [float(slot == owner_slot)]                  # is ball carrier (1)

    # --- 7. Enemy goalie (4) ---
    eg = frame.characters[_enemy_goalie_slot(subject_team)]
    feat += [mx(eg.pos_x) - bpx, eg.pos_y - bpy, eg.pos_z]  # Δpos (3)
    feat += [_goalie_state_idx(eg.action_state)]               # state index (1)

    # --- 8. Own inventory (2 slots × 11 = 22) ---
    own_inv = frame.left_inventory if subject_team == LEFT_TEAM else frame.right_inventory
    for slot in own_inv:
        feat += _powerup_oh(slot.type)                       # type (10)
        feat += [slot.charge_count / 5.0]                    # count (1)

    # --- 9. Enemy inventory (2 slots × 11 = 22) ---
    enemy_inv = frame.right_inventory if subject_team == LEFT_TEAM else frame.left_inventory
    for slot in enemy_inv:
        feat += _powerup_oh(slot.type)
        feat += [slot.charge_count / 5.0]

    # --- 10. 4 nearest field items (8 × 4 = 32) ---
    # Items sorted by distance to controlled character. Unused slots zero-padded.
    items_with_dist = []
    for item in frame.items:
        ipx = mx(item.pos_x)
        ipy = item.pos_y
        d2 = (ipx - sc_px) ** 2 + (ipy - sc_py) ** 2
        items_with_dist.append((d2, item, ipx, ipy))
    items_with_dist.sort(key=lambda t: t[0])
    for k in range(FIELD_ITEM_K):
        if k < len(items_with_dist):
            _, item, ipx, ipy = items_with_dist[k]
            feat += [float(min(item.powerup_type, 8))]       # type index (1), clamp to 0-8
            feat += [ipx - sc_px, ipy - sc_py, item.pos_z]      # Δpos to self (3)
            feat += [mx(item.vel_x), item.vel_y, item.vel_z]    # velocity (3)
            feat += [item.strength_level / 2.0]                  # strength (1)
        else:
            feat += [9.0]         # padding type index (indicates empty slot)
            feat += [0.0] * 7    # zero-pad remaining dims

    # --- 11. Tactical summary (5) ---
    self_dist_to_ball = math.sqrt((sc_px - bpx) ** 2 + (sc_py - bpy) ** 2)
    self_rank = sum(
        1 for s in _striker_slots(subject_team)
        if s != self_slot and
           (mx(frame.characters[s].pos_x) - bpx) ** 2 + (frame.characters[s].pos_y - bpy) ** 2
           < self_dist_to_ball ** 2
    )
    en_ch              = frame.characters[enemies[0]]
    en_px              = mx(en_ch.pos_x)
    en_dx              = en_px - sc_px
    en_dy              = en_ch.pos_y - sc_py
    nearest_enemy_dist = math.sqrt(en_dx ** 2 + en_dy ** 2)
    goal_dx            = goal_x - sc_px
    goal_dy            = 0.0 - sc_py
    goal_len           = math.sqrt(goal_dx ** 2 + goal_dy ** 2) + 1e-6
    en_len             = nearest_enemy_dist + 1e-6
    enemy_in_path = (goal_dx / goal_len) * (en_dx / en_len) + (goal_dy / goal_len) * (en_dy / en_len)
    feat += [self_dist_to_ball / 40.0]
    feat += [self_rank / 3.0]
    feat += [float(self_rank == 0)]
    feat += [nearest_enemy_dist / 40.0]
    feat += [enemy_in_path]

    # --- 12. Possession booleans (2) ---
    friendly_slots = set(_LEFT_STRIKER_SLOTS + [_LEFT_GOALIE_SLOT]) if subject_team == LEFT_TEAM \
                     else set(_RIGHT_STRIKER_SLOTS + [_RIGHT_GOALIE_SLOT])
    feat += [float(owner_slot in friendly_slots)]            # friendly_has_ball
    feat += [float(owner_slot != 10 and owner_slot not in friendly_slots)]  # enemy_has_ball

    # --- 13. Phase booleans (2) ---
    feat += [float(frame.game_phase == 1)]                   # is_kickoff
    gk_slot = _goalie_slot(subject_team)
    feat += [float(owner_slot == gk_slot)]                   # goalie_has_ball

    assert len(feat) == CORE_FEATURE_DIM, (
        f"Core feature dim mismatch: got {len(feat)}, expected {CORE_FEATURE_DIM}"
    )
    return feat


def extract_labels(frame: GameStateFrame,
                   subject_port: int,
                   subject_team: int) -> List[float]:
    """Build the 11-float label vector for one frame (v6).

    Buttons (7): A, B, X, Y, lob_pass, chip_shot, R.
    Analog (4):  (raw_byte - 128) / 128.0, i.e. [-1, 1].
    Stick X and C-stick X are negated when mirrored (RIGHT_TEAM).

    v6 change: standalone L removed. Replaced by composite labels:
      lob_pass  = L AND A (lob pass)
      chip_shot = L AND B (chip shot / lob shot)
    """
    ctrl   = frame.controllers[subject_port]
    mirror = (subject_team == RIGHT_TEAM)
    btn    = ctrl.buttons

    has_a = bool(btn & PAD_A)
    has_b = bool(btn & PAD_B)
    has_l = bool(btn & PAD_L)

    sx = (ctrl.stick_x - 128) / 128.0
    cx = (ctrl.substick_x - 128) / 128.0
    if mirror:
        sx = -sx
        cx = -cx

    return [
        float(has_a),                    # A (pass / switch)
        float(has_b),                    # B (shoot / slide tackle)
        float(bool(btn & PAD_X)),        # X (powerup)
        float(bool(btn & PAD_Y)),        # Y (deke / hit)
        float(has_l and has_a),          # lob_pass (L+A)
        float(has_l and has_b),          # chip_shot (L+B)
        float(bool(btn & PAD_R)),        # R (turbo / sprint)
        sx,                              # stick_x
        (ctrl.stick_y    - 128) / 128.0, # stick_y
        cx,                              # cstick_x
        (ctrl.substick_y - 128) / 128.0, # cstick_y
    ]


# ---- Label names (for logging) ----------------------------------------------
# Index: 0=A  1=B  2=X  3=Y  4=lob_pass  5=chip_shot  6=R
#        7=stick_x  8=stick_y  9=cstick_x  10=cstick_y
# Strikers semantics:
#   A         — pass (offense) / switch controlled character (defense)
#   B         — shoot / charge shot (offense) / slide tackle (defense)
#   X         — powerup usage (offense or defense)
#   Y         — deke (offense, also available via C-stick) / hit attempt (defense)
#   lob_pass  — L+A: lob pass (composite label, v6)
#   chip_shot — L+B: chip shot / lob shot (composite label, v6)
#   R         — turbo / sprint modifier on stick (hold for speed boost)
#   Z         — switch active item (not in labels; rarely used)
LABEL_NAMES = [
    "A(pass/sw)", "B(shoot/sl)", "X(powerup)", "Y(deke/hit)",
    "lob_pass(L+A)", "chip_shot(L+B)", "R(turbo)",
    "stick_x", "stick_y", "cstick_x", "cstick_y",
]

# Expected press-rate ranges derived from v5 data.
# Used to flag anomalies during the build. 3× / ÷3 tolerance.
_BTN_EXPECTED = {
    0: (0.02, 0.20),   # A:         v5 ~7.6%
    1: (0.02, 0.20),   # B:         v5 ~7.9%
    2: (0.004, 0.04),  # X:         v5 ~0.8%
    3: (0.006, 0.06),  # Y:         v5 ~1.9%
    4: (0.001, 0.03),  # lob_pass:  subset of L (~1.3%) ∩ A
    5: (0.001, 0.03),  # chip_shot: subset of L (~1.3%) ∩ B
    6: (0.15, 0.75),   # R:         v5 ~54%
}

# ---- Checkpoint helpers -----------------------------------------------------

CHECKPOINT_INTERVAL = 100  # save partial results every N matched files


def _save_checkpoint(X_chunks: list, y_chunks: list, seg_chunks: list,
                     window_match_starts: list,
                     output_path: str, ckpt_num: int, ckpt_dir: str) -> str:
    """Write checkpoint incrementally via memmap — never concatenates full arrays."""
    os.makedirs(ckpt_dir, exist_ok=True)
    stem = os.path.basename(output_path)
    ckpt_base = os.path.join(ckpt_dir, f"{stem}.ckpt{ckpt_num}")
    n_rows   = sum(c.shape[0] for c in X_chunks)
    feat_dim = X_chunks[0].shape[1]
    lbl_dim  = y_chunks[0].shape[1]

    X_mm = np.lib.format.open_memmap(
        ckpt_base + "_X.npy", mode="w+", dtype=np.float16, shape=(n_rows, feat_dim))
    y_mm = np.lib.format.open_memmap(
        ckpt_base + "_y.npy", mode="w+", dtype=np.float32, shape=(n_rows, lbl_dim))
    s_mm = np.lib.format.open_memmap(
        ckpt_base + "_seg.npy", mode="w+", dtype=np.int32, shape=(n_rows,))

    off = 0
    for xc, yc, sc in zip(X_chunks, y_chunks, seg_chunks):
        n = len(xc)
        X_mm[off:off + n] = xc
        y_mm[off:off + n] = yc
        s_mm[off:off + n] = sc.astype(np.int32)
        off += n
    del X_mm, y_mm, s_mm  # flush to disk

    np.save(ckpt_base + "_ms.npy", np.array(window_match_starts, dtype=np.int64))
    return ckpt_base


def _save_partial_manifest(output_path: str, expert_ids: Set[int],
                           seen_game_keys: Set[str],
                           ckpt_paths: list, global_frame_idx: int,
                           seg_counter: int) -> None:
    """Write a partial manifest after each checkpoint for preemption recovery.

    On resume, build_dataset reads this file via --resume and restores
    ckpt_paths / global_frame_idx / seg_counter so the new run continues
    seamlessly without reprocessing already-captured games.
    """
    output_base = output_path[:-4] if output_path.endswith(".npz") else output_path
    manifest = {
        "partial": True,
        "expert_ids": sorted(expert_ids),
        "build_date": datetime.now(timezone.utc).isoformat(),
        "seen_game_keys": sorted(seen_game_keys),
        "checkpoints": ckpt_paths,
        "global_frame_idx": global_frame_idx,
        "seg_counter": seg_counter,
    }
    manifest_path = output_base + "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _merge_to_memmap(
    ckpt_paths: list, X_chunks: list, y_chunks: list, seg_chunks: list,
    remaining_ms: "np.ndarray", output_base: str,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Merge checkpoint .npy files + any trailing in-memory chunks into
    memory-mapped .npy files without loading all data into RAM at once."""
    ckpt_rows: List[int] = []
    all_ms:  List["np.ndarray"] = []
    all_seg: List["np.ndarray"] = []
    feature_dim: Optional[int] = None
    label_dim:   Optional[int] = None
    for p in ckpt_paths:
        X_ck = np.load(p + "_X.npy", mmap_mode="r")
        ckpt_rows.append(X_ck.shape[0])
        if feature_dim is None:
            feature_dim = X_ck.shape[1]
            y_ck = np.load(p + "_y.npy", mmap_mode="r")
            label_dim = y_ck.shape[1]
        all_ms.append(np.load(p + "_ms.npy"))
        all_seg.append(np.load(p + "_seg.npy"))
        del X_ck

    remaining_rows = sum(c.shape[0] for c in X_chunks) if X_chunks else 0
    total_rows     = sum(ckpt_rows) + remaining_rows

    # Fall back to in-memory dims if no checkpoints exist (small --limit run)
    if feature_dim is None and X_chunks:
        feature_dim = X_chunks[0].shape[1]
        label_dim   = y_chunks[0].shape[1]

    x_path   = output_base + "_X.npy"
    y_path   = output_base + "_y.npy"
    ms_path  = output_base + "_ms.npy"
    seg_path = output_base + "_seg.npy"

    print(f"  Allocating {total_rows * feature_dim * 4 / 1e9:.1f} GB → {x_path}",
          flush=True)
    X_mm = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float16, shape=(total_rows, feature_dim)
    )
    y_mm = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.float32, shape=(total_rows, label_dim)
    )

    offset = 0
    for i, p in enumerate(ckpt_paths):
        X_ck = np.load(p + "_X.npy", mmap_mode="r")
        y_ck = np.load(p + "_y.npy", mmap_mode="r")
        n = ckpt_rows[i]
        X_mm[offset:offset + n] = X_ck
        y_mm[offset:offset + n] = y_ck
        offset += n
        print(f"  Wrote checkpoint {i + 1}/{len(ckpt_paths)}: "
              f"{n:,} rows → offset {offset:,}", flush=True)
        del X_ck, y_ck

    # Trailing in-memory chunks — write one at a time, no full concatenate
    off2 = offset
    for xc, yc in zip(X_chunks, y_chunks):
        n = len(xc)
        X_mm[off2:off2 + n] = xc
        y_mm[off2:off2 + n] = yc
        off2 += n
    if remaining_rows:
        print(f"  Wrote trailing {remaining_rows:,} rows", flush=True)

    # Flush memory maps to disk
    del X_mm, y_mm

    all_ms.append(remaining_ms)
    ms_combined = np.concatenate(all_ms)
    np.save(ms_path, ms_combined)

    # Merge seg arrays and save
    if seg_chunks:
        all_seg.append(np.concatenate(seg_chunks).astype(np.int32))
    seg_combined = np.concatenate(all_seg).astype(np.int32) if all_seg else np.array([], dtype=np.int32)
    np.save(seg_path, seg_combined)

    # Re-open as read-only for stats printing (supports slicing same as ndarray)
    X_out = np.lib.format.open_memmap(x_path, mode="r")
    y_out = np.lib.format.open_memmap(y_path, mode="r")
    return X_out, y_out, ms_combined


# ---- Main -------------------------------------------------------------------

def build_dataset(citf_dir: str, expert_ids: Set[int],
                  output_path: str, print_stats: bool,
                  limit: Optional[int] = None,
                  resume_manifest: Optional[str] = None) -> None:
    import time
    citf_paths = sorted(Path(citf_dir).rglob("*.citframes"))
    if limit:
        citf_paths = citf_paths[:limit]
    total_files = len(citf_paths)
    print(f"Found {total_files} .citframes files", flush=True)
    print(f"Expert IDs: {sorted(expert_ids)}", flush=True)

    seen_game_keys: Set[str] = set()

    X_chunks:   List["np.ndarray"] = []
    y_chunks:   List["np.ndarray"] = []
    seg_chunks: List["np.ndarray"] = []
    ckpt_paths: List[str] = []
    global_frame_idx = 0
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_checkpoints")
    seg_counter      = 0

    # Pre-seed from a prior manifest so already-processed games are skipped.
    # Only new-format manifests (with seen_game_keys) are compatible.
    if resume_manifest:
        with open(resume_manifest) as f:
            manifest_data = json.load(f)
        if "seen_game_keys" in manifest_data:
            for key in manifest_data["seen_game_keys"]:
                seen_game_keys.add(key)
            print(f"Resumed from manifest: {len(seen_game_keys):,} games already seen "
                  f"({resume_manifest})", flush=True)
            if manifest_data.get("partial"):
                ckpt_paths       = manifest_data.get("checkpoints", [])
                global_frame_idx = manifest_data.get("global_frame_idx", 0)
                seg_counter      = manifest_data.get("seg_counter", 0)
                print(f"  Partial resume: {len(ckpt_paths)} checkpoint(s), "
                      f"global_frame_idx={global_frame_idx:,}, "
                      f"seg_counter={seg_counter:,}", flush=True)
        else:
            print(f"  WARN: manifest uses old (epoch, discord_id) format — "
                  f"cannot resume; starting fresh.", flush=True)

    # match_starts: absolute frame index where each expert-game entry begins.
    # One entry per (unique game × expert port) combination.
    all_match_starts: List[int] = []
    ms_window_start  = 0

    running_y_sum   = np.zeros(LABEL_DIM, dtype=np.float64)
    running_y_count = 0

    n_no_experts    = 0   # games where no allowlisted expert appears
    n_not_hvh       = 0
    n_dedup         = 0   # duplicate game submissions skipped
    n_filtered      = 0   # raw frames excluded by phase/pause filter (counted once per game)
    n_goalie_frames = 0   # kept frames where the expert controls the goalie
    n_matched       = 0   # unique games processed
    n_expert_entries = 0  # total (game × expert) entries written
    total_frames    = 0
    t_start         = time.time()

    for file_idx, path in enumerate(citf_paths):

        # ---- Progress report every 50 files --------------------------------
        if file_idx % 50 == 0 and file_idx > 0:
            elapsed = time.time() - t_start
            rate    = file_idx / elapsed
            eta     = (total_files - file_idx) / rate if rate > 0 else 0
            print(f"\n[{file_idx:4d}/{total_files}]  unique_games={n_matched}  "
                  f"expert_entries={n_expert_entries}  frames={total_frames:,}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)

            if running_y_count > 0:
                rates = running_y_sum / running_y_count
                parts = []
                flags = []
                for i in range(BUTTON_DIM):
                    r = rates[i]
                    parts.append(f"{LABEL_NAMES[i]}={r:.3f}")
                    lo, hi = _BTN_EXPECTED[i]
                    if not (lo <= r <= hi):
                        flags.append(f"  WARNING: {LABEL_NAMES[i]} rate {r:.3f} "
                                     f"outside expected [{lo:.3f}, {hi:.3f}]")
                print(f"  Rates: {' | '.join(parts)}", flush=True)
                for flag in flags:
                    print(flag, flush=True)

        # ---- Load and parse file -------------------------------------------
        try:
            data = load_citf_bytes(str(path))
        except Exception as e:
            print(f"  WARN: failed to load {path.name}: {e}", flush=True)
            continue

        header = parse_header(data)

        if header.version < 11:
            print(f"  SKIP {path.name}: version {header.version} < 11", flush=True)
            continue

        ok_hvh, hvh_reason = _is_human_vs_human(header)
        if not ok_hvh:
            n_not_hvh += 1
            continue

        # ---- Parse all frames (one pass — reused for dedup + per-expert extraction) ---
        frames: List[GameStateFrame] = []
        offset = header.header_size
        for _ in range(header.frame_count):
            frame, consumed = parse_frame(data, offset, header.fixed_frame_size)
            frames.append(frame)
            offset += consumed

        # ---- Game-level dedup ----------------------------------------------
        game_key = _compute_game_key(header, frames)
        if game_key in seen_game_keys:
            n_dedup += 1
            continue
        seen_game_keys.add(game_key)

        # ---- Require at least one allowlisted expert -----------------------
        expert_ports = _find_expert_ports(header, expert_ids)
        if not expert_ports:
            n_no_experts += 1
            continue

        n_matched += 1

        # Count filtered frames once per game (informational; not multiplied by experts)
        for frame in frames:
            if frame.game_phase not in (1, 4, 5) or frame.is_paused:
                n_filtered += 1

        # ---- Extract features for each expert found in this game -----------
        for subject_port, subject_discord_id in expert_ports:
            subject_team = _get_subject_team(header, subject_port)

            file_X:   List[List[float]] = []
            file_y:   List[List[float]] = []
            file_seg: List[int] = []

            prev_was_active   = False
            prev_phase_family = -1
            prev_labels       = [0.0] * LABEL_DIM

            for frame in frames:
                if frame.game_phase not in (1, 4, 5) or frame.is_paused:
                    prev_was_active = False
                    continue

                self_slot, is_goalie = _find_self_slot(frame, subject_team)
                if is_goalie:
                    n_goalie_frames += 1

                phase_family = 1 if frame.game_phase == 1 else 4

                if not prev_was_active or phase_family != prev_phase_family:
                    seg_counter += 1
                    prev_labels  = [0.0] * LABEL_DIM

                curr_labels = extract_labels(frame, subject_port, subject_team)
                feat = (extract_features(header, frame, subject_team, self_slot, is_goalie)
                        + prev_labels)
                file_X.append(feat)
                file_y.append(curr_labels)
                file_seg.append(seg_counter)

                prev_labels       = curr_labels
                prev_was_active   = True
                prev_phase_family = phase_family

            if file_X:
                arr_X   = np.array(file_X,   dtype=np.float32)
                arr_y   = np.array(file_y,   dtype=np.float32)
                arr_seg = np.array(file_seg, dtype=np.int32)
                all_match_starts.append(global_frame_idx)
                X_chunks.append(arr_X)
                y_chunks.append(arr_y)
                seg_chunks.append(arr_seg)
                total_frames     += len(file_X)
                global_frame_idx += len(file_X)
                running_y_sum    += arr_y.sum(axis=0)
                running_y_count  += len(arr_y)
                n_expert_entries += 1

        # ---- Checkpoint every CHECKPOINT_INTERVAL unique games -------------
        if n_matched > 0 and n_matched % CHECKPOINT_INTERVAL == 0:
            ckpt_num       = len(ckpt_paths) + 1
            window_ms      = all_match_starts[ms_window_start:]
            frames_in_ckpt = sum(len(c) for c in X_chunks)
            ckpt_path = _save_checkpoint(X_chunks, y_chunks, seg_chunks, window_ms,
                                         output_path, ckpt_num, ckpt_dir)
            ckpt_paths.append(ckpt_path)
            _save_partial_manifest(output_path, expert_ids, seen_game_keys,
                                   ckpt_paths, global_frame_idx, seg_counter)
            X_chunks        = []
            y_chunks        = []
            seg_chunks      = []
            ms_window_start = len(all_match_starts)
            print(f"\n  Checkpoint {ckpt_num} saved ({frames_in_ckpt:,} frames, "
                  f"{len(window_ms)} entries) → {Path(ckpt_path).name}", flush=True)

    # ---- Final summary ------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE", flush=True)
    print(f"  Unique games processed          : {n_matched}", flush=True)
    print(f"  Expert-game entries written     : {n_expert_entries}", flush=True)
    print(f"  Skipped — no expert present     : {n_no_experts}", flush=True)
    print(f"  Skipped — not HvH               : {n_not_hvh}", flush=True)
    print(f"  Skipped — duplicate game        : {n_dedup}", flush=True)
    print(f"  Frames kept (phase 1/4/5)       : {total_frames:,}", flush=True)
    print(f"  Frames filtered (non-active)    : {n_filtered:,}", flush=True)
    print(f"  Frames kept (goalie control)    : {n_goalie_frames:,}", flush=True)
    print(f"  Segments created                : {seg_counter:,}", flush=True)
    print(f"  Checkpoints written             : {len(ckpt_paths)}", flush=True)
    print(f"  Wall time                       : {elapsed:.1f}s  "
          f"({elapsed/60:.1f} min)", flush=True)
    print(f"{'='*60}", flush=True)

    if not X_chunks and not ckpt_paths:
        print("\nERROR: No frames collected. Check --experts IDs and citf_dir.",
              flush=True)
        sys.exit(1)

    print("\nMerging and saving final dataset...", flush=True)
    remaining_ms = np.array(all_match_starts[ms_window_start:], dtype=np.int64)

    output_base = output_path[:-4] if output_path.endswith(".npz") else output_path

    X, y, ms = _merge_to_memmap(ckpt_paths, X_chunks, y_chunks, seg_chunks,
                                 remaining_ms, output_base)

    for p in ckpt_paths:
        for suffix in ("_X.npy", "_y.npy", "_seg.npy", "_ms.npy"):
            try:
                Path(p + suffix).unlink()
            except Exception:
                pass

    x_path   = output_base + "_X.npy"
    y_path   = output_base + "_y.npy"
    ms_path  = output_base + "_ms.npy"
    seg_path = output_base + "_seg.npy"
    print(f"Saved: {X.shape[0]:,} frames × {X.shape[1]} features", flush=True)
    print(f"  X   → {x_path}", flush=True)
    print(f"  y   → {y_path}", flush=True)
    print(f"  ms  → {ms_path}", flush=True)
    print(f"  seg → {seg_path}  ({seg_counter:,} unique segments)", flush=True)
    print(f"  match_starts: {len(ms):,} entries  "
          f"(avg {X.shape[0]//max(len(ms),1):,} frames/entry)", flush=True)

    if print_stats:
        _print_stats(X, y)

    # ---- Write manifest -----------------------------------------------------
    manifest_path = output_base + "_manifest.json"
    manifest = {
        "expert_ids": sorted(expert_ids),
        "build_date": datetime.now(timezone.utc).isoformat(),
        "unique_games": n_matched,
        "expert_entries": n_expert_entries,
        "total_frames": int(X.shape[0]),
        "output_files": {
            "X":   x_path,
            "y":   y_path,
            "ms":  ms_path,
            "seg": seg_path,
        },
        "seen_game_keys": sorted(seen_game_keys),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest → {manifest_path}  ({len(seen_game_keys):,} game keys)", flush=True)


def _print_stats(X: "np.ndarray", y: "np.ndarray") -> None:
    print("\n--- Label statistics ---", flush=True)
    for i, name in enumerate(LABEL_NAMES):
        col = y[:, i]
        if i < BUTTON_DIM:
            print(f"  {name:<18s}  press_rate={col.mean():.4f}  "
                  f"({int(col.sum()):,} / {len(col):,} frames)", flush=True)
        else:
            print(f"  {name:<18s}  mean={col.mean():.3f}  std={col.std():.3f}  "
                  f"range=[{col.min():.3f}, {col.max():.3f}]", flush=True)

    print("\n--- Feature statistics (ball block: dims 0-10) ---", flush=True)
    ball_names = ["ball_pos_x", "ball_pos_y", "ball_pos_z",
                  "ball_vel_x", "ball_vel_y", "ball_vel_z",
                  "charge/35", "perfect_pass",
                  "b2g_dist/40", "b2g_dx_norm", "b2g_dy_norm"]
    for i, name in enumerate(ball_names):
        col = X[:, i]
        print(f"  [{i:3d}] {name:<14s}  mean={col.mean():.3f}  "
              f"std={col.std():.3f}  range=[{col.min():.3f}, {col.max():.3f}]",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build CITF → numpy ML training dataset (multi-expert, game-level dedup)"
    )
    parser.add_argument("citf_dir", help="Directory to search recursively for *.citframes files")
    parser.add_argument("output",   help="Output base path (e.g. sweetieman_v4)")

    expert_group = parser.add_mutually_exclusive_group(required=True)
    expert_group.add_argument(
        "--experts", metavar="IDS",
        help="Comma-separated allowlisted expert discord IDs  "
             "(e.g. 774682411052040202,9876543210)")
    expert_group.add_argument(
        "--experts-file", metavar="FILE",
        help="Text file with one discord ID per line")

    parser.add_argument("--stats",  action="store_true",
                        help="Print feature and label statistics after building")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Process only the first N files (for quick testing)")
    parser.add_argument("--resume", default=None, metavar="MANIFEST",
                        help="Path to existing _manifest.json; skip already-processed games")
    args = parser.parse_args()

    if args.experts:
        expert_ids = {int(x.strip()) for x in args.experts.split(",") if x.strip()}
    else:
        with open(args.experts_file) as f:
            expert_ids = {int(line.strip()) for line in f if line.strip()}

    if not expert_ids:
        parser.error("No expert IDs provided.")

    build_dataset(args.citf_dir, expert_ids, args.output, args.stats,
                  args.limit, args.resume)
