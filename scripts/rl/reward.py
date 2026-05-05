"""Reward function for PPO over Trajectory rollouts.

Four reward sources combine into a single per-frame reward for the
agent's perspective:

    - **Sparse — goal differential**.  ``±GOAL_REWARD`` per goal scored
      across the transition.  Zero-sum: agent's team scores → +X; their
      team scores → -X.
    - **Dense — possession turnover, zoned, symmetric**.  Zone is decided
      by ball x at the moment of recovery (where the new owner picked it
      up).  Losses (we → them) are penalized; gains (them → us) are
      rewarded.  Magnitudes are intentionally asymmetric (losses larger
      than gains) so the policy stays cautious, but both signs exist so
      PPO has a positive direction to climb.  Goalie pickups get a flat
      small term either way.
    - **Dense — possession holding**.  Tiny per-frame bonus when our
      team holds the ball.  Gives PPO a non-zero gradient on quiet
      rollouts where no events fire.
    - **Dense — shot attempts**.  Fires when one of our strikers enters
      a shot action state (eFielderActionState ∈ {0x05, 0x07, 0x08,
      0x11, 0x12}).  Counter-balances the goalie-pickup penalty so the
      AI learns to shoot, not to avoid shooting.  Detection is purely
      state-driven, not velocity-heuristic.

Reset boundaries are handled carefully: the **goal** signal is preserved
across a reset boundary (the gap between our last in-play frame and the
post-celebration kickoff is the goal celebration; score deltas there are
real), but the **possession** / **shot** signals are skipped (kickoff
resets the ball to a goalie and re-poses everyone, which would otherwise
look like spurious turnovers + state transitions).

All numbers are file-top constants — iterate by editing here, no code dive.

Vs-CPU MVP only computes the agent's reward (the opponent CPU isn't
learning).  For self-play later the same function applied to the
opponent's perspective gives a zero-sum reward by symmetry.
"""

from __future__ import annotations

import numpy as np

from .protocol import StateFrame
from .trajectory import Trajectory


# ── Tunable reward weights ───────────────────────────────────────────────────

# All magnitudes are scaled around a goal terminal of ±1.0 (slippi-ai
# style) so the numbers read cleanly as fractions of "scored a goal".
# Absolute scale doesn't affect PPO learning — advantages are normalized
# in the learner — but the smaller numbers are easier to eyeball in logs.

# Sparse: per goal.  Symmetric — +X for us, -X for them.  Anchored at
# 1.0 so every other term reads as a fraction of "scored a goal".
GOAL_REWARD = 1.0

# Dense: possession turnover penalties.  Applied on a true loss
# (prev_owner on our team, curr_owner on theirs).  Zoned by ball x at
# recovery.
LOSS_TO_OPPONENT_DEFENSIVE_THIRD = -0.20   # they recover near our goal — bad
LOSS_TO_OPPONENT_MIDDLE_THIRD    = -0.10
LOSS_TO_OPPONENT_ATTACKING_THIRD = -0.03   # we cleared deep, they recovered far — fine
# The goalie-pickup case overlaps with shot attempts (most shots end in
# goalie pickup).  With SHOT_REWARD now firing on shot state entry, a
# successful shot net-rewards the AI; this small term still penalizes
# weak passes that wander into the goalie's hands without any shot intent.
LOSS_TO_OPPONENT_GOALIE          = -0.01

# Dense: possession turnover GAINS — mirror of the loss terms.  Without
# these, every reward in the function is ≤ 0 and PPO's optimal policy is
# "minimize event count" → don't engage, don't shoot, sit still.  Magnitudes
# are smaller than the matching losses so the policy stays defensively
# biased, but the positive signal lets it actually learn what to do.
GAIN_FROM_OPPONENT_DEFENSIVE_THIRD = +0.03   # we recovered deep in our half — relief
GAIN_FROM_OPPONENT_MIDDLE_THIRD    = +0.07
GAIN_FROM_OPPONENT_ATTACKING_THIRD = +0.15   # we tackled them in their box — great
GAIN_FROM_OPPONENT_GOALIE          = +0.01   # opponent goalie released, we picked up

# Dense: per-frame possession holding bonus.  Tiny on purpose — at 240
# frames/rollout, holding the ball the entire rollout is +0.06 (well
# below a single goal terminal at ±1.0).  The point is to give PPO a
# non-zero gradient on every frame so quiet rollouts aren't all-zero
# advantages.
POSSESSION_BONUS_PER_FRAME = +0.00025

# Dense: shot attempt — fires once when one of our strikers enters a
# shot action state from a non-shot state.  State-driven (not velocity
# heuristic): the carrier's eFielderActionState transitions into the
# shot animation when the player commits to a shot.
SHOT_REWARD = +0.05
SHOT_STATES = frozenset({0x05, 0x07, 0x08, 0x11, 0x12})

# Field geometry — agent's mirrored frame so +X = their goal, -X = our goal.
# goal_x ≈ 38 in feature units for standard Strikers stadiums; the field
# is ±goal_x.  Thirds split at ±goal_x/3.
GOAL_X = 38.0
DEF_THIRD_BOUNDARY = -GOAL_X / 3.0
ATK_THIRD_BOUNDARY = +GOAL_X / 3.0


# ── Slot conventions (must mirror AIController::ReadGameStateCore) ───────────

LEFT_TEAM_SLOTS   = frozenset({0, 1, 2, 3, 4})    # strikers 0-3 + goalie 4
RIGHT_TEAM_SLOTS  = frozenset({5, 6, 7, 8, 9})    # strikers 5-8 + goalie 9
LEFT_GOALIE_SLOT  = 4
RIGHT_GOALIE_SLOT = 9
NO_OWNER_SLOT     = 10

# Ball-owner one-hot location in core_features (layout from
# AIController::ReadGameStateCore):
#   features[ 0..10] = ball pos/vel/charge/perfect_pass/ball_to_goal (11)
#   features[11..21] = ball owner one-hot, slots 0..10 (11)
_OWNER_OH_OFFSET = 11
_OWNER_OH_LEN    = 11


def _owner_slot(state: StateFrame) -> int:
    """Decode ball owner (0-10) from the one-hot at features[11..21]."""
    oh = state.core_features[_OWNER_OH_OFFSET : _OWNER_OH_OFFSET + _OWNER_OH_LEN]
    return int(np.argmax(oh))


def _zone_penalty_for(ball_x: float) -> float:
    """Zoned penalty for an opponent striker recovery at ``ball_x``."""
    if ball_x < DEF_THIRD_BOUNDARY:
        return LOSS_TO_OPPONENT_DEFENSIVE_THIRD
    if ball_x < ATK_THIRD_BOUNDARY:
        return LOSS_TO_OPPONENT_MIDDLE_THIRD
    return LOSS_TO_OPPONENT_ATTACKING_THIRD


def _zone_gain_for(ball_x: float) -> float:
    """Zoned reward for our striker recovery at ``ball_x`` — mirror of
    ``_zone_penalty_for``."""
    if ball_x < DEF_THIRD_BOUNDARY:
        return GAIN_FROM_OPPONENT_DEFENSIVE_THIRD
    if ball_x < ATK_THIRD_BOUNDARY:
        return GAIN_FROM_OPPONENT_MIDDLE_THIRD
    return GAIN_FROM_OPPONENT_ATTACKING_THIRD


# Feature offsets for the four friendly-striker state indices.  These
# mirror AIController::ReadGameStateCore exactly:
#   - Self block (19 floats) starts at index 22, state_idx at 25
#   - Friendly1 (7 floats) at 41-47, state_idx at 44
#   - Friendly2 (7 floats) at 48-54, state_idx at 51
#   - Friendly3 (7 floats) at 55-61, state_idx at 58
# StrikerStateIdx in AIController.cpp returns the float index into
# STRIKER_VOCAB, which is identity for 0x00..0x1B.  So feature value
# 5.0 == state 0x05 etc., and we can compare against SHOT_STATES (the
# raw byte set) directly after rounding.
_OUR_STRIKER_STATE_OFFSETS = (25, 44, 51, 58)


def _our_striker_states(state: StateFrame) -> tuple:
    """Return (s0, s1, s2, s3) — the action state byte for each of our
    four strikers, decoded from the float feature indices.  Empty/unknown
    blocks register as 0 (non-shot state) and are harmless.
    """
    f = state.core_features
    return tuple(int(round(float(f[i]))) for i in _OUR_STRIKER_STATE_OFFSETS)


def compute(traj: Trajectory) -> np.ndarray:
    """Per-frame reward array of shape ``(T,)`` aligned with ``traj.actions``.

    ``rewards[t]`` rewards the transition ``(s_t, a_t, s_{t+1})``.
    """
    T = traj.length

    if traj.mirror_x:
        ours        = RIGHT_TEAM_SLOTS
        theirs      = LEFT_TEAM_SLOTS
        opp_goalie  = LEFT_GOALIE_SLOT
    else:
        ours        = LEFT_TEAM_SLOTS
        theirs      = RIGHT_TEAM_SLOTS
        opp_goalie  = RIGHT_GOALIE_SLOT

    rewards = np.zeros(T, dtype=np.float32)
    our_goalie = LEFT_GOALIE_SLOT if not traj.mirror_x else RIGHT_GOALIE_SLOT

    for t in range(T):
        s_t  = traj.states[t]
        s_t1 = traj.states[t + 1]
        is_reset = bool(traj.is_resetting[t + 1])

        # ── Goal events (sparse) ────────────────────────────────────────────
        # Always credited.  Score changes that happen during a goal-
        # celebration gap (no STATE packets, then a reset frame on the
        # next kickoff) need to land on the action that led to the goal.
        if traj.mirror_x:
            our_delta   = int(s_t1.score_right) - int(s_t.score_right)
            their_delta = int(s_t1.score_left)  - int(s_t.score_left)
        else:
            our_delta   = int(s_t1.score_left)  - int(s_t.score_left)
            their_delta = int(s_t1.score_right) - int(s_t.score_right)
        rewards[t] += GOAL_REWARD * float(our_delta - their_delta)

        # ── Reset boundary gate ─────────────────────────────────────────────
        # Skip dense shaping across kickoffs: ball is repositioned,
        # everyone re-poses, action states reset.  Anything we'd compute
        # here (turnovers, shot transitions, possession) would be
        # spurious.  The goal differential above already fired for any
        # real score change in the celebration gap.
        if is_reset:
            continue

        prev_owner = _owner_slot(s_t)
        curr_owner = _owner_slot(s_t1)

        # ── Possession holding (dense, every frame) ─────────────────────────
        # Tiny bonus while our team has the ball.  Goalie counts — them
        # holding the ball is still our possession.  Without this the
        # AI gets reward=0 on most frames and learns nothing from quiet
        # play; with it, every frame contributes a small gradient.
        if curr_owner in ours:
            rewards[t] += POSSESSION_BONUS_PER_FRAME

        # ── Possession turnover events ──────────────────────────────────────
        # Loss: ours → theirs.
        if prev_owner in ours and curr_owner in theirs:
            if curr_owner == opp_goalie:
                rewards[t] += LOSS_TO_OPPONENT_GOALIE
            else:
                ball_x = float(s_t1.core_features[0])  # mirrored: +X = their goal
                rewards[t] += _zone_penalty_for(ball_x)
        # Gain: theirs → ours.  Mirror of the loss case.
        elif prev_owner in theirs and curr_owner in ours:
            if curr_owner == our_goalie:
                rewards[t] += GAIN_FROM_OPPONENT_GOALIE
            else:
                ball_x = float(s_t1.core_features[0])
                rewards[t] += _zone_gain_for(ball_x)

        # ── Shot attempt (dense, state-driven) ──────────────────────────────
        # Fires once per shot when any of our strikers enters a shot
        # action state from a non-shot state.  Counter-balances
        # LOSS_TO_OPPONENT_GOALIE so a goalie-saved shot still nets
        # positive (we tried).  We don't restrict to the carrier's
        # block — only the carrier ever transitions into a shot state
        # in normal play, so checking all four blocks is robust to the
        # exact frame where ball_owner_ptr clears.
        prev_states = _our_striker_states(s_t)
        curr_states = _our_striker_states(s_t1)
        for ps, cs in zip(prev_states, curr_states):
            if cs in SHOT_STATES and ps not in SHOT_STATES:
                rewards[t] += SHOT_REWARD
                break

    return rewards


def stats(rewards: np.ndarray) -> dict:
    """Compact summary stats for per-rollout logging.

    Returns a dict suitable for ``str.format`` or wandb.  Captures both
    the aggregate (sum, mean) and the event-level distribution (count,
    biggest single hit) so we can tell ``low total because few events``
    apart from ``low total because shaping is too weak``.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    n_events = int(np.count_nonzero(rewards))
    nz = rewards[rewards != 0]
    return {
        "sum":      float(rewards.sum()),
        "mean":     float(rewards.mean()) if rewards.size else 0.0,
        "n_events": n_events,
        "max_abs":  float(np.max(np.abs(nz))) if nz.size else 0.0,
        "n_pos":    int((rewards > 0).sum()),
        "n_neg":    int((rewards < 0).sum()),
    }
