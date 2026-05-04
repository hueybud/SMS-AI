"""Reward function for PPO over Trajectory rollouts.

Two reward sources combine into a single per-frame reward for the agent's
perspective:

    - **Sparse — goal differential**.  ``±GOAL_REWARD`` per goal scored
      across the transition.  Zero-sum: agent's team scores → +X; their
      team scores → -X.
    - **Dense — possession turnover, zoned**.  When prev_owner is on our
      team and curr_owner is on theirs, apply a penalty.  Zone is decided
      by ball x at the moment of recovery (where the opponent picked it up,
      not where we lost it — clearing into their half is fine, getting
      tackled in our box is not).  Goalie pickup gets a flat low penalty
      regardless of zone (shooting is good even though most shots miss).

Reset boundaries are handled carefully: the **goal** signal is preserved
across a reset boundary (the gap between our last in-play frame and the
post-celebration kickoff is the goal celebration; score deltas there are
real), but the **possession** signal is skipped (kickoff resets the ball
to a goalie regardless of pre-celebration state, which would otherwise
look like a turnover).

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

# Sparse: per goal.  Symmetric — +X for us, -X for them.  Big to dominate
# the dense shaping signal in the long run; PPO normalizes advantages so
# absolute scale only matters relative to the dense terms below.
GOAL_REWARD = 10.0

# Dense: possession turnover penalties.  Applied only on a true loss
# (prev_owner on our team, curr_owner on theirs).  No counter-bonus for
# us recovering — keep the signal one-sided to bias toward keeping the ball;
# add the symmetric reward later if play feels too passive in defense.
LOSS_TO_OPPONENT_DEFENSIVE_THIRD = -2.0   # they recover near our goal — bad
LOSS_TO_OPPONENT_MIDDLE_THIRD    = -1.0
LOSS_TO_OPPONENT_ATTACKING_THIRD = -0.3   # we cleared deep, they recovered far — fine
LOSS_TO_OPPONENT_GOALIE          = -0.3   # goalie pickup regardless of zone

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

        # ── Possession turnover (dense) ─────────────────────────────────────
        # Skip across reset boundaries: at a kickoff the ball goes to a
        # goalie regardless of pre-celebration state, which would falsely
        # look like a turnover.
        if is_reset:
            continue

        prev_owner = _owner_slot(s_t)
        curr_owner = _owner_slot(s_t1)
        if prev_owner in ours and curr_owner in theirs:
            if curr_owner == opp_goalie:
                rewards[t] += LOSS_TO_OPPONENT_GOALIE
            else:
                ball_x = float(s_t1.core_features[0])  # mirrored: +X = their goal
                rewards[t] += _zone_penalty_for(ball_x)

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
