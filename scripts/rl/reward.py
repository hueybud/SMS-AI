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
      small term either way.  Detection tracks the LAST team to own the
      ball, not just the previous frame's owner — clears (us → loose →
      them) and hit-recoveries (them → loose → us) both fire correctly
      even though the ball passes through a loose-ball intermediary.
    - **Dense — possession progress**.  Per-frame bonus proportional to
      forward ball motion (+X = toward opponent goal) while one of our
      strikers has the ball.  Goalie possession is excluded — the
      goalie isn't really playing.  Replaced an earlier static
      possession bonus that the AI exploited by running into a wall to
      pin the ball and farm the constant signal.
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

from typing import Callable, Optional

import numpy as np

from .protocol import StateFrame
from .trajectory import Trajectory


# Optional sink for the live reward-overlay window.  Each non-zero reward
# event is reported as ``sink(component_name, value)``; the aggregate per
# frame is still summed into ``rewards[t]`` exactly as before.  Pass
# ``None`` (default) to skip — zero overhead when unused.
EventSink = Optional[Callable[[str, float], None]]


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

# Dense: possession progress.  Rewards forward ball motion (+X = toward
# opponent goal) while one of our STRIKERS has the ball — explicitly
# not when our goalie has it (the goalie isn't really playing the
# game, just holding the ball).  Replaced an earlier static per-frame
# possession bonus that was reward-hackable: the AI learned to run
# into a wall holding the ball and farm the constant bonus risk-free.
# Progress-based shaping makes wall-hugging earn 0 while dribbling
# forward earns the credit.  max(0, Δ) — backward passes are sometimes
# tactically correct (reset, switch field), so they earn zero rather
# than negative.
#
# Magnitude: ~0.001 per ball-x unit advanced.  Walking forward 2s
# (~18 units) ≈ +0.018; full-field clear (76 units) ≈ +0.08.
POSSESSION_PROGRESS_PER_UNIT = +0.001

# Dense: stagnation penalty.  Forward-progress alone doesn't punish
# "carry the ball backward into the wall" — backward motion just earns
# zero, which is competitive with most legitimate play in a sparse-
# reward setting.  This adds an explicit per-frame cost once a
# possession streak goes too long without setting a new ball-x maximum.
#
# Mechanism: while one of our STRIKERS holds (goalie excluded), track
# the highest ball x reached during the current possession streak.  Each
# frame the streak fails to advance that maximum, increment a counter.
# Past GRACE_FRAMES, every frame adds STAGNATION_PENALTY_PER_FRAME.  A
# new high-water mark resets the counter to 0; a possession change (to
# opponent, loose, or our goalie) or kickoff reset clears the streak
# entirely.
#
# Why grace-then-linear instead of asymmetric per-frame progress: a
# straight per-frame backward penalty would punish a 30-frame backpass
# the same as a 5-second wall-camp.  Goal here is "brief retreat is
# fine, sustained is bad," which is exactly what a grace window
# expresses.
#
# Tuning: 120 frames (~2s @ 60fps) covers a backpass + beat to receive.
# At -0.0003/frame, 60 frames past grace ≈ -0.018, the same magnitude
# as the forward-progress reward you'd earn walking forward those 2s.
# So once stagnation engages, the policy can't break even by
# alternating backward-stall and forward-walk.
STAGNATION_GRACE_FRAMES      = 120
STAGNATION_PENALTY_PER_FRAME = -0.0003
# Minimum ball-x advance required to reset the stagnation counter.
# Without this, side-wall physics jitter (~0.01–0.1 units per frame)
# constantly resets stagnation_frames to zero before the grace period
# expires, so wall-camping never triggers the penalty.  Set to 0.5 units
# — unmistakable forward progress, invisible to normal dribbling.
STAGNATION_MIN_ADVANCE = 0.5

# Dense: shot attempt — fires once when one of our strikers enters a
# shot action state from a non-shot state, AND that specific striker
# is on the offensive side of the field (their mirrored x > 0).
# State-driven (not velocity heuristic): the carrier's
# eFielderActionState transitions into the shot animation when the
# player commits to a shot.
#
# The field-side gate matters because these same five animation states
# are ALSO used when a striker clears the ball away from their own
# goal.  Without the gate, the AI gets rewarded for clearing — which
# we explicitly don't want, since BC already over-clears and we're
# trying to push the policy toward keeping the ball, not booting it.
# We gate on the striker's position, not the ball's: nearly always the
# same thing since the carrier holds the ball, but the striker's
# location is what semantically distinguishes a shot from a clear.
SHOT_REWARD = +0.05
# Mirror penalty for the opponent entering a shot state on their
# offensive side (the agent's defensive side, mirrored x < 0).  Without
# this the reward is asymmetric — we get +SHOT for trying, they get a
# free pass for trying — and PPO's optimal play is to camp in their
# half and dare them to shoot from distance.  Same-magnitude mirror keeps
# the shaping zero-sum.  Also same five animation states; gating on
# their striker's mirrored x < 0 separates a real shot at our goal from
# them clearing the ball out of their own half.
THEIR_SHOT_REWARD = -SHOT_REWARD
SHOT_STATES = frozenset({0x05, 0x07, 0x08, 0x11, 0x12})

# Possession-progress is per-frame at ~0.001/unit of forward ball
# motion — individual frames are mostly sub-millireward and would flood
# the overlay with noise.  The compute() loop accumulates progress into
# a bucket and only emits an event when the bucket crosses this
# threshold, so the overlay shows POSS_PROGRESS events at roughly
# GAIN_GOALIE magnitude (~0.01).  The reward array itself is unchanged
# — every per-frame contribution still lands in ``rewards[t]``.
POSS_PROGRESS_EMIT_THRESHOLD = 0.005

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

# Same striker blocks, but the FIRST float of each — Δpos_x relative
# to the ball.  Striker's mirrored x = ball_x + Δx, so we can
# reconstruct each striker's absolute x position from features[0] +
# features[Δx_offset] without needing extra protocol fields.  Used to
# gate the shot reward per-striker so we only credit shots taken from
# the offensive side of the field, not clears.
_OUR_STRIKER_DX_OFFSETS = (22, 41, 48, 55)

# Opponent striker blocks, mirroring the friendly section but for
# AIController::ReadGameStateCore section 6 (4 enemy strikers × 7
# floats each, layout = Δpos(3) + state(1) + heading(2) + is_carrier(1)).
# Block starts at index 66; each block is 7 floats wide.  state_idx is
# the 4th float (offset +3); Δpos_x is the 1st (offset +0).  Used to
# detect opponent shot attempts so SHOT_REWARD is symmetric.
_THEIR_STRIKER_STATE_OFFSETS = (69, 76, 83, 90)
_THEIR_STRIKER_DX_OFFSETS    = (66, 73, 80, 87)


def _anyone_shooting(states_tuple: tuple, xs_tuple: tuple, side_gate) -> bool:
    """True if any striker in the given block is in a shot state on the
    correct side of the field.  Sort-order invariant: we check PRESENCE of
    a shot-state striker, not per-slot transitions."""
    return any(
        cs in SHOT_STATES and side_gate(sx)
        for cs, sx in zip(states_tuple, xs_tuple)
    )


def _our_striker_states(state: StateFrame) -> tuple:
    """Return (s0, s1, s2, s3) — the action state byte for each of our
    four strikers, decoded from the float feature indices.  Empty/unknown
    blocks register as 0 (non-shot state) and are harmless.
    """
    f = state.core_features
    return tuple(int(round(float(f[i]))) for i in _OUR_STRIKER_STATE_OFFSETS)


def _our_striker_xs(state: StateFrame) -> tuple:
    """Return (x0, x1, x2, x3) — mirrored x position for each of our four
    strikers, computed from the ball-relative Δpos features (striker_x =
    ball_x + Δx).  Empty striker blocks are zeroed in C++, so they
    register as ``ball_x + 0 = ball_x``; the shot reward's state gate
    filters those out anyway (state == 0 isn't a shot state).
    """
    f = state.core_features
    bpx = float(f[0])
    return tuple(bpx + float(f[i]) for i in _OUR_STRIKER_DX_OFFSETS)


def _their_striker_states(state: StateFrame) -> tuple:
    """Same as ``_our_striker_states`` but for the four opponent strikers."""
    f = state.core_features
    return tuple(int(round(float(f[i]))) for i in _THEIR_STRIKER_STATE_OFFSETS)


def _their_striker_xs(state: StateFrame) -> tuple:
    """Same as ``_our_striker_xs`` but for the four opponent strikers."""
    f = state.core_features
    bpx = float(f[0])
    return tuple(bpx + float(f[i]) for i in _THEIR_STRIKER_DX_OFFSETS)


class RewardComputer:
    """Stateful per-frame reward computer.

    Carries the cross-frame tracking state (last ball owner, stagnation
    high-water mark, possession-progress bucket) so rewards can be
    computed one frame at a time — enabling real-time event emission to
    the overlay instead of batch emission at rollout end.

    Typical usage in the collection loop::

        rc = RewardComputer(mirror_x, event_sink=reward_event_sink)
        for each frame:
            s_t = state
            state = env.step(...)
            rollout_rewards.append(rc.step(s_t, state, state.reset_context))
        rc.flush()                          # drain poss-progress bucket
        traj.rewards = np.array(rollout_rewards, dtype=np.float32)

    ``compute()`` below is a thin wrapper around this class for batch use
    (replay, unit tests, etc.).
    """

    def __init__(self, mirror_x: bool, event_sink: EventSink = None) -> None:
        if mirror_x:
            self._ours       = RIGHT_TEAM_SLOTS
            self._theirs     = LEFT_TEAM_SLOTS
            self._opp_goalie = LEFT_GOALIE_SLOT
            self._our_goalie = RIGHT_GOALIE_SLOT
        else:
            self._ours       = LEFT_TEAM_SLOTS
            self._theirs     = RIGHT_TEAM_SLOTS
            self._opp_goalie = RIGHT_GOALIE_SLOT
            self._our_goalie = LEFT_GOALIE_SLOT

        self._mirror_x = mirror_x
        self._sink = event_sink

        self._last_owner_team: Optional[str] = None
        self._initialized = False          # seed last_owner_team on first step
        self._best_x_streak: Optional[float] = None
        self._stagnation_frames: int = 0
        self._poss_progress_bucket: float = 0.0

    def _emit(self, name: str, value: float) -> None:
        if self._sink is not None and value != 0.0:
            self._sink(name, float(value))

    def step(self, s_t: StateFrame, s_t1: StateFrame, is_resetting: bool) -> float:
        """Compute reward for one transition (s_t → s_t1). Returns scalar."""
        reward = 0.0

        # Seed last_owner_team from the first s_t we see.
        if not self._initialized:
            owner0 = _owner_slot(s_t)
            if owner0 in self._ours:
                self._last_owner_team = "ours"
            elif owner0 in self._theirs:
                self._last_owner_team = "theirs"
            self._initialized = True

        # ── Goal events (sparse) ────────────────────────────────────────────
        if self._mirror_x:
            our_delta   = int(s_t1.score_right) - int(s_t.score_right)
            their_delta = int(s_t1.score_left)  - int(s_t.score_left)
        else:
            our_delta   = int(s_t1.score_left)  - int(s_t.score_left)
            their_delta = int(s_t1.score_right) - int(s_t.score_right)
        reward += GOAL_REWARD * float(our_delta - their_delta)
        if our_delta:
            self._emit("GOAL_FOR",     GOAL_REWARD * our_delta)
        if their_delta:
            self._emit("GOAL_AGAINST", -GOAL_REWARD * their_delta)

        # ── Reset boundary gate ─────────────────────────────────────────────
        if is_resetting:
            self._last_owner_team = None
            self._best_x_streak = None
            self._stagnation_frames = 0
            return reward

        curr_owner = _owner_slot(s_t1)

        # ── Possession progress + stagnation (dense) ────────────────────────
        if curr_owner in self._ours and curr_owner != self._our_goalie:
            bpx_t  = float(s_t.core_features[0])
            bpx_t1 = float(s_t1.core_features[0])
            progress = POSSESSION_PROGRESS_PER_UNIT * max(0.0, bpx_t1 - bpx_t)
            reward += progress
            self._poss_progress_bucket += progress
            if self._poss_progress_bucket >= POSS_PROGRESS_EMIT_THRESHOLD:
                self._emit("POSS_PROGRESS", self._poss_progress_bucket)
                self._poss_progress_bucket = 0.0

            if self._best_x_streak is None or bpx_t1 > self._best_x_streak + STAGNATION_MIN_ADVANCE:
                self._best_x_streak = bpx_t1
                self._stagnation_frames = 0
            else:
                self._stagnation_frames += 1
                if self._stagnation_frames > STAGNATION_GRACE_FRAMES:
                    reward += STAGNATION_PENALTY_PER_FRAME
                    self._emit("STAGNATION", STAGNATION_PENALTY_PER_FRAME)
        else:
            self._best_x_streak = None
            self._stagnation_frames = 0

        # ── Possession turnover events (loose-ball-aware) ───────────────────
        if curr_owner in self._ours and self._last_owner_team == "theirs":
            if curr_owner == self._our_goalie:
                reward += GAIN_FROM_OPPONENT_GOALIE
                self._emit("GAIN_GOALIE", GAIN_FROM_OPPONENT_GOALIE)
            else:
                ball_x = float(s_t1.core_features[0])
                gain = _zone_gain_for(ball_x)
                reward += gain
                if ball_x < DEF_THIRD_BOUNDARY:
                    self._emit("GAIN_DEF_THIRD", gain)
                elif ball_x < ATK_THIRD_BOUNDARY:
                    self._emit("GAIN_MID_THIRD", gain)
                else:
                    self._emit("GAIN_ATK_THIRD", gain)
        elif curr_owner in self._theirs and self._last_owner_team == "ours":
            if curr_owner == self._opp_goalie:
                reward += LOSS_TO_OPPONENT_GOALIE
                self._emit("LOSS_GOALIE", LOSS_TO_OPPONENT_GOALIE)
            else:
                ball_x = float(s_t1.core_features[0])
                loss = _zone_penalty_for(ball_x)
                reward += loss
                if ball_x < DEF_THIRD_BOUNDARY:
                    self._emit("LOSS_DEF_THIRD", loss)
                elif ball_x < ATK_THIRD_BOUNDARY:
                    self._emit("LOSS_MID_THIRD", loss)
                else:
                    self._emit("LOSS_ATK_THIRD", loss)

        if curr_owner in self._ours:
            self._last_owner_team = "ours"
        elif curr_owner in self._theirs:
            self._last_owner_team = "theirs"

        # ── Shot attempt (dense, state-driven, side-gated) ──────────────────
        our_prev = _anyone_shooting(_our_striker_states(s_t),
                                    _our_striker_xs(s_t),
                                    lambda x: x > 0.0)
        our_curr = _anyone_shooting(_our_striker_states(s_t1),
                                    _our_striker_xs(s_t1),
                                    lambda x: x > 0.0)
        if our_curr and not our_prev:
            reward += SHOT_REWARD
            self._emit("SHOT", SHOT_REWARD)

        their_prev = _anyone_shooting(_their_striker_states(s_t),
                                      _their_striker_xs(s_t),
                                      lambda x: x < 0.0)
        their_curr = _anyone_shooting(_their_striker_states(s_t1),
                                      _their_striker_xs(s_t1),
                                      lambda x: x < 0.0)
        if their_curr and not their_prev:
            reward += THEIR_SHOT_REWARD
            self._emit("THEIR_SHOT", THEIR_SHOT_REWARD)

        return reward

    def flush(self) -> None:
        """Drain the possession-progress bucket at rollout end."""
        if self._poss_progress_bucket > 0.0:
            self._emit("POSS_PROGRESS", self._poss_progress_bucket)
            self._poss_progress_bucket = 0.0


def compute(traj: Trajectory, event_sink: EventSink = None) -> np.ndarray:
    """Per-frame reward array of shape ``(T,)`` — thin wrapper around
    ``RewardComputer`` for batch/replay use.  The collection loop uses
    ``RewardComputer.step()`` directly for real-time overlay emission."""
    rc = RewardComputer(traj.mirror_x, event_sink)
    rewards = np.zeros(traj.length, dtype=np.float32)
    for t in range(traj.length):
        rewards[t] = rc.step(
            traj.states[t], traj.states[t + 1], bool(traj.is_resetting[t + 1])
        )
    rc.flush()
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
