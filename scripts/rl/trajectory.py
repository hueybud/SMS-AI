"""Rollout trajectory for PPO.

A ``Trajectory`` captures one slice of agent-environment interaction:
``T`` actions taken from ``T+1`` observed states, plus the policy log-probs
and value estimates *at sample time* (frozen π_old / V_old that PPO needs
for importance sampling and value regression targets).

Lifecycle:
    1. ``TrajectoryBuffer.push_state(state)``      — observation
    2. ``TrajectoryBuffer.push_action(...)``       — what the agent did from it
    3. (repeat steps 1-2)
    4. ``TrajectoryBuffer.push_state(final_state)``  — observation T+1
    5. ``traj = TrajectoryBuffer.finalize()``      — produces ``Trajectory``
    6. ``traj.rewards = reward.compute(traj, ...)``
    7. ``compute_gae(traj, last_value=...)``       — fills advantages + returns
    8. ``learner.update(traj)``

Conventions:
    - ``states`` length ``T+1``, ``actions`` length ``T`` — standard.
    - Every per-state field is ``T+1`` long; every per-action field is ``T``.
    - ``is_resetting[t]=True`` means ``state[t]`` was the first frame of a
      new game / match / kickoff segment (matches the C++ ``reset_context``
      flag, which mirrors slippi-ai's ``is_resetting``). GAE never
      bootstraps across a reset boundary.

Numpy throughout; the learner converts to torch on its way to the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .protocol import StateFrame


@dataclass
class Trajectory:
    """One PPO rollout slice.  Shape contract::

        T = length            (number of action steps)
        states           : list[StateFrame] of length T+1
        is_resetting     : np.ndarray (T+1,)   bool
        action_idx       : np.ndarray (T,)     int   — sampled action vocab idx
        stick_bin_idx    : np.ndarray (T, 4)   int   — sampled stick-bin per axis
        btn_flags        : np.ndarray (T, 7)   f32   — what was sent to C++
        stick_vals       : np.ndarray (T, 4)   f32   — what was sent to C++ ([-1,1])
        log_probs        : np.ndarray (T,)     f32   — log π_old of the joint sample
        values           : np.ndarray (T,)     f32   — V_old at each state[0..T-1]
        rewards          : np.ndarray (T,) or None    — set by reward.compute()
        advantages       : np.ndarray (T,) or None    — set by compute_gae()
        returns          : np.ndarray (T,) or None    — set by compute_gae()
        initial_kv       : np.ndarray (3,2,1,127,512) — KV cache at rollout-start
        initial_prev_labels : np.ndarray (11,) f32    — prev_labels at rollout-start

    The "ones we'd be sending to C++" duplication (action_idx + stick_bin_idx
    AND btn_flags + stick_vals) is small and worth it for debugging — when
    something goes wrong you can sanity-check the env replay vs the policy
    sample without rederiving one from the other.

    ``initial_kv`` and ``initial_prev_labels`` exist so the PPO update step
    can replay the rollout through the (now-modified) policy from the same
    starting state.  Without them the temporal model would see different
    context during replay than during sampling, breaking the importance-
    sampling assumption PPO depends on.
    """

    states: List[StateFrame]
    is_resetting: np.ndarray

    action_idx: np.ndarray
    stick_bin_idx: np.ndarray
    btn_flags: np.ndarray
    stick_vals: np.ndarray

    log_probs: np.ndarray
    values: np.ndarray

    rewards: Optional[np.ndarray] = None
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None

    # mirror_x latched at finalize() time so reward.compute() / debug code
    # don't have to fish it out of the StateFrames.  Reward semantics depend
    # on which team the agent is playing.
    mirror_x: bool = False

    # Policy-state snapshot at rollout-start — required for PPO replay.
    initial_kv: Optional[np.ndarray] = None
    initial_prev_labels: Optional[np.ndarray] = None

    @property
    def length(self) -> int:
        return int(self.action_idx.shape[0])


class TrajectoryBuffer:
    """Accumulator that fills one Trajectory of fixed length.

    Carries the *last* observed state across finalize() so the next rollout
    starts where the previous one ended (continuous play, slippi-ai style).
    """

    def __init__(self, length: int):
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")
        self._length = length
        self._states: List[StateFrame] = []
        self._is_resetting: List[bool] = []
        self._action_idx: List[int] = []
        self._stick_bin_idx: List[np.ndarray] = []
        self._btn_flags: List[np.ndarray] = []
        self._stick_vals: List[np.ndarray] = []
        self._log_probs: List[float] = []
        self._values: List[float] = []

    @property
    def length(self) -> int:
        return self._length

    def n_actions(self) -> int:
        return len(self._action_idx)

    def push_state(self, state: StateFrame) -> None:
        self._states.append(state)
        self._is_resetting.append(bool(state.reset_context))

    def push_action(
        self,
        action_idx: int,
        stick_bin_idx: np.ndarray,        # shape (4,), int
        btn_flags: np.ndarray,             # shape (7,), f32
        stick_vals: np.ndarray,            # shape (4,), f32
        log_prob: float,
        value: float,
    ) -> None:
        self._action_idx.append(int(action_idx))
        self._stick_bin_idx.append(np.asarray(stick_bin_idx, dtype=np.int64))
        self._btn_flags.append(np.asarray(btn_flags, dtype=np.float32))
        self._stick_vals.append(np.asarray(stick_vals, dtype=np.float32))
        self._log_probs.append(float(log_prob))
        self._values.append(float(value))

    def is_full(self) -> bool:
        return self.n_actions() >= self._length

    def finalize(
        self,
        mirror_x: bool = False,
        initial_kv: Optional[np.ndarray] = None,
        initial_prev_labels: Optional[np.ndarray] = None,
    ) -> Trajectory:
        """Snapshot the current buffer into a ``Trajectory`` and reset.

        Carries the most recent state (``states[-1]``) forward so the next
        rollout's first state matches the previous rollout's last — that's
        the slippi-ai pattern that lets PPO bootstrap value across
        boundaries (when no reset happened).
        """
        T = self.n_actions()
        if T != self._length:
            raise RuntimeError(
                f"finalize called with {T} actions, expected {self._length}"
            )
        if len(self._states) != T + 1:
            raise RuntimeError(
                f"states={len(self._states)} but actions={T} "
                f"(expected states == actions + 1)"
            )

        traj = Trajectory(
            states=list(self._states),
            is_resetting=np.asarray(self._is_resetting, dtype=bool),
            action_idx=np.asarray(self._action_idx, dtype=np.int64),
            stick_bin_idx=np.stack(self._stick_bin_idx, axis=0),       # (T, 4)
            btn_flags=np.stack(self._btn_flags, axis=0),               # (T, 7)
            stick_vals=np.stack(self._stick_vals, axis=0),             # (T, 4)
            log_probs=np.asarray(self._log_probs, dtype=np.float32),
            values=np.asarray(self._values, dtype=np.float32),
            mirror_x=mirror_x,
            initial_kv=(initial_kv.copy() if initial_kv is not None else None),
            initial_prev_labels=(
                initial_prev_labels.copy() if initial_prev_labels is not None else None
            ),
        )

        last_state = self._states[-1]
        last_reset = self._is_resetting[-1]

        self._states = [last_state]
        self._is_resetting = [last_reset]
        self._action_idx = []
        self._stick_bin_idx = []
        self._btn_flags = []
        self._stick_vals = []
        self._log_probs = []
        self._values = []

        return traj


def compute_gae(
    traj: Trajectory,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> None:
    """Generalized Advantage Estimation, in-place on ``traj``.

    Sets ``traj.advantages`` and ``traj.returns`` (= advantages + values,
    used as targets for the value-head MSE loss).

    ``last_value`` is V(s_T) — the value estimate for the state JUST AFTER
    the final action.  The caller produces this with one extra forward
    pass on the policy, since the trajectory only stored values for
    s_0..s_{T-1}.

    Reset masking: if ``is_resetting[t+1]`` is True, ``state[t+1]`` was the
    first frame of a fresh match.  Bootstrapping V(s_{t+1}) into the
    advantage of action a_t would mix unrelated trajectories, so we zero
    out both ``next_value`` and the carried-forward GAE term at that
    boundary.  Match slippi-ai's ``learner.py`` exactly here.
    """
    if traj.rewards is None:
        raise RuntimeError("rewards must be set before compute_gae")
    T = traj.length
    if traj.rewards.shape != (T,):
        raise RuntimeError(f"rewards shape {traj.rewards.shape} != ({T},)")
    if traj.values.shape != (T,):
        raise RuntimeError(f"values shape {traj.values.shape} != ({T},)")

    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        # Bootstrap value for state s_{t+1}.  is_resetting[t+1] is True
        # when s_{t+1} starts a new episode — don't carry value across.
        if traj.is_resetting[t + 1]:
            next_value = 0.0
            gae = 0.0
        else:
            next_value = float(last_value) if t == T - 1 else float(traj.values[t + 1])
        delta = float(traj.rewards[t]) + gamma * next_value - float(traj.values[t])
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    traj.advantages = advantages.astype(np.float32)
    traj.returns = (advantages + traj.values).astype(np.float32)
