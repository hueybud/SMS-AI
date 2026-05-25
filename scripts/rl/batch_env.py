"""Batched multi-Dolphin environment for parallel RL rollouts.

Holds N headless ``Dolphin`` processes and drives them in lock-step:
scatter one ACTION to every worker, then gather one STATE back from each.
Because all N sends go out before any gather blocks, the N games advance
*concurrently* — the wall-clock per step is ~one game-frame plus a single
batched policy forward (see :class:`rl.rl_agent.BatchedRLAgent`), not N of
them.

This is step 4 of the ThunderCompute scale-out (see the design memo): the
emulator already runs unbounded headless, so throughput is gated by how
many envs we can keep busy per policy forward.  N envs + one batched
forward is the lever.

Isolation (no Docker — ThunderCompute's container networking isn't
isolated):
  * distinct loopback ports, range-pinned ``base_port + worker_id``
  * distinct ``-u /tmp/citrus_env_<i>`` user dirs (own INI, no patch race)
  * per-worker log files under ``log_dir``

Lifecycle::

    benv = BatchedEnvironment(num_envs=8, iso_path=..., savestate_path=...)
    states = benv.start()                 # list[StateFrame], one per env
    while training:
        outs = agent.act_batch(states)    # list[ActOutput]
        states, drained = benv.step(outs) # advance all N one step
    benv.close()

The trainer owns the per-env TrajectoryBuffers / RewardComputers; this
class only owns process lifecycle + the IPC scatter/gather.
"""

from __future__ import annotations

import random
import socket
import time
from pathlib import Path
from typing import Callable, Optional

from .dolphin import HEADLESS_BASE_PORT, Dolphin
from .env import Env
from .protocol import StateFrame


class BatchedEnvironment:
    """N headless Dolphins driven as one lock-step batched env.

    Parameters
    ----------
    num_envs
        Number of parallel Dolphin workers.
    iso_path
        Game ISO (shared, read-only — safe across workers).
    savestate_path
        ``.sav`` every worker boots into (``-s``).  All workers share the
        same canonical kickoff state for now (no per-env diversity yet).
    exe
        ``dolphin-emu-nogui`` path.  ``None`` => Dolphin's headless default.
    base_port
        Worker ``i`` listens on ``base_port + i``.  Pin per training job to
        avoid cross-job collisions on a shared box.
    ai_controlled_port, ai_mirror_x
        Passed through to every worker's ``[Movie]`` INI block.
    log_dir
        If set, each worker's stdout/stderr goes to
        ``log_dir/env_<i>.log``.  If ``None``, worker output is silenced
        (DEVNULL) — N passthrough streams would be unreadable anyway.
    connect_timeout_s
        Per-worker connect retry budget.
    first_state_timeout_s
        How long to wait for each worker's first STATE (savestate boot is
        fast, but the AI port only emits once it's in phase 1/4/5).
    """

    def __init__(
        self,
        num_envs: int,
        iso_path: str,
        savestate_path: str,
        exe: Optional[str] = None,
        base_port: int = HEADLESS_BASE_PORT,
        ai_controlled_port: int = 0,
        ai_mirror_x: bool = False,
        log_dir: Optional[Path] = None,
        connect_timeout_s: float = 60.0,
        first_state_timeout_s: float = 600.0,
    ):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        self.n = num_envs
        self.base_port = base_port
        self.first_state_timeout_s = first_state_timeout_s
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.dolphins: list[Dolphin] = [
            Dolphin(
                iso_path=iso_path,
                ipc_port=base_port + i,
                dolphin_exe=exe,
                headless=True,
                worker_id=i,
                savestate_path=savestate_path,
                ai_controlled_port=ai_controlled_port,
                ai_mirror_x=ai_mirror_x,
                connect_timeout_s=connect_timeout_s,
                log_file=(self.log_dir / f"env_{i}.log") if self.log_dir else None,
                log_passthrough=False,
            )
            for i in range(num_envs)
        ]
        self.envs: list[Env] = []
        self.states: list[StateFrame] = []
        self.mirror_x: bool = ai_mirror_x

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def start(self) -> list[StateFrame]:
        """Launch + connect all workers, then block for each one's first
        STATE.  Returns the per-env first-state list (also stored on
        ``self.states``).

        Launch order: ``start()`` every process first (they boot in
        parallel), THEN ``connect()`` each.  This overlaps the ~5-15s boot
        across all N workers instead of paying it serially.
        """
        t0 = time.monotonic()
        for d in self.dolphins:
            d.start()
        # Connect serially — each retries with its own deadline; the
        # processes are already booting concurrently.
        for d in self.dolphins:
            d.connect()
        self.envs = [Env(d.socket) for d in self.dolphins]
        print(
            f"[BatchedEnv] {self.n} workers launched + connected in "
            f"{time.monotonic() - t0:.1f}s; waiting for first STATE...",
            flush=True,
        )

        self.states = [
            self._wait_first_state(i) for i in range(self.n)
        ]
        # mirror_x is a per-savestate property; all workers share the
        # canonical savestate, so latch from env 0 and sanity-check.
        self.mirror_x = self.states[0].mirror_x
        if any(s.mirror_x != self.mirror_x for s in self.states):
            raise RuntimeError(
                "workers disagree on mirror_x — they must all boot the same "
                "savestate"
            )
        print(
            f"[BatchedEnv] all {self.n} workers in play after "
            f"{time.monotonic() - t0:.1f}s; mirror_x={self.mirror_x}",
            flush=True,
        )
        return self.states

    def _wait_first_state(self, i: int) -> StateFrame:
        """Heartbeat-poll worker ``i`` for its first STATE packet."""
        env = self.envs[i]
        sock = self.dolphins[i].socket
        deadline = time.monotonic() + self.first_state_timeout_s
        t_start = time.monotonic()
        while time.monotonic() < deadline:
            sock.settimeout(2.0)
            try:
                state = env.recv_state()
                sock.settimeout(None)
                return state
            except socket.timeout:
                err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if err != 0:
                    raise RuntimeError(
                        f"[env {i}] socket SO_ERROR={err} after "
                        f"{int(time.monotonic() - t_start)}s — connection died"
                    )
                print(
                    f"[BatchedEnv] env {i}: still waiting for first STATE "
                    f"({int(time.monotonic() - t_start)}s)...",
                    flush=True,
                )
        raise TimeoutError(
            f"[env {i}] no STATE within {self.first_state_timeout_s}s; "
            f"AI port never entered phase 1/4/5"
        )

    # ------------------------------------------------------------------
    # Scatter / gather step
    # ------------------------------------------------------------------
    def send_actions(self, outs: list) -> None:
        """Scatter one ACTION to every worker (non-blocking).

        Each action echoes the frame_id of the state it responds to
        (``self.states[i].frame_id``) so the C++ staleness filter can drop
        actions that arrive too late.
        """
        if len(outs) != self.n:
            raise ValueError(f"expected {self.n} actions, got {len(outs)}")
        for env, st, out in zip(self.envs, self.states, outs):
            env.send_action(st.frame_id, out["btn_flags"], out["stick_vals"])

    def gather_states(
        self, drain: bool = True
    ) -> tuple[list[StateFrame], list[list[StateFrame]]]:
        """Gather the next STATE (+ drained backlog) from every worker.

        Blocking-recv per env in turn: while we block on env i, envs
        i+1..N-1 keep computing, so the total gather cost is ~one game
        frame, not N.  Returns ``(next_states, per_env_drained_lists)``.
        """
        next_states: list[StateFrame] = []
        drained_lists: list[list[StateFrame]] = []
        for env in self.envs:
            s, d = env.recv_fresh(drain=drain)
            next_states.append(s)
            drained_lists.append(d)
        return next_states, drained_lists

    def step(
        self,
        outs: list,
        drain: bool = True,
        savestate_picker: Optional[Callable[[int], int]] = None,
    ) -> tuple[list[StateFrame], list[list[StateFrame]]]:
        """One batched step: scatter ``outs`` then gather next states.

        On a worker whose returned state has ``match_end=True``, reload a
        savestate (via ``savestate_picker(env_idx) -> savestate_id``;
        default 0) before returning, so its rollout continues.  The
        post-reset state replaces the match-end state in the returned list.

        Updates ``self.states`` to the gathered (post-reset) states and
        returns ``(next_states, per_env_drained_lists)``.
        """
        self.send_actions(outs)
        next_states, drained_lists = self.gather_states(drain=drain)
        for i, s in enumerate(next_states):
            if s.match_end:
                sid = savestate_picker(i) if savestate_picker else 0
                next_states[i] = self.envs[i].reset(sid)
        self.states = next_states
        return next_states, drained_lists

    # ------------------------------------------------------------------
    # Pause / resume (freeze all games across a PPO update)
    # ------------------------------------------------------------------
    def pause_all(self) -> None:
        for env in self.envs:
            env.pause()

    def resume_all(self) -> None:
        for env in self.envs:
            env.resume()

    @property
    def total_drained(self) -> int:
        return sum(env.drained_states for env in self.envs)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        for d in self.dolphins:
            try:
                d.close()
            except Exception:
                pass

    def __enter__(self) -> "BatchedEnvironment":
        try:
            self.start()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def default_savestate_picker(num_savestates: int) -> Callable[[int], int]:
    """Picker that returns a uniformly-random savestate id, ignoring the
    env index — matches the single-env trainer's match-end behavior."""
    def _pick(_env_idx: int) -> int:
        return random.randrange(num_savestates)
    return _pick
