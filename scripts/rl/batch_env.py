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
        max_restarts_per_env: int = 5,
        savestate_paths: Optional[list[str]] = None,
    ):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        self.n = num_envs
        self.base_port = base_port
        self.first_state_timeout_s = first_state_timeout_s
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        # Savestate rotation pool for match-end restarts.  Each match-end
        # event tears the worker's Dolphin down and relaunches a fresh
        # one with a (possibly different) savestate from this list --
        # instead of calling State::LoadAs on the running instance,
        # which has a non-zero rate of wedging the JIT post-load (see
        # dolphin_wedge_post_savestate memory).  All initial workers
        # boot from ``savestate_path`` (the canonical entry); the pool
        # only matters after the first match ends per env.
        self.savestate_paths: list[str] = (
            list(savestate_paths) if savestate_paths else [savestate_path]
        )
        # Auto-restart: when a worker's socket dies (send/recv/reset error
        # mid-step), tear it down and relaunch in place. ``max_restarts_per_env``
        # is the circuit breaker — if one env hits this many crashes, something
        # is persistently broken with that worker (port collision, OOM, bad
        # savestate) and we'd rather fail loudly than loop forever.
        self.max_restarts_per_env = max_restarts_per_env
        self.env_restart_count: list[int] = [0] * num_envs
        # Indices restarted during the most recent step() call. The trainer
        # reads this immediately after step() to decide whether to discard
        # the in-progress rollout cycle (per-env trajectory partial data is
        # not salvageable once that env reset to a fresh savestate).
        self.restarted_this_step: list[int] = []
        # Stagnation detection: Dolphin can wedge in two distinct modes
        # after a savestate load, both showing FPS=0 in the OSD:
        #
        # (a) Full CPU stall — game JIT + VI hooks all stop.  frame_id
        #     stops advancing.  Detected by frame_id repeats.
        # (b) "FPS=0 / VPS>0" partial stall — game CPU thread is stuck
        #     but VI interrupts keep ticking OnFrameEnd, so frame_id
        #     keeps incrementing.  The game state ISN'T advancing
        #     though, so core_features come back bit-identical across
        #     STATEs.  Detected by features.tobytes() repeats.
        #
        # Either mode produces the same visible bug: no useful game
        # progress + the existing socket-watching auto-restart misses
        # both (socket is alive in both cases).  Trigger force-restart
        # once consecutive-no-progress steps exceed the threshold.
        self.no_progress_steps: list[int] = [0] * num_envs
        self.no_progress_threshold: int = 60
        # Most recent core_features per env, for mode (b) detection.
        # Set to None when an env is fresh / just-restarted so the next
        # comparison is unambiguous.
        self.last_core_features: list = [None] * num_envs

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

        **Resilience:** per-env failures during send/recv/reset are caught
        and trigger ``_restart_env(i)`` instead of bubbling up.  Indices
        that were restarted appear in ``self.restarted_this_step``; the
        trainer should treat those envs as having reset and discard any
        partial in-progress trajectory data for them (simplest: discard the
        whole cycle, since lockstep is broken otherwise).

        Updates ``self.states`` to the gathered (post-reset) states and
        returns ``(next_states, per_env_drained_lists)``.
        """
        self.restarted_this_step = []

        # 1. Scatter sends.  Per-env failures push the env onto failed[].
        failed: set[int] = set()
        for i in range(self.n):
            try:
                self.envs[i].send_action(
                    self.states[i].frame_id,
                    outs[i]["btn_flags"],
                    outs[i]["stick_vals"],
                )
            except (OSError, ConnectionError) as e:
                print(f"[BatchedEnv] env {i}: send_action failed: "
                      f"{type(e).__name__}: {e}", flush=True)
                failed.add(i)

        # 2. Gather recvs (skip envs we've already failed on).
        next_states = list(self.states)
        drained_lists: list[list[StateFrame]] = [[] for _ in range(self.n)]
        for i in range(self.n):
            if i in failed:
                continue
            try:
                s, d = self.envs[i].recv_fresh(drain=drain)
                next_states[i] = s
                drained_lists[i] = d
            except (OSError, ConnectionError) as e:
                print(f"[BatchedEnv] env {i}: recv_fresh failed: "
                      f"{type(e).__name__}: {e}", flush=True)
                failed.add(i)

        # 3. Match-end handling.  We DON'T call env.reset() (which would
        # send IPC RESET and trigger Dolphin's in-process State::LoadAs
        # path).  That path has a non-trivial probability of wedging the
        # JIT post-load (FPS=0 / VPS>0 mode) -- see
        # dolphin_wedge_post_savestate.  Instead we route match-end
        # through the existing restart pass: tear down the Dolphin,
        # relaunch fresh with `-s <savestate>`, which is the same
        # well-tested boot path we use at startup.
        match_end_savestates: dict[int, int] = {}
        for i in range(self.n):
            if i in failed:
                continue
            s = next_states[i]
            if not s.match_end:
                continue
            sid = savestate_picker(i) if savestate_picker else 0
            # Preserve the match-end terminal state in the chain so
            # whatever-goal-ended-the-match isn't lost (its score field
            # carries the final tally).
            drained_lists[i].append(s)
            failed.add(i)
            match_end_savestates[i] = sid

        # 4. Stagnation detection — see __init__ for the two wedge modes
        # this catches.  We check BOTH frame_id repeats (mode a, full CPU
        # stall) and bit-identical core_features (mode b, VI-only ticks,
        # OSD: FPS=0/VPS>0).  Either pattern triggers force-restart once
        # it persists past the threshold.  Skip envs already in `failed`
        # — they're getting restarted anyway and the bookkeeping below
        # resets their state cleanly.
        for i in range(self.n):
            if i in failed:
                self.no_progress_steps[i] = 0
                self.last_core_features[i] = None
                continue
            new = next_states[i]
            if new.reset_context:
                self.no_progress_steps[i] = 0
                self.last_core_features[i] = new.core_features
                continue
            frame_stalled = (new.frame_id == self.states[i].frame_id)
            features_stalled = (
                self.last_core_features[i] is not None
                and new.core_features.tobytes()
                    == self.last_core_features[i].tobytes()
            )
            if frame_stalled or features_stalled:
                self.no_progress_steps[i] += 1
                if self.no_progress_steps[i] >= self.no_progress_threshold:
                    mode = "frame_id" if frame_stalled else "features"
                    print(f"[BatchedEnv] env {i}: STAGNATION "
                          f"({self.no_progress_steps[i]} steps stuck via "
                          f"{mode} check, frame_id={new.frame_id}); "
                          f"forcing restart", flush=True)
                    failed.add(i)
                    self.no_progress_steps[i] = 0
                    self.last_core_features[i] = None
            else:
                self.no_progress_steps[i] = 0
                self.last_core_features[i] = new.core_features

        # 5. Restart any envs that failed (socket crash, stagnation, OR
        # match-end -- all routed through here now so there's exactly
        # one teardown+relaunch path).  Pause the alive envs first so
        # they don't free-run + pile up backlog while we relaunch
        # (~5-15s per worker).  Re-raises if any env has hit
        # max_restarts_per_env -- that's a hard fail, the trainer's
        # outer except will save the policy and exit.  For match-end
        # restarts, drained_lists already has the terminal state
        # appended above; we DON'T blank it out (only blank pre-crash
        # drained data for socket/stagnation failures).
        if failed:
            alive = [i for i in range(self.n) if i not in failed]
            self._pause_envs(alive)
            try:
                for i in sorted(failed):
                    if self.env_restart_count[i] >= self.max_restarts_per_env:
                        raise RuntimeError(
                            f"env {i} hit max_restarts_per_env="
                            f"{self.max_restarts_per_env}; aborting run"
                        )
                    sid = match_end_savestates.get(i)
                    new_state = self._restart_env(i, savestate_id=sid)
                    next_states[i] = new_state
                    if i not in match_end_savestates:
                        # Crash/stagnation: drained data is from before the
                        # failure and is no longer meaningful.  Match-end:
                        # keep drained (we just appended the terminal state).
                        drained_lists[i] = []
                    self.restarted_this_step.append(i)
            finally:
                self._resume_envs(alive)

        self.states = next_states
        return next_states, drained_lists

    # ------------------------------------------------------------------
    # Auto-restart support
    # ------------------------------------------------------------------
    def _pause_envs(self, indices: list[int]) -> None:
        """Best-effort pause on a list of env indices (skips dead ones)."""
        for i in indices:
            try:
                self.envs[i].pause()
            except (OSError, ConnectionError):
                pass

    def _resume_envs(self, indices: list[int]) -> None:
        for i in indices:
            try:
                self.envs[i].resume()
            except (OSError, ConnectionError):
                pass

    def _restart_env(
        self,
        i: int,
        savestate_id: Optional[int] = None,
    ) -> StateFrame:
        """Tear down + relaunch worker ``i`` in place.  Returns the new
        first STATE.  Raises on relaunch failure (caller handles).

        ``savestate_id`` selects a path from ``self.savestate_paths`` for
        the new Dolphin to boot into; if None, reuses the worker's
        current ``savestate_path`` (the right behavior for crash /
        stagnation restarts).  Match-end restarts pass a picker-chosen
        id so stadium diversity is preserved across matches.
        """
        old = self.dolphins[i]
        if savestate_id is not None and self.savestate_paths:
            savestate_path = self.savestate_paths[
                savestate_id % len(self.savestate_paths)
            ]
        else:
            savestate_path = old.savestate_path

        # Archive the dead worker's log so the new launch doesn't truncate
        # it (Dolphin opens log_file with "w").  Indexed suffix so repeat
        # crashes don't overwrite each other.
        if old.log_file is not None and old.log_file.exists():
            try:
                seq = self.env_restart_count[i] + 1
                archived = old.log_file.with_name(
                    f"{old.log_file.stem}.crashed{seq:02d}.log"
                )
                old.log_file.rename(archived)
                print(f"[BatchedEnv] env {i}: archived crash log -> "
                      f"{archived.name}", flush=True)
            except OSError as e:
                print(f"[BatchedEnv] env {i}: log archive failed: {e}",
                      flush=True)

        # Best-effort teardown.  Two parameters worth thinking about:
        # * send_shutdown=True: cheap signal to the worker threads to
        #   set m_stop and exit promptly; safe on a broken socket too
        #   (close() catches the OSError).  Helps clean thread join
        #   for both healthy (match-end) and broken (crash) cases.
        # * wait_s=1.0: AIController's SHUTDOWN handler only stops the
        #   IpcBackend threads -- the Dolphin process never exits on
        #   its own.  We always have to kill().  10s of waiting before
        #   kill is pure waste; 1s is plenty for any graceful behavior
        #   that exists today.  At 16 sequential teardowns per
        #   match-end wave, this cuts the per-wave overhead from ~160s
        #   to ~16s.  Proper fix is a C++ change to actually call
        #   Core::Stop on SHUTDOWN; until then this caps the loss.
        try:
            old.close(send_shutdown=True, wait_s=1.0)
        except Exception as e:
            print(f"[BatchedEnv] env {i}: old.close raised "
                  f"{type(e).__name__}: {e}", flush=True)

        print(f"[BatchedEnv] env {i}: relaunching on port {old.ipc_port} "
              f"savestate={Path(savestate_path).name} "
              f"(restart #{self.env_restart_count[i] + 1})", flush=True)
        new = Dolphin(
            iso_path=old.iso_path,
            ipc_port=old.ipc_port,
            dolphin_exe=old.dolphin_exe,
            headless=True,
            worker_id=i,
            savestate_path=savestate_path,
            user_dir=old.user_dir,
            ai_controlled_port=old.ai_controlled_port,
            ai_mirror_x=old.ai_mirror_x,
            connect_timeout_s=old.connect_timeout_s,
            log_file=old.log_file,
            log_passthrough=False,
        )
        new.start()
        new.connect()
        self.dolphins[i] = new
        self.envs[i] = Env(new.socket)
        state = self._wait_first_state(i)
        self.env_restart_count[i] += 1
        print(f"[BatchedEnv] env {i}: restart complete; "
              f"frame_id={state.frame_id} "
              f"(lifetime restarts={self.env_restart_count[i]})", flush=True)
        return state

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
