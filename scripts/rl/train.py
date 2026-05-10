"""PPO training entry point.

Single-env Phase C MVP.  Loads BC checkpoint as both policy (trainable) and
teacher (frozen).  Spawns Dolphin, connects, then loops::

    snapshot agent state          (kv_cache + prev_labels at rollout start)
    collect 240-frame rollout
    compute per-frame reward      (rl/reward.py)
    compute GAE advantages        (rl/trajectory.py::compute_gae)
    learner.update(traj)          (PPO + teacher KL, num_epochs passes)
    log metrics
    save checkpoint every N rollouts

Run::

    cd "C:/Users/Brian/Documents/SMS AI/scripts"
    py -3.9 -m rl.train                        # all defaults, single env
    py -3.9 -m rl.train --device cuda          # GPU forward pass
    py -3.9 -m rl.train --rollout-length 480   # 8s rollouts instead of 4s
    py -3.9 -m rl.train --reward-overlay       # auto-spawn reward overlay window

The match running on screen will be your-controlled-port (the AI agent)
vs whatever CPU you set up.  Press Ctrl+C to save final weights and exit.
"""

from __future__ import annotations

import argparse
import csv
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Reuse model + constants from the export script.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_onnx_transformer import CitrusTransformerBC  # noqa: E402

from . import reward as reward_mod
from .dolphin import DEFAULT_DOLPHIN_EXE, Dolphin
from .env import Env
from .learner import PPOConfig, PPOLearner
from .rl_agent import RLAgent
from .trajectory import TrajectoryBuffer, compute_gae

DEFAULT_ISO = (
    r"C:\Users\Brian\Downloads\Super Mario Strikers (USA)\Super Mario Strikers (USA).iso"
)
DEFAULT_CHECKPOINT = (
    r"C:\Users\Brian\Documents\SMS AI\runs\transformer_v3\best_model.pt"
)
DEFAULT_NORM_STATS = (
    r"C:\Users\Brian\Documents\SMS AI\runs\transformer_v3\norm_stats.npz"
)
DEFAULT_RUN_DIR = (
    r"C:\Users\Brian\Documents\SMS AI\runs\rl_phase_c"
)


def _load_bc(
    checkpoint_path: str, device: torch.device, zero_value_head: bool = True
) -> CitrusTransformerBC:
    model = CitrusTransformerBC().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        print(
            f"  loading from full checkpoint (epoch {state.get('epoch', '?')})"
        )
        state = state["model"]
    model.load_state_dict(state)

    # The BC training script (`train_transformer.py:525`) marks the value
    # head as "reserved for future RL/PPO" — it's NOT trained by BC, so
    # its weights are random init.  Letting it predict V(s) ~ random ~
    # O(1) ruins GAE: advantages become noise and PPO trains on garbage,
    # which collapses the policy in a few rollouts (observed: kl=2.5 on
    # rollout 0, mode-locked to one action by rollout 5).  Zero the head
    # so V(s)=0 initially → advantages are pure discounted rewards
    # (Monte-Carlo style); the head will learn its own weights as PPO
    # progresses.  See learner._forward_one for the matching detach that
    # keeps value-loss gradients from flowing into the shared encoder.
    if zero_value_head:
        with torch.no_grad():
            model.value_head.weight.zero_()
            model.value_head.bias.zero_()
        print("  zeroed value_head (untrained by BC; learns from PPO)")

    return model


def _wait_for_first_state(env: Env, sock: socket.socket, timeout_s: float):
    """Heartbeat-poll for first STATE.  Reused logic from smoke_test/play_bc."""
    deadline = time.monotonic() + timeout_s
    t0 = time.monotonic()
    while time.monotonic() < deadline:
        sock.settimeout(2.0)
        try:
            return env.recv_state()
        except socket.timeout:
            err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            elapsed = int(time.monotonic() - t0)
            if err != 0:
                raise RuntimeError(
                    f"socket has SO_ERROR={err} after {elapsed}s — connection died"
                )
            print(
                f"[train] still waiting for first STATE... {elapsed}s elapsed; "
                f"navigate to a kickoff if needed",
                flush=True,
            )
    raise TimeoutError(
        f"no STATE within {timeout_s}s; AI port never entered phase 1/4/5"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iso", default=DEFAULT_ISO)
    p.add_argument("--exe", default=DEFAULT_DOLPHIN_EXE)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="BC checkpoint to seed both policy and teacher")
    p.add_argument("--norm-stats", default=DEFAULT_NORM_STATS)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                   help="where to save policy checkpoints + training log")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--rollout-length", type=int, default=240,
                   help="frames per PPO rollout (default 240 = ~4s @ 60fps)")
    p.add_argument("--batch-rollouts", type=int, default=4,
                   help="rollouts to accumulate before each PPO update "
                        "(slippi-ai uses ppo.num_batches=16; we default to 4 "
                        "to keep update wall-clock manageable on CPU)")
    p.add_argument("--save-every", type=int, default=50,
                   help="save policy checkpoint every N rollouts")
    p.add_argument("--max-rollouts", type=int, default=-1,
                   help="stop after N rollouts (-1 = run forever)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--first-state-timeout-s", type=float, default=600.0)
    p.add_argument("--keep-alive", action="store_true",
                   help="leave Dolphin running on exit (read logs etc.)")
    # PPO knobs (override config defaults)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo-epsilon", type=float, default=1e-2)
    p.add_argument("--kl-teacher", type=float, default=1e-2,
                   help="teacher KL weight.  Higher = tighter leash, policy "
                        "stays close to BC.  Lower = looser leash, PPO can "
                        "deviate further toward the reward signal.  A weaker "
                        "BC model calls for a TIGHTER leash, not looser: BC "
                        "has good defensive instincts that a loose leash lets "
                        "PPO discard immediately.  We tried 3e-3 (slippi-ai "
                        "default) and saw defensive collapse; 1e-2 preserves "
                        "BC's defensive behavior while still letting PPO "
                        "improve offensive play over time.  1e-1 was too "
                        "tight — policy couldn't improve at all.")
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--replay-mode", choices=("parallel", "loop"),
                   default="parallel",
                   help="PPO replay path: 'parallel' batches the whole "
                        "rollout through one SDPA per layer (default, "
                        "slippi-ai-style); 'loop' is the original 240-frame "
                        "Python loop kept for fallback / debugging")
    p.add_argument("--pause-on-update", action="store_true",
                   help="Freeze the live game across PPO updates so "
                        "opponents can't score against a held-action AI "
                        "during the wall-clock gap.  Recommended on CPU "
                        "(updates take ~6s); unnecessary on GPU (updates "
                        "are sub-second).  Routed via Core::QueueHostJob "
                        "on the C++ side — see AIController.cpp.")
    p.add_argument("--reward-overlay", action="store_true",
                   help="Spawn the reward overlay window automatically in a "
                        "new console.  Picks a free port and wires everything "
                        "up — no need to run reward_overlay.py manually.")
    p.add_argument("--reward-overlay-port", type=int, default=0,
                   help="Broadcast reward events to 127.0.0.1:<port> (manual "
                        "mode; ignored when --reward-overlay is set).")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    print(f"[train] run_dir={run_dir}")
    print(f"[train] checkpoint={args.checkpoint}")
    print(f"[train] device={args.device}")
    print(f"[train] rollout_length={args.rollout_length}")

    device = torch.device(args.device)

    # ── Load policy + teacher from same BC checkpoint ────────────────────
    print("[train] loading policy...")
    policy = _load_bc(args.checkpoint, device)
    print("[train] loading teacher...")
    teacher = _load_bc(args.checkpoint, device)

    # Norm stats applied in Python (not baked) so we can update later.
    stats = np.load(args.norm_stats)
    norm_mean = torch.from_numpy(stats["mean"].astype(np.float32))
    norm_std = torch.from_numpy(stats["std"].astype(np.float32))
    print(f"[train] norm stats loaded from {args.norm_stats}")

    # ── Optional reward-overlay UDP emitter ──────────────────────────────
    # UDP fire-and-forget — never blocks, dropped packets are fine since
    # the overlay only displays a recent window anyway.
    _overlay_proc: Optional[subprocess.Popen] = None
    if args.reward_overlay:
        # Grab a free UDP port by binding then releasing it.
        _tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _tmp.bind(("127.0.0.1", 0))
        args.reward_overlay_port = _tmp.getsockname()[1]
        _tmp.close()
        _overlay_proc = subprocess.Popen(
            [sys.executable, "-m", "rl.reward_overlay",
             "--port", str(args.reward_overlay_port)],
            cwd=str(_SCRIPTS_DIR),
            creationflags=(subprocess.CREATE_NEW_CONSOLE
                           if sys.platform == "win32" else 0),
        )
        print(
            f"[train] reward overlay spawned on port {args.reward_overlay_port} "
            f"(pid={_overlay_proc.pid})"
        )

    reward_event_sink = None
    if args.reward_overlay_port:
        import json
        _overlay_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _overlay_addr = ("127.0.0.1", args.reward_overlay_port)

        def reward_event_sink(name: str, value: float) -> None:
            try:
                _overlay_sock.sendto(
                    json.dumps({"comp": name, "val": value}).encode("utf-8"),
                    _overlay_addr,
                )
            except OSError:
                pass

        print(f"[train] reward overlay → udp://127.0.0.1:{args.reward_overlay_port}")

    config = PPOConfig(
        epsilon=args.ppo_epsilon,
        kl_teacher_weight=args.kl_teacher,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        replay_mode=args.replay_mode,
    )
    print(f"[train] replay_mode={args.replay_mode}")
    print(f"[train] pause_on_update={args.pause_on_update}")
    learner = PPOLearner(
        policy=policy,
        teacher=teacher,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
        config=config,
    )
    agent = RLAgent(model=policy, norm_mean=norm_mean, norm_std=norm_std,
                    device=device)

    # ── Dolphin ──────────────────────────────────────────────────────────
    with Dolphin(
        iso_path=args.iso,
        ipc_port=args.port,
        dolphin_exe=args.exe,
        keep_alive=args.keep_alive,
    ) as dolphin:
        env = Env(dolphin.socket)
        sock = dolphin.socket

        print("[train] waiting for first STATE packet (heartbeat every 2s)...")
        try:
            state = _wait_for_first_state(env, sock, args.first_state_timeout_s)
        except (TimeoutError, RuntimeError, OSError) as e:
            print(f"[train] FAIL: {e}", file=sys.stderr)
            return 1
        sock.settimeout(None)
        mirror_x = state.mirror_x
        print(
            f"[train] first STATE: frame_id={state.frame_id} "
            f"reset={state.reset_context} mirror={state.mirror_x} "
            f"score=({state.score_left},{state.score_right})"
        )
        print(f"[train] mirror_x latched: {mirror_x}")

        buffer = TrajectoryBuffer(length=args.rollout_length)
        buffer.push_state(state)

        rollout_idx = 0
        total_frames = 0
        log_f = open(log_path, "a", encoding="utf-8", buffering=1)
        log_f.write(f"\n=== train start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        # CSV metrics dump — one row per rollout, append-mode so multiple
        # runs in the same run_dir concatenate.  rl/plot_metrics.py reads
        # this and produces metrics.png.  Same idea as the BC training
        # CSV/PNG: avoids combing log lines to gauge progress.
        csv_path = run_dir / "metrics.csv"
        csv_is_new = not csv_path.exists()
        csv_f = open(csv_path, "a", encoding="utf-8", newline="", buffering=1)
        csv_w = csv.writer(csv_f)
        csv_columns = [
            "wallclock", "rollout", "total_frames", "update_tag",
            "collect_s", "update_s",
            "reward_sum", "reward_events",
            "adv_mean", "adv_std",
            "score_left", "score_right",
            "loss_total", "loss_ppo", "loss_value", "loss_kl_teacher",
            "loss_entropy", "rho_mean", "log_rho_abs_max", "grad_norm",
            "act_avg_ms", "act_max_ms", "unique_actions", "actions_total",
            "drained_states",
        ]
        if csv_is_new:
            csv_w.writerow(csv_columns)
            print(f"[train] metrics csv -> {csv_path}")
        else:
            print(f"[train] metrics csv (appending) -> {csv_path}")
        run_t0 = time.monotonic()

        # Trajectory batch — we accumulate ``--batch-rollouts`` rollouts before
        # running a single PPO update over the union.  Why batch:
        #  - Single-rollout updates on a rare goal-against rollout assign
        #    -10 reward across all 240 frames via GAE, which translates to
        #    "decrease prob of EVERYTHING you just did" — the policy
        #    collapses immediately.  Batching dilutes the outlier event
        #    across 4×240=960 frames, advantage normalization compresses
        #    the dynamic range, gradient direction is more reliable.
        #  - Slippi-ai's ``ppo.num_batches=16`` does the same thing.
        batch_trajs: list = []

        # All-null sentinel for the metrics dict when we skip an update.
        NULL_METRICS = dict.fromkeys(
            ("loss/total", "loss/ppo_obj", "loss/value",
             "loss/kl_teacher", "loss/entropy",
             "log_rho/mean", "log_rho/abs_max", "grad_norm"),
            0.0,
        )
        NULL_METRICS["rho/mean"] = 1.0
        NULL_METRICS["epoch"] = -1

        try:
            while args.max_rollouts < 0 or rollout_idx < args.max_rollouts:
                # Snapshot agent state for THIS rollout's PPO replay.
                init_kv = agent.snapshot_kv()
                init_pl = agent.snapshot_prev_labels()

                t_collect_start = time.monotonic()
                # Diagnostic: time per agent.act() to see whether RL inference
                # is slower than 60fps (= 16.7 ms/frame); track unique actions
                # produced to confirm Gumbel sampling actually varies; track
                # drained STATE packets (= game frames not in trajectory)
                # so we can see how aggressive the drain is being.
                _act_times: list = []
                _seen_actions: set = set()
                _drained_at_start = env.drained_states
                # Per-frame reward accumulator — RewardComputer.step() emits
                # events to the overlay immediately (real-time) rather than
                # waiting until rollout end for the batch compute() call.
                _rc = reward_mod.RewardComputer(
                    mirror_x, event_sink=reward_event_sink
                )
                _rollout_rewards: list = []
                while not buffer.is_full():
                    s_t = state                        # capture before step
                    _t0 = time.monotonic()
                    out = agent.act(state)
                    _act_times.append(time.monotonic() - _t0)
                    buffer.push_action(
                        action_idx=out["action_idx"],
                        stick_bin_idx=out["stick_bin_idx"],
                        btn_flags=out["btn_flags"],
                        stick_vals=out["stick_vals"],
                        log_prob=out["log_prob"],
                        value=out["value"],
                    )
                    # Sample a tag for uniqueness check (action_idx + 4 stick bins).
                    _seen_actions.add(
                        (int(out["action_idx"]),
                         tuple(int(b) for b in out["stick_bin_idx"]))
                    )
                    # Log every 60th action to show what Python is actually
                    # sending and the frame_id we echoed back to C++.
                    if (total_frames % 60) == 0:
                        print(
                            f"[diag] act #{total_frames} frame_id={state.frame_id} "
                            f"action_idx={int(out['action_idx'])} "
                            f"stick_bins={list(int(b) for b in out['stick_bin_idx'])} "
                            f"btn_flags={out['btn_flags'].astype(int).tolist()} "
                            f"stick_vals={[round(float(v), 3) for v in out['stick_vals']]}",
                            flush=True,
                        )
                    state, _drained = env.step(state.frame_id, out["btn_flags"], out["stick_vals"])
                    # Walk every transition in the chain, including drained
                    # intermediate frames, so turnovers/shots that happened
                    # in skipped frames still emit overlay events and
                    # contribute reward.  Drained rewards fold into this
                    # action's slot (no corresponding buffer entry for them).
                    _chain = [s_t] + _drained + [state]
                    _step_reward = sum(
                        _rc.step(_chain[i], _chain[i + 1],
                                 bool(_chain[i + 1].reset_context))
                        for i in range(len(_chain) - 1)
                    )
                    buffer.push_state(state)
                    _rollout_rewards.append(_step_reward)
                    total_frames += 1
                t_collect = time.monotonic() - t_collect_start
                _act_avg = sum(_act_times) / max(len(_act_times), 1)
                _act_max = max(_act_times) if _act_times else 0.0
                _drained_this_rollout = env.drained_states - _drained_at_start
                print(
                    f"[diag] rollout {rollout_idx} "
                    f"act_avg={_act_avg * 1000:.1f}ms "
                    f"act_max={_act_max * 1000:.1f}ms "
                    f"unique_actions={len(_seen_actions)}/{len(_act_times)} "
                    f"drained_states={_drained_this_rollout} "
                    f"(>16.7ms means we lag 60fps; drained=skipped game frames)",
                    flush=True,
                )

                traj = buffer.finalize(
                    mirror_x=mirror_x,
                    initial_kv=init_kv,
                    initial_prev_labels=init_pl,
                )

                # Assign rewards collected per-frame during the rollout loop.
                # flush() drains the poss-progress bucket so the tail shows
                # up in the overlay even if it didn't cross the emit threshold.
                _rc.flush()
                traj.rewards = np.array(_rollout_rewards, dtype=np.float32)
                last_value = agent.value_only(traj.states[-1])
                compute_gae(
                    traj,
                    last_value=last_value,
                    gamma=args.gamma,
                    lam=args.gae_lambda,
                )
                rstats = reward_mod.stats(traj.rewards)

                batch_trajs.append(traj)
                rollout_idx += 1

                # Don't update until the batch is full.  When it is, decide
                # UPD vs skp based on whether ANY rollout in the batch had
                # events (skip-on-null still applies, just at batch
                # granularity now — see learner.update docstring).
                t_update = 0.0
                update_tag = "col"  # collecting more rollouts for the batch
                metrics = NULL_METRICS
                if len(batch_trajs) >= args.batch_rollouts:
                    batch_events = sum(
                        int(np.count_nonzero(t.rewards)) for t in batch_trajs
                    )
                    t_update_start = time.monotonic()
                    if batch_events > 0:
                        # Pause the live game across the update so the AI
                        # doesn't keep holding its last action while
                        # opponents pile on goals — those goals would get
                        # mis-attributed to the next rollout's actions.
                        # Skipped on null batches because there's nothing
                        # to learn anyway and the update returns instantly.
                        # try/finally so a Python crash mid-update still
                        # resumes the game (and the C++ side has its own
                        # auto-resume on connection drop as a backstop).
                        if args.pause_on_update:
                            env.pause()
                        try:
                            metrics = learner.update(batch_trajs)
                        finally:
                            if args.pause_on_update:
                                env.resume()
                        update_tag = "UPD"
                    else:
                        update_tag = "skp"
                    t_update = time.monotonic() - t_update_start
                    batch_trajs = []  # reset for next batch

                line = (
                    f"[train] rollout={rollout_idx - 1:5d} {update_tag} "
                    f"frames={total_frames:7d} "
                    f"reward_sum={rstats['sum']:+.2f} "
                    f"reward_events={rstats['n_events']:3d} "
                    f"adv_mean={float(traj.advantages.mean()):+.3f} "
                    f"loss={metrics['loss/total']:+.4f} "
                    f"ppo={metrics['loss/ppo_obj']:+.4f} "
                    f"value={metrics['loss/value']:.4f} "
                    f"kl={metrics['loss/kl_teacher']:.4f} "
                    f"|gn|={metrics['grad_norm']:.3f} "
                    f"|t| collect={t_collect:.1f}s update={t_update:.1f}s "
                    f"score=({state.score_left},{state.score_right})"
                )
                print(line, flush=True)
                log_f.write(line + "\n")

                csv_w.writerow([
                    f"{time.monotonic() - run_t0:.2f}",
                    rollout_idx - 1,
                    total_frames,
                    update_tag,
                    f"{t_collect:.3f}",
                    f"{t_update:.3f}",
                    f"{rstats['sum']:.4f}",
                    rstats["n_events"],
                    f"{float(traj.advantages.mean()):.4f}",
                    f"{float(traj.advantages.std()):.4f}",
                    state.score_left,
                    state.score_right,
                    f"{metrics['loss/total']:.4f}",
                    f"{metrics['loss/ppo_obj']:.4f}",
                    f"{metrics['loss/value']:.4f}",
                    f"{metrics['loss/kl_teacher']:.4f}",
                    f"{metrics['loss/entropy']:.4f}",
                    f"{metrics['rho/mean']:.4f}",
                    f"{metrics['log_rho/abs_max']:.4f}",
                    f"{metrics['grad_norm']:.4f}",
                    f"{_act_avg * 1000:.2f}",
                    f"{_act_max * 1000:.2f}",
                    len(_seen_actions),
                    len(_act_times),
                    _drained_this_rollout,
                ])

                # Periodic checkpoint.
                if args.save_every > 0 and rollout_idx % args.save_every == 0:
                    ckpt_path = run_dir / f"policy_rollout_{rollout_idx:06d}.pt"
                    torch.save({
                        "model": policy.state_dict(),
                        "rollout_idx": rollout_idx,
                        "total_frames": total_frames,
                    }, ckpt_path)
                    print(f"[train] saved checkpoint -> {ckpt_path}", flush=True)

        except KeyboardInterrupt:
            print("\n[train] interrupted by user", flush=True)
        except (ConnectionError, OSError) as e:
            print(f"[train] socket died: {type(e).__name__}: {e}",
                  file=sys.stderr)
        finally:
            # Always save final weights — even after a partial rollout PPO
            # has changed weights and we want to keep them.
            final_path = run_dir / "policy_final.pt"
            torch.save({
                "model": policy.state_dict(),
                "rollout_idx": rollout_idx,
                "total_frames": total_frames,
            }, final_path)
            print(f"[train] saved final policy -> {final_path}", flush=True)
            log_f.write(
                f"=== train end rollout={rollout_idx} frames={total_frames} ===\n"
            )
            log_f.close()
            csv_f.close()
            if _overlay_proc is not None:
                _overlay_proc.terminate()
                print("[train] reward overlay terminated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
