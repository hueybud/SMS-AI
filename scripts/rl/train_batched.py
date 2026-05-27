"""Batched PPO training for N parallel headless Dolphins (step 4).

Mirrors :mod:`rl.train` but uses :class:`rl.batch_env.BatchedEnvironment` +
:class:`rl.rl_agent.BatchedRLAgent` so one batched policy forward serves
all N envs per step.  With synchronous IPC pacing
(Project-Citrus ``ai-controller`` ``79dee304b``), each rollout cycle
produces N on-distribution trajectories — and the existing PPO learner
already accepts ``List[Trajectory]``, so the update path is unchanged.

Why this exists (vs ``rl/train.py``):
    Single-env training was capped at ~60 useful decisions/sec (60fps
    real-time × 1 env).  With N=16 sync-paced envs and a batched forward,
    we measured ~187 decisions/sec on Thunder — ~3× the training-signal
    rate.  Diminishing returns hit hard above N=32 on Thunder's box;
    N=16 is the sweet spot for cycle-time-vs-throughput.

Run on ThunderCompute::

    python3 -m rl.train_batched \\
        --num-envs 16 --device cuda

Override defaults via CLI; the defaults match the Thunder layout
(``~/Project-Citrus/build/Binaries`` + ``~/SMS-AI/runs/...``).  Press
Ctrl+C to save final weights and exit.
"""

from __future__ import annotations

import argparse
import csv
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_onnx_transformer import CitrusTransformerBC  # noqa: E402

from . import reward as reward_mod
from .batch_env import BatchedEnvironment
from .dolphin import DEFAULT_NOGUI_EXE, HEADLESS_BASE_PORT
from .learner import PPOConfig, PPOLearner
from .rl_agent import BatchedRLAgent
from .train import _load_bc
from .trajectory import TrajectoryBuffer, compute_gae

# Thunder-friendly defaults; override via CLI for other hosts.
DEFAULT_ISO        = "/home/ubuntu/ISO/Super Mario Strikers (USA).iso"
DEFAULT_SAVESTATE  = "~/Project-Citrus/build/Binaries/rl_palace.sav"
DEFAULT_CHECKPOINT = "/home/ubuntu/SMS-AI/runs/transformer_v3/best_model.pt"
DEFAULT_NORM_STATS = "/home/ubuntu/SMS-AI/runs/transformer_v3/norm_stats.npz"
DEFAULT_RUN_DIR    = "/home/ubuntu/SMS-AI/runs/rl_batched"

# Must match kSavestateFiles in Movie.cpp::InitAIControllerIpc.
NUM_SAVESTATES = 3


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=16,
                   help="parallel headless Dolphin workers. 16 is the "
                        "Thunder sweet spot per the smoke-test ramp; "
                        "ramp down if RAM-constrained, up to ~32 if not.")
    p.add_argument("--iso", default=DEFAULT_ISO)
    p.add_argument("--exe", default=DEFAULT_NOGUI_EXE)
    p.add_argument("--savestate-path", default=DEFAULT_SAVESTATE,
                   help="canonical kickoff savestate every worker boots into")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="BC checkpoint to seed both policy and teacher")
    p.add_argument("--norm-stats", default=DEFAULT_NORM_STATS)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--rollout-length", type=int, default=240,
                   help="frames per env per cycle (240 = ~4s @ 60fps)")
    p.add_argument("--batch-cycles", type=int, default=1,
                   help="cycles of N rollouts to accumulate before each PPO "
                        "update.  Default 1: update every cycle on the N "
                        "trajectories produced this cycle.  At N=16 that's "
                        "16×240=3840 frames per update, comparable to "
                        "slippi-ai's ppo.num_batches setup.")
    p.add_argument("--save-every", type=int, default=20,
                   help="save policy checkpoint every N completed cycles")
    p.add_argument("--max-cycles", type=int, default=-1,
                   help="stop after N cycles (-1 = run forever)")
    p.add_argument("--base-port", type=int, default=HEADLESS_BASE_PORT)
    p.add_argument("--log-dir", default=None,
                   help="per-worker Dolphin stdout/stderr dir; default "
                        "silences worker output (N>1 passthrough is unreadable)")
    p.add_argument("--first-state-timeout-s", type=float, default=600.0)
    # PPO knobs (mirror rl/train.py defaults)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo-epsilon", type=float, default=1e-2)
    p.add_argument("--kl-teacher", type=float, default=1e-2,
                   help="teacher KL weight; see rl/train.py for tuning notes")
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--replay-mode", choices=("parallel", "loop"), default="parallel")
    p.add_argument("--pause-on-update", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="pause all N envs across the PPO update so opponents "
                        "can't pile on goals against a held-action AI")
    p.add_argument("--reset-agent-on-update",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="zero ALL envs' KV/prev_labels after each update "
                        "(post-update weights make pre-update cache stale)")
    p.add_argument("--reward-overlay", action="store_true",
                   help="spawn the reward overlay window (env 0 only — "
                        "showing one game keeps the overlay legible)")
    p.add_argument("--reward-overlay-port", type=int, default=0)
    args = p.parse_args()

    # Expand ~ everywhere users might write it.
    args.savestate_path = str(Path(args.savestate_path).expanduser())
    args.run_dir        = str(Path(args.run_dir).expanduser())
    args.checkpoint     = str(Path(args.checkpoint).expanduser())
    args.norm_stats     = str(Path(args.norm_stats).expanduser())

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train-batched] run_dir={run_dir}")
    print(f"[train-batched] num_envs={args.num_envs} device={args.device} "
          f"rollout_length={args.rollout_length} batch_cycles={args.batch_cycles}")

    device = torch.device(args.device)

    # ── Policy + teacher (same BC seed) + norm stats ─────────────────────
    print("[train-batched] loading policy...")
    policy = _load_bc(args.checkpoint, device)
    print("[train-batched] loading teacher...")
    teacher = _load_bc(args.checkpoint, device)
    stats = np.load(args.norm_stats)
    norm_mean = torch.from_numpy(stats["mean"].astype(np.float32))
    norm_std = torch.from_numpy(stats["std"].astype(np.float32))

    # ── Optional reward overlay (env 0 only) ─────────────────────────────
    # Cross-env aggregation in the overlay would be misleading (events from
    # different games mashed together); pick one game to display.
    _overlay_proc: Optional[subprocess.Popen] = None
    reward_event_sink = None
    if args.reward_overlay:
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
        print(f"[train-batched] reward overlay -> 127.0.0.1:"
              f"{args.reward_overlay_port} (env 0 only)")
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

    # ── Learner + batched agent ──────────────────────────────────────────
    config = PPOConfig(
        epsilon=args.ppo_epsilon,
        kl_teacher_weight=args.kl_teacher,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        replay_mode=args.replay_mode,
    )
    learner = PPOLearner(
        policy=policy, teacher=teacher,
        norm_mean=norm_mean, norm_std=norm_std,
        device=device, config=config,
    )
    agent = BatchedRLAgent(
        model=policy, norm_mean=norm_mean, norm_std=norm_std,
        device=device, num_envs=args.num_envs,
    )

    # ── BatchedEnvironment ───────────────────────────────────────────────
    benv = BatchedEnvironment(
        num_envs=args.num_envs,
        iso_path=args.iso,
        savestate_path=args.savestate_path,
        exe=args.exe,
        base_port=args.base_port,
        log_dir=(Path(args.log_dir) if args.log_dir else None),
        first_state_timeout_s=args.first_state_timeout_s,
    )
    try:
        states = benv.start()
    except Exception as e:
        print(f"[train-batched] FAIL during BatchedEnvironment.start: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        benv.close()
        return 1

    mirror_x = benv.mirror_x
    print(f"[train-batched] all {args.num_envs} envs in play; "
          f"mirror_x={mirror_x}")

    # Each env owns its own trajectory buffer + reward computer.  Sync
    # pacing keeps the envs lockstep, so all N buffers fill simultaneously.
    buffers = [TrajectoryBuffer(length=args.rollout_length)
               for _ in range(args.num_envs)]
    rcs = [
        reward_mod.RewardComputer(
            mirror_x,
            event_sink=(reward_event_sink if i == 0 else None),
        )
        for i in range(args.num_envs)
    ]
    for b, s in zip(buffers, states):
        b.push_state(s)

    # Match-end resets pick a random stadium for diversity (mirrors single-env).
    def savestate_picker(_env_idx: int) -> int:
        return random.randrange(NUM_SAVESTATES)

    # ── Logging ──────────────────────────────────────────────────────────
    log_path = run_dir / "training.log"
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    log_f.write(f"\n=== train-batched start "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"N={args.num_envs} ===\n")
    csv_path = run_dir / "metrics.csv"
    csv_is_new = not csv_path.exists()
    csv_f = open(csv_path, "a", encoding="utf-8", newline="", buffering=1)
    csv_w = csv.writer(csv_f)
    csv_cols = [
        "wallclock", "cycle", "total_frames", "update_tag",
        "collect_s", "update_s", "num_envs",
        "reward_sum", "reward_events",
        "adv_mean", "adv_std",
        "loss_total", "loss_ppo", "loss_value", "loss_kl_teacher",
        "loss_entropy", "rho_mean", "log_rho_abs_max", "grad_norm",
        "act_avg_ms", "act_max_ms", "drained_states",
    ]
    if csv_is_new:
        csv_w.writerow(csv_cols)
        print(f"[train-batched] metrics csv -> {csv_path}")
    else:
        print(f"[train-batched] metrics csv (appending) -> {csv_path}")

    # Null metrics for "no update this cycle" rows.
    NULL_METRICS = dict.fromkeys(
        ("loss/total", "loss/ppo_obj", "loss/value",
         "loss/kl_teacher", "loss/entropy",
         "log_rho/mean", "log_rho/abs_max", "grad_norm"),
        0.0,
    )
    NULL_METRICS["rho/mean"] = 1.0
    NULL_METRICS["epoch"] = -1

    batch_trajs: list = []
    cycle_idx = 0
    total_frames = 0   # sum of env-frames (each cycle adds N * rollout_length)
    run_t0 = time.monotonic()

    try:
        while args.max_cycles < 0 or cycle_idx < args.max_cycles:
            # Snapshot per-env KV + prev_labels for PPO replay.
            init_kvs = agent.snapshot_kv_per_env()
            init_pls = agent.snapshot_prev_labels_per_env()

            t_collect_start = time.monotonic()
            act_times: list = []
            drained_at_start = benv.total_drained
            per_env_rewards: list = [[] for _ in range(args.num_envs)]

            # ── Collect one cycle: each buffer fills to rollout_length ───
            # Sync pacing keeps the envs lockstep — they all reach full at
            # the same iteration.  Defensive `all(...)` rather than checking
            # buffers[0] alone in case of a transient drift.
            while not all(b.is_full() for b in buffers):
                prev_states = list(benv.states)

                _t = time.monotonic()
                outs = agent.act_batch(prev_states)
                act_times.append(time.monotonic() - _t)

                for e in range(args.num_envs):
                    buffers[e].push_action(
                        action_idx=outs[e]["action_idx"],
                        stick_bin_idx=outs[e]["stick_bin_idx"],
                        btn_flags=outs[e]["btn_flags"],
                        stick_vals=outs[e]["stick_vals"],
                        log_prob=outs[e]["log_prob"],
                        value=outs[e]["value"],
                    )

                next_states, drained_lists = benv.step(
                    outs, savestate_picker=savestate_picker
                )

                # Per-env: reward over the chain (prev → drained... → next)
                # and push the next state into that env's trajectory buffer.
                for e in range(args.num_envs):
                    chain = [prev_states[e]] + drained_lists[e] + [next_states[e]]
                    step_reward = sum(
                        rcs[e].step(
                            chain[i], chain[i + 1],
                            bool(chain[i + 1].reset_context),
                            intended_stick=outs[e]["stick_vals"],
                        )
                        for i in range(len(chain) - 1)
                    )
                    per_env_rewards[e].append(step_reward)
                    buffers[e].push_state(next_states[e])

            t_collect = time.monotonic() - t_collect_start
            total_frames += args.num_envs * args.rollout_length
            act_avg = sum(act_times) / max(len(act_times), 1)
            act_max = max(act_times) if act_times else 0.0
            drained_this_cycle = benv.total_drained - drained_at_start

            # ── Finalize N trajectories + compute GAE per env ────────────
            trajs: list = []
            for e in range(args.num_envs):
                rcs[e].flush()
                t = buffers[e].finalize(
                    mirror_x=mirror_x,
                    initial_kv=init_kvs[e],
                    initial_prev_labels=init_pls[e],
                )
                t.rewards = np.array(per_env_rewards[e], dtype=np.float32)
                trajs.append(t)
            # last_value per env — one batched forward at the rollout's
            # final state, no KV mutation.
            last_values = agent.value_only_batch(benv.states)
            for e in range(args.num_envs):
                compute_gae(
                    trajs[e],
                    last_value=float(last_values[e]),
                    gamma=args.gamma,
                    lam=args.gae_lambda,
                )

            # Aggregate stats for the cycle.
            reward_sum = sum(float(t.rewards.sum()) for t in trajs)
            reward_events = sum(int(np.count_nonzero(t.rewards)) for t in trajs)
            adv_concat = np.concatenate([t.advantages for t in trajs])

            batch_trajs.extend(trajs)
            cycle_idx += 1

            # ── Update? ──────────────────────────────────────────────────
            update_tag = "col"
            metrics = NULL_METRICS
            t_update = 0.0
            if len(batch_trajs) >= args.batch_cycles * args.num_envs:
                t_update_start = time.monotonic()
                batch_events = sum(
                    int(np.count_nonzero(t.rewards)) for t in batch_trajs
                )
                if batch_events > 0:
                    if args.pause_on_update:
                        benv.pause_all()
                    try:
                        metrics = learner.update(batch_trajs)
                    finally:
                        if args.pause_on_update:
                            benv.resume_all()
                    update_tag = "UPD"
                    if args.reset_agent_on_update:
                        # All envs' KV/prev_labels were produced by the
                        # pre-update weights; zero them so the next rollout
                        # starts from a clean cache under the new policy.
                        agent.reset_all()
                else:
                    update_tag = "skp"
                t_update = time.monotonic() - t_update_start
                batch_trajs = []

            line = (
                f"[train-batched] cycle={cycle_idx - 1:5d} {update_tag} "
                f"frames={total_frames:9d} "
                f"reward_sum={reward_sum:+.2f} events={reward_events:3d} "
                f"adv_mean={float(adv_concat.mean()):+.3f} "
                f"loss={metrics['loss/total']:+.4f} "
                f"ppo={metrics['loss/ppo_obj']:+.4f} "
                f"value={metrics['loss/value']:.4f} "
                f"kl={metrics['loss/kl_teacher']:.4f} "
                f"|gn|={metrics['grad_norm']:.3f} "
                f"|t| collect={t_collect:.1f}s update={t_update:.1f}s "
                f"act_avg={act_avg * 1000:.1f}ms drained={drained_this_cycle}"
            )
            print(line, flush=True)
            log_f.write(line + "\n")

            csv_w.writerow([
                f"{time.monotonic() - run_t0:.2f}",
                cycle_idx - 1, total_frames, update_tag,
                f"{t_collect:.3f}", f"{t_update:.3f}", args.num_envs,
                f"{reward_sum:.4f}", reward_events,
                f"{float(adv_concat.mean()):.4f}",
                f"{float(adv_concat.std()):.4f}",
                f"{metrics['loss/total']:.4f}",
                f"{metrics['loss/ppo_obj']:.4f}",
                f"{metrics['loss/value']:.4f}",
                f"{metrics['loss/kl_teacher']:.4f}",
                f"{metrics['loss/entropy']:.4f}",
                f"{metrics['rho/mean']:.4f}",
                f"{metrics['log_rho/abs_max']:.4f}",
                f"{metrics['grad_norm']:.4f}",
                f"{act_avg * 1000:.2f}", f"{act_max * 1000:.2f}",
                drained_this_cycle,
            ])

            if args.save_every > 0 and cycle_idx % args.save_every == 0:
                ckpt = run_dir / f"policy_cycle_{cycle_idx:06d}.pt"
                torch.save({
                    "model": policy.state_dict(),
                    "cycle_idx": cycle_idx,
                    "total_frames": total_frames,
                    "num_envs": args.num_envs,
                }, ckpt)
                print(f"[train-batched] saved checkpoint -> {ckpt}",
                      flush=True)

    except KeyboardInterrupt:
        print("\n[train-batched] interrupted by user", flush=True)
    except (ConnectionError, OSError) as e:
        print(f"[train-batched] socket died: {type(e).__name__}: {e}",
              file=sys.stderr)
    finally:
        final_path = run_dir / "policy_final.pt"
        torch.save({
            "model": policy.state_dict(),
            "cycle_idx": cycle_idx,
            "total_frames": total_frames,
            "num_envs": args.num_envs,
        }, final_path)
        print(f"[train-batched] saved final policy -> {final_path}",
              flush=True)
        log_f.write(
            f"=== train-batched end cycle={cycle_idx} "
            f"frames={total_frames} ===\n"
        )
        log_f.close()
        csv_f.close()
        benv.close()
        if _overlay_proc is not None:
            _overlay_proc.terminate()
            print("[train-batched] reward overlay terminated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
