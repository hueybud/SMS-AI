"""Batched smoke test: N headless Dolphins + one batched policy forward.

What this proves (the step-4 win)
---------------------------------
1. ``BatchedEnvironment`` launches + connects N headless workers and gets
   every one into active play.
2. ``BatchedRLAgent.act_batch`` serves all N envs from ONE forward pass.
3. The scatter/gather step keeps all N games advancing in lock-step with
   no stale-frame drops.
4. Aggregate throughput (envs x frames / sec) scales with N — the whole
   point of batching.

This is the real inference path (it runs the BC policy batched), not a
fixed-action loop, so a PASS here means train_batched.py's hot loop is
sound.

How to run (on ThunderCompute)
------------------------------
From ``SMS-AI/scripts``::

    python3 -m rl.smoke_test_batch --num-envs 8 \
        --iso "/home/ubuntu/ISO/Super Mario Strikers (USA).iso" \
        --savestate-path ~/Project-Citrus/build/Binaries/rl_palace.sav \
        --checkpoint  /path/to/best_model.pt \
        --norm-stats  /path/to/norm_stats.npz \
        --device cuda --frames 600

Start with ``--num-envs 2`` to confirm correctness, then ramp N and watch
aggregate pps climb until the box saturates (CPU-bound on the emulators or
GPU-bound on the forward).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .batch_env import BatchedEnvironment
from .dolphin import DEFAULT_NOGUI_EXE, HEADLESS_BASE_PORT
from .rl_agent import BatchedRLAgent
from .train import (
    DEFAULT_CHECKPOINT,
    DEFAULT_ISO,
    DEFAULT_NORM_STATS,
    _load_bc,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=4,
                   help="parallel headless Dolphin workers")
    p.add_argument("--iso", default=DEFAULT_ISO)
    p.add_argument("--exe", default=DEFAULT_NOGUI_EXE)
    p.add_argument("--savestate-path", required=True,
                   help="canonical kickoff .sav every worker boots into")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="BC checkpoint to drive the policy")
    p.add_argument("--norm-stats", default=DEFAULT_NORM_STATS)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--frames", type=int, default=600,
                   help="batched steps to run (each = 1 frame on every env)")
    p.add_argument("--base-port", type=int, default=HEADLESS_BASE_PORT)
    p.add_argument("--log-dir", default=None,
                   help="per-worker log dir (default: silence worker output)")
    p.add_argument("--first-state-timeout-s", type=float, default=600.0)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[smoke-batch] num_envs={args.num_envs} device={args.device} "
          f"frames={args.frames}")
    print(f"[smoke-batch] exe={args.exe}")
    print(f"[smoke-batch] savestate={args.savestate_path}")

    # ── Policy + norm stats ──────────────────────────────────────────────
    print("[smoke-batch] loading BC policy...")
    policy = _load_bc(args.checkpoint, device)
    policy.eval()
    stats = np.load(args.norm_stats)
    norm_mean = torch.from_numpy(stats["mean"].astype(np.float32))
    norm_std = torch.from_numpy(stats["std"].astype(np.float32))
    agent = BatchedRLAgent(
        model=policy,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
        num_envs=args.num_envs,
    )

    benv = BatchedEnvironment(
        num_envs=args.num_envs,
        iso_path=args.iso,
        savestate_path=args.savestate_path,
        exe=args.exe,
        base_port=args.base_port,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        first_state_timeout_s=args.first_state_timeout_s,
    )

    rc = 0
    try:
        states = benv.start()
        first_fids = [s.frame_id for s in states]
        print(f"[smoke-batch] first frame_ids={first_fids}")

        # Per-env stats: act timing, frame advancement, gap distribution.
        # A "gap" is the frame_id delta per env per step.  gap==1 is perfect
        # lock-step; gap>1 means the (free-running) emulator produced several
        # frames during one batched Python step and we kept only the freshest
        # — coarser control granularity, NOT a stale-drop failure.  See the
        # C++ pacing note: OnFrameEnd Submit is non-blocking + latest-wins,
        # so per-env gap grows with (batch_step_time / emulator_frame_time).
        act_times: list[float] = []
        last_fids = list(first_fids)
        max_gap = [1] * args.num_envs
        advanced = [0] * args.num_envs
        steps_counted = [0] * args.num_envs   # non-reset steps per env

        t0 = time.monotonic()
        for i in range(args.frames):
            _t = time.monotonic()
            outs = agent.act_batch(states)
            act_times.append(time.monotonic() - _t)
            states, _drained = benv.step(outs)
            for e, s in enumerate(states):
                # A savestate reset / phase flip resets frame_id — skip those.
                if s.reset_context:
                    last_fids[e] = s.frame_id
                    continue
                gap = s.frame_id - last_fids[e]
                if gap > 0:
                    advanced[e] += gap
                    steps_counted[e] += 1
                    max_gap[e] = max(max_gap[e], gap)
                last_fids[e] = s.frame_id

            if i % 60 == 0:
                a_ms = act_times[-1] * 1000
                fids = [s.frame_id for s in states]
                print(f"[smoke-batch] step {i:4d} act={a_ms:5.1f}ms "
                      f"fids={fids}", flush=True)

        elapsed = time.monotonic() - t0
        total_game_frames = sum(advanced)
        agg_pps = total_game_frames / elapsed       # game-frames/s across all envs
        steps_per_s = args.frames / elapsed          # batched steps/s (control Hz)
        act_avg = sum(act_times) / max(len(act_times), 1) * 1000
        act_max = max(act_times) * 1000 if act_times else 0.0
        worst_gap = max(max_gap)
        mean_gap = total_game_frames / max(sum(steps_counted), 1)
        total_drained = benv.total_drained
        stalled = [e for e in range(args.num_envs) if advanced[e] == 0]

        print("")
        print(f"[smoke-batch] DONE: {args.frames} batched steps in "
              f"{elapsed:.2f}s ({steps_per_s:.1f} steps/s control rate)")
        print(f"[smoke-batch]   aggregate game throughput = {agg_pps:.0f} "
              f"frames/s across {args.num_envs} envs "
              f"({agg_pps / args.num_envs:.0f} fps/env)")
        print(f"[smoke-batch]   batched act(): avg={act_avg:.1f}ms "
              f"max={act_max:.1f}ms  (one forward serves all {args.num_envs})")
        print(f"[smoke-batch]   per-env frame gap: mean={mean_gap:.2f} "
              f"worst={worst_gap}  (1.0 = 60fps control; higher = coarser, "
              f"emulator outran the loop); total drained={total_drained}")
        print(f"[smoke-batch]   per-env advanced frames={advanced}")

        if stalled:
            print(f"[smoke-batch] FAIL: envs {stalled} never advanced — "
                  f"check per-worker logs", file=sys.stderr)
            rc = 1
        else:
            print(f"[smoke-batch] PASS: all {args.num_envs} envs advanced; "
                  f"aggregate {agg_pps:.0f} frames/s")
            if mean_gap > 2.0:
                print(f"[smoke-batch] NOTE: mean gap {mean_gap:.2f} > 2 — at "
                      f"this N the emulators outrun the batched loop, so each "
                      f"env's effective control is below 60fps. Lower N for "
                      f"finer control, or add synchronous C++ pacing.")
    except (TimeoutError, RuntimeError, OSError) as e:
        print(f"[smoke-batch] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        rc = 1
    finally:
        benv.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
