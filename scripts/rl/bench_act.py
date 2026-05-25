"""Isolated inference benchmark for BatchedRLAgent.act_batch.

No Dolphins, no IPC — just synthetic states fed through the batched policy
at several N.  This answers the question the batched smoke test can't:

    Does the policy forward AMORTIZE over the batch (flat act() as N
    grows), or does it scale ~linearly (no batching win)?

The smoke test conflates two costs: the forward itself, and CPU
contention from N free-running emulators starving the Python driver.
This bench removes the emulators so the forward stands alone.

It reports two times per N:
  * ``act_ms``  — full act_batch (feature build + upload + forward +
                  6 host downloads + per-env dict build).  This is what
                  the trainer actually pays.
  * ``fwd_ms``  — entity-encode + temporal + action head only, with no
                  host<->device transfers and no Python output marshaling.

Reading the result:
  * fwd_ms flat, act_ms grows  -> host-side bound (transfers / Python).
        The smoke test's linear act() was CPU contention; synchronous
        C++ pacing (idle emulators) should recover it.
  * fwd_ms also grows ~linearly -> the GPU forward genuinely isn't
        batching on this hardware; need CUDA graphs / op fusion or a
        different parallelism story.

Run (on ThunderCompute)::

    python3 -m rl.bench_act \
        --checkpoint /home/ubuntu/SMS-AI/runs/transformer_v3/best_model.pt \
        --norm-stats /home/ubuntu/SMS-AI/runs/transformer_v3/norm_stats.npz \
        --device cuda --ns 1,2,4,8,16,32

Compare ``--device cuda`` vs ``--device cpu`` to see whether the GPU is
helping at all for this model size.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from export_onnx_transformer import _flat_to_entities_onnx  # noqa: E402

from .protocol import CORE_FEATURE_DIM, StateFrame
from .rl_agent import BatchedRLAgent
from .train import DEFAULT_CHECKPOINT, DEFAULT_NORM_STATS, _load_bc


def _make_states(n: int) -> list[StateFrame]:
    """N synthetic active-play states with random core features."""
    return [
        StateFrame(
            frame_id=1,
            reset_context=False,
            mirror_x=False,
            game_phase=4,
            match_end=False,
            score_left=0,
            score_right=0,
            core_features=np.random.randn(CORE_FEATURE_DIM).astype(np.float32),
        )
        for _ in range(n)
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--norm-stats", default=DEFAULT_NORM_STATS)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--ns", default="1,2,4,8,16,32",
                   help="comma-separated batch sizes to benchmark")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    args = p.parse_args()

    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    ns = [int(x) for x in args.ns.split(",") if x.strip()]

    print(f"[bench-act] device={args.device} iters={args.iters} "
          f"warmup={args.warmup}")
    policy = _load_bc(args.checkpoint, device)
    policy.eval()
    stats = np.load(args.norm_stats)
    norm_mean = torch.from_numpy(stats["mean"].astype(np.float32))
    norm_std = torch.from_numpy(stats["std"].astype(np.float32))

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    print(f"\n{'N':>4} {'act_ms':>9} {'act/env':>8} {'fwd_ms':>9} "
          f"{'fwd/env':>8} {'acts/s':>9} {'env-acts/s':>11}")
    print("-" * 64)
    for n in ns:
        agent = BatchedRLAgent(policy, norm_mean, norm_std, device, n)
        states = _make_states(n)

        # ── Warmup (absorbs CUDA autotune / lazy init) ──────────────────
        for _ in range(args.warmup):
            agent.act_batch(states)
        sync()

        # ── Full act_batch (what the trainer pays) ──────────────────────
        t0 = time.perf_counter()
        for _ in range(args.iters):
            agent.act_batch(states)
        sync()
        act_ms = (time.perf_counter() - t0) / args.iters * 1000.0

        # ── Forward only: no host transfers, no Python marshaling ───────
        feat = agent._build_features(states)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.iters):
                ent = _flat_to_entities_onnx(feat)
                frame_emb = policy.entity_encoder(ent)
                policy_t, _ = policy.temporal.forward_cached(frame_emb, agent.kv_cache)
                _ = policy.ctrl_head.action_head(policy_t)
        sync()
        fwd_ms = (time.perf_counter() - t0) / args.iters * 1000.0

        acts_per_s = 1000.0 / act_ms
        env_acts_per_s = n * acts_per_s
        print(f"{n:>4} {act_ms:>9.2f} {act_ms / n:>8.2f} {fwd_ms:>9.2f} "
              f"{fwd_ms / n:>8.2f} {acts_per_s:>9.1f} {env_acts_per_s:>11.1f}")

    print("\n[bench-act] env-acts/s = decisions/sec across the batch. If it "
          "stays flat as N grows, batching gives no win on this hardware.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
