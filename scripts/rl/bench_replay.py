"""Quick wall-clock comparison: _replay (loop) vs _replay_parallel.

Builds a synthetic 240-frame Trajectory and times one update() call under
each replay mode.  CPU-only here (Thinkpad sanity check) — A6000 numbers
will be ~10–30x faster across the board.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_onnx_transformer import (  # noqa: E402
    BUTTON_DIM,
    CitrusTransformerBC,
    FEATURE_DIM,
    NUM_ACTIONS,
    SEQ_LEN,
    STICK_BINS,
    STICK_DIM,
    TEMPORAL_DIM,
    TEMPORAL_LAYERS,
)

from .learner import PPOConfig, PPOLearner  # noqa: E402
from .protocol import CORE_FEATURE_DIM, StateFrame  # noqa: E402
from .trajectory import Trajectory  # noqa: E402

PREV_ACTION_DIM = BUTTON_DIM + STICK_DIM


def make_traj(T: int = 240) -> Trajectory:
    states = [
        StateFrame(
            frame_id=t,
            reset_context=False,
            mirror_x=False,
            game_phase=4,  # treat synthetic frames as active play
            score_left=0,
            score_right=0,
            core_features=np.random.randn(CORE_FEATURE_DIM).astype(np.float32),
        )
        for t in range(T + 1)
    ]
    is_resetting = np.zeros(T + 1, dtype=bool)
    is_resetting[80] = True  # one mid-rollout reset
    return Trajectory(
        states=states,
        is_resetting=is_resetting,
        action_idx=np.random.randint(0, NUM_ACTIONS, size=T, dtype=np.int64),
        stick_bin_idx=np.random.randint(0, STICK_BINS, size=(T, STICK_DIM), dtype=np.int64),
        btn_flags=np.random.rand(T, BUTTON_DIM).astype(np.float32),
        stick_vals=(np.random.rand(T, STICK_DIM) * 2 - 1).astype(np.float32),
        log_probs=np.random.randn(T).astype(np.float32) * 0.1,
        values=np.random.randn(T).astype(np.float32) * 0.1,
        rewards=np.random.randn(T).astype(np.float32) * 0.1,
        advantages=np.random.randn(T).astype(np.float32) * 0.1,
        returns=np.random.randn(T).astype(np.float32) * 0.1,
        mirror_x=False,
        initial_kv=np.random.randn(
            TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM
        ).astype(np.float32),
        initial_prev_labels=np.random.randn(PREV_ACTION_DIM).astype(np.float32),
    )


def bench(mode: str, n_trajs: int = 4, n_warmup: int = 1, n_iters: int = 3):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    policy = CitrusTransformerBC().to(device)
    teacher = CitrusTransformerBC().to(device)
    teacher.load_state_dict(policy.state_dict())
    norm_mean = torch.zeros(1, FEATURE_DIM)
    norm_std = torch.ones(1, FEATURE_DIM)
    learner = PPOLearner(
        policy=policy,
        teacher=teacher,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
        config=PPOConfig(replay_mode=mode),
    )
    trajs = [make_traj() for _ in range(n_trajs)]

    for _ in range(n_warmup):
        learner.update(trajs)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        learner.update(trajs)
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def main() -> int:
    print("Benchmarking PPO update on CPU (4 trajectories × 240 frames × 2 epochs)")
    print("(numbers are wall-clock seconds per learner.update() call)\n")

    par_min, par_avg = bench("parallel")
    print(f"  parallel: min={par_min:.2f}s  avg={par_avg:.2f}s")

    loop_min, loop_avg = bench("loop")
    print(f"  loop    : min={loop_min:.2f}s  avg={loop_avg:.2f}s")

    print(f"\nspeedup (loop / parallel): {loop_avg / par_avg:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
