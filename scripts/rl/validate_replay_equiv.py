"""Validate parallel replay matches sequential cached replay.

Checks that ``CausalTemporalTransformer.forward_with_initial_kv`` produces
the same per-frame outputs as repeated calls to ``forward_cached`` starting
from the same initial KV.  This is the equivalence the PPO learner needs:
``_replay_parallel`` calls the parallel path; rollout-time inference uses
the cached path.  If they diverge, ``log_rho = new_log_probs -
old_log_probs`` is biased even when policy weights are unchanged, and PPO
clipping fires for the wrong reasons.

Tolerance: 1e-4 max abs diff in fp32.  Larger drift means the parallel
path's banded mask is wrong and the segmentation logic in
``_replay_parallel`` won't be safe.

Usage::

    cd "C:/Users/Brian/Documents/SMS AI/scripts"
    py -3.9 -m rl.validate_replay_equiv
"""

from __future__ import annotations

import sys
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
    _flat_to_entities_onnx,
)

from .learner import PPOConfig, PPOLearner  # noqa: E402
from .protocol import CORE_FEATURE_DIM, StateFrame  # noqa: E402
from .trajectory import Trajectory  # noqa: E402

PREV_ACTION_DIM = BUTTON_DIM + STICK_DIM


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cpu")
    model = CitrusTransformerBC().to(device).eval()

    T = 240
    feats = torch.randn(T, FEATURE_DIM, device=device)
    initial_kv = torch.randn(
        TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM, device=device
    )

    # ── Sequential cached path: T forwards, evolving KV cache ──────────────
    cached_outputs = []
    kv = initial_kv.clone()
    with torch.no_grad():
        for t in range(T):
            entities = _flat_to_entities_onnx(feats[t : t + 1])  # [1, N_E, ERD]
            frame_emb = model.entity_encoder(entities)            # [1, D]
            policy_t, kv = model.temporal.forward_cached(frame_emb, kv)
            cached_outputs.append(policy_t)                       # [1, D]
    cached_full = torch.cat(cached_outputs, dim=0)                # [T, D]

    # ── Parallel path: one forward_with_initial_kv over the whole segment ─
    with torch.no_grad():
        entities_seq = _flat_to_entities_onnx(feats)              # [T, N_E, ERD]
        frame_emb_seq = model.entity_encoder(entities_seq)        # [T, D]
        emb = frame_emb_seq.unsqueeze(0)                          # [1, T, D]
        parallel_full = model.temporal.forward_with_initial_kv(
            emb, initial_kv
        ).squeeze(0)                                              # [T, D]

    # ── Compare ────────────────────────────────────────────────────────────
    diff = (cached_full - parallel_full).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    print(f"max  |cached - parallel| = {max_diff:.3e}")
    print(f"mean |cached - parallel| = {mean_diff:.3e}")
    # Also report per-frame max to spot drift accumulation across the segment.
    per_frame = diff.max(dim=-1).values                           # [T]
    print(f"per-frame max diff: first={per_frame[0]:.3e} "
          f"mid={per_frame[T // 2]:.3e} last={per_frame[-1]:.3e}")

    tol = 1e-4
    if max_diff > tol:
        print(f"FAIL: max diff {max_diff:.3e} exceeds tolerance {tol:.0e}")
        return 1

    # ── Reset semantics: zero initial_kv → same as cached path starting
    #    from a fresh zero cache.  This is the contract _replay_parallel
    #    relies on for the post-reset segment.
    zero_kv = torch.zeros_like(initial_kv)
    cached_zero = []
    kv = zero_kv.clone()
    with torch.no_grad():
        for t in range(T):
            entities = _flat_to_entities_onnx(feats[t : t + 1])
            frame_emb = model.entity_encoder(entities)
            policy_t, kv = model.temporal.forward_cached(frame_emb, kv)
            cached_zero.append(policy_t)
    cached_zero_full = torch.cat(cached_zero, dim=0)

    with torch.no_grad():
        parallel_zero_full = model.temporal.forward_with_initial_kv(
            emb, zero_kv
        ).squeeze(0)

    diff_zero = (cached_zero_full - parallel_zero_full).abs()
    max_diff_zero = float(diff_zero.max().item())
    print(f"\n[zero initial_kv] max  |cached - parallel| = {max_diff_zero:.3e}")
    if max_diff_zero > tol:
        print(f"FAIL (zero init): max diff {max_diff_zero:.3e} exceeds {tol:.0e}")
        return 1

    # ── Full _replay vs _replay_parallel on a synthetic trajectory ─────────
    # Builds a Trajectory with reset events at multiple frames so the
    # segmentation logic in _replay_parallel is exercised.  Both paths must
    # produce numerically equivalent (new_log_probs, new_values, entropies,
    # teacher_kls).
    print("\n[full replay] checking _replay vs _replay_parallel ...")
    T = 200
    states = []
    for t in range(T + 1):
        states.append(StateFrame(
            frame_id=t,
            reset_context=False,
            mirror_x=False,
            game_phase=4,  # treat synthetic frames as active play
            score_left=0,
            score_right=0,
            core_features=np.random.randn(CORE_FEATURE_DIM).astype(np.float32),
        ))
    is_resetting = np.zeros(T + 1, dtype=bool)
    # Inject resets mid-rollout to exercise segmentation.
    is_resetting[0] = False     # segment 0 uses initial_kv
    is_resetting[37] = True     # split → segment 1 with zero kv
    is_resetting[150] = True    # split → segment 2 with zero kv

    action_idx = np.random.randint(0, NUM_ACTIONS, size=T, dtype=np.int64)
    stick_bin_idx = np.random.randint(0, STICK_BINS, size=(T, STICK_DIM), dtype=np.int64)
    btn_flags = np.random.rand(T, BUTTON_DIM).astype(np.float32)
    stick_vals = (np.random.rand(T, STICK_DIM) * 2 - 1).astype(np.float32)

    traj = Trajectory(
        states=states,
        is_resetting=is_resetting,
        action_idx=action_idx,
        stick_bin_idx=stick_bin_idx,
        btn_flags=btn_flags,
        stick_vals=stick_vals,
        log_probs=np.zeros(T, dtype=np.float32),
        values=np.zeros(T, dtype=np.float32),
        rewards=np.zeros(T, dtype=np.float32),
        advantages=np.zeros(T, dtype=np.float32),
        returns=np.zeros(T, dtype=np.float32),
        mirror_x=False,
        initial_kv=np.random.randn(
            TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM
        ).astype(np.float32),
        initial_prev_labels=np.random.randn(PREV_ACTION_DIM).astype(np.float32),
    )

    norm_mean = torch.zeros(1, FEATURE_DIM)
    norm_std = torch.ones(1, FEATURE_DIM)
    teacher = CitrusTransformerBC().to(device).eval()
    teacher.load_state_dict(model.state_dict())
    learner = PPOLearner(
        policy=model,
        teacher=teacher,
        norm_mean=norm_mean,
        norm_std=norm_std,
        device=device,
        config=PPOConfig(),
    )
    # Disable grad on policy too — we only need numerical comparison here.
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        loop_out = learner._replay(traj)
        par_out = learner._replay_parallel(traj)

    names = ("new_log_probs", "new_values", "entropies", "teacher_kls")
    full_ok = True
    for name, lo, pa in zip(names, loop_out, par_out):
        d = (lo - pa).abs()
        m = float(d.max().item())
        print(f"  {name:<14s} max diff = {m:.3e}")
        if m > tol:
            print(f"    FAIL: {name} diff {m:.3e} exceeds {tol:.0e}")
            full_ok = False
    if not full_ok:
        return 1

    print(f"\nPASS  (tolerance {tol:.0e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
