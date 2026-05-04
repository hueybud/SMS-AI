"""RL rollout-time agent.

Like ``BCAgent`` but exposes the **categorical samples**, their **joint
log-probability**, and the **value-head estimate** for PPO trajectory
collection.  ``BCAgent`` collapses these inside ``ctrl_head.forward_infer``
and only returns the post-Gumbel ``btn_flags`` + ``stick_vals``; the
learner needs the raw bookkeeping.

Architecture:
    - Identical model class (``CitrusTransformerBC``) as BC.
    - Owns its own ``kv_cache`` + ``prev_labels`` between calls (mirrors
      ``LocalOnnxBackend`` C++-side).
    - Calls model components directly (entity_encoder, temporal,
      ctrl_head's heads, value_head) instead of ``forward_infer``, so we
      can capture (action_idx, stick_bin_idx, log_prob, value).

Snapshots:
    ``snapshot_kv()`` and ``snapshot_prev_labels()`` produce numpy copies
    suitable for ``Trajectory.initial_kv`` / ``initial_prev_labels``.
    Caller takes a snapshot **before** the rollout starts — those are the
    inputs PPO replay will use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_onnx_transformer import (  # noqa: E402
    ACTION_VOCAB,
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

from .protocol import StateFrame

PREV_ACTION_DIM = BUTTON_DIM + STICK_DIM
KV_CACHE_SHAPE = (TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM)


class ActOutput(TypedDict):
    """Everything one ``act`` call needs to ship to both the env and the
    trajectory buffer."""
    action_idx: int                 # 0..NUM_ACTIONS-1
    stick_bin_idx: np.ndarray       # shape (4,) int
    btn_flags: np.ndarray           # shape (7,) f32 — to send to C++
    stick_vals: np.ndarray          # shape (4,) f32 — to send to C++
    log_prob: float                 # joint log π(a, sticks | s) under current policy
    value: float                    # V(s) from the value head


class RLAgent:
    """One agent per env.  Stateful between calls.

    Construct from an existing ``CitrusTransformerBC`` so the trainer can
    share the same nn.Module instance with the learner (for in-place
    weight updates).
    """

    def __init__(
        self,
        model: CitrusTransformerBC,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        # norm stats expected as [1, FEATURE_DIM]; broadcast for the (1, FD)
        # input we'll build per-frame.
        self.norm_mean = norm_mean.view(1, FEATURE_DIM).to(device)
        self.norm_std = norm_std.view(1, FEATURE_DIM).to(device)
        self.action_vocab = ACTION_VOCAB.to(device)

        self.kv_cache = torch.zeros(
            KV_CACHE_SHAPE, dtype=torch.float32, device=device
        )
        self.prev_labels = np.zeros(PREV_ACTION_DIM, dtype=np.float32)

    # ----------------------------------------------------------------
    # State management
    # ----------------------------------------------------------------
    def reset(self) -> None:
        self.kv_cache.zero_()
        self.prev_labels.fill(0.0)

    def snapshot_kv(self) -> np.ndarray:
        """Numpy copy of the current KV cache.  Use BEFORE a rollout starts."""
        return self.kv_cache.detach().cpu().numpy().copy()

    def snapshot_prev_labels(self) -> np.ndarray:
        return self.prev_labels.copy()

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    @torch.no_grad()
    def act(self, state: StateFrame) -> ActOutput:
        if state.reset_context:
            self.reset()

        # 194-feature vector: 183 core + 11 prev_labels.
        feat_full = np.concatenate(
            [state.core_features, self.prev_labels], dtype=np.float32
        )
        feat_t = torch.from_numpy(feat_full).view(1, FEATURE_DIM).to(self.device)
        feat_t = (feat_t - self.norm_mean) / self.norm_std

        entities = _flat_to_entities_onnx(feat_t)
        frame_emb = self.model.entity_encoder(entities)
        policy_t, kv_out = self.model.temporal.forward_cached(
            frame_emb, self.kv_cache
        )  # policy_t: [1, TEMPORAL_DIM]

        # ── Action head: 12-class categorical, Gumbel-max sample ────────
        action_logits = self.model.ctrl_head.action_head(policy_t)  # [1, 12]
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        action_probs = action_log_probs.exp()
        gumbel_a = -torch.log(
            -torch.log(torch.rand_like(action_logits) + 1e-20) + 1e-20
        )
        action_idx_t = (action_logits + gumbel_a).argmax(dim=-1)  # [1]
        action_idx = int(action_idx_t.item())
        log_prob_action = float(action_log_probs[0, action_idx].item())

        # btn_flags from action vocab table
        btn_flags = (
            self.action_vocab[action_idx].detach().cpu().numpy().astype(np.float32)
        )

        # ── Stick heads: 21-bin per axis, conditioned on action probs ───
        stk_input = torch.cat([policy_t, action_probs], dim=-1)
        stick_bin_idx = np.zeros(STICK_DIM, dtype=np.int64)
        stick_vals = np.zeros(STICK_DIM, dtype=np.float32)
        log_prob_sticks = 0.0
        for i in range(STICK_DIM):
            stk_logits = self.model.ctrl_head.stk_heads[i](stk_input)  # [1, 21]
            stk_log_probs = F.log_softmax(stk_logits, dim=-1)
            gumbel_s = -torch.log(
                -torch.log(torch.rand_like(stk_logits) + 1e-20) + 1e-20
            )
            bin_t = (stk_logits + gumbel_s).argmax(dim=-1)
            bin_i = int(bin_t.item())
            stick_bin_idx[i] = bin_i
            stick_vals[i] = bin_i / (STICK_BINS - 1) * 2.0 - 1.0
            log_prob_sticks += float(stk_log_probs[0, bin_i].item())

        # ── Value head ──────────────────────────────────────────────────
        value = float(self.model.value_head(policy_t).item())

        # ── Update internal state for next call ─────────────────────────
        self.kv_cache = kv_out
        self.prev_labels[:BUTTON_DIM] = btn_flags
        self.prev_labels[BUTTON_DIM : BUTTON_DIM + STICK_DIM] = stick_vals

        return ActOutput(
            action_idx=action_idx,
            stick_bin_idx=stick_bin_idx,
            btn_flags=btn_flags,
            stick_vals=stick_vals,
            log_prob=log_prob_action + log_prob_sticks,
            value=value,
        )

    @torch.no_grad()
    def value_only(self, state: StateFrame) -> float:
        """Forward pass that returns only ``V(s)``.  Used for the
        ``last_value`` argument to ``compute_gae`` — the value at the
        state JUST AFTER the last action of a rollout, where we need the
        value but no action.  Does NOT update internal state."""
        feat_full = np.concatenate(
            [state.core_features, self.prev_labels], dtype=np.float32
        )
        feat_t = torch.from_numpy(feat_full).view(1, FEATURE_DIM).to(self.device)
        feat_t = (feat_t - self.norm_mean) / self.norm_std
        entities = _flat_to_entities_onnx(feat_t)
        frame_emb = self.model.entity_encoder(entities)
        # Use a temporary KV; don't mutate self.kv_cache (the next rollout
        # starts from the same state we're peeking at).
        policy_t, _ = self.model.temporal.forward_cached(frame_emb, self.kv_cache)
        return float(self.model.value_head(policy_t).item())
