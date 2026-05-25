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


class BatchedRLAgent:
    """N-env batched twin of :class:`RLAgent`.

    Same math as ``RLAgent.act`` but vectorized over a batch dimension so
    N parallel Dolphins are served by ONE forward pass per step instead of
    N.  This is the throughput win step 4 (BatchedEnvironment) exists for:
    the transformer forward dominates per-step wall-clock, and batching it
    amortizes the fixed kernel-launch / Python overhead across all envs.

    State layout mirrors ``RLAgent`` with a leading batch dim N:
        kv_cache    : [L, 2, N, SEQ_LEN-1, TEMPORAL_DIM]
        prev_labels : [N, PREV_ACTION_DIM]

    The model itself is already batch-agnostic (EntityEncoder /
    CausalTemporalTransformer.forward_cached / the heads all carry a
    leading dim), so we feed it stacked features and slice the outputs
    back out per env.

    Per-env helpers (``reset_env``, ``snapshot_kv_per_env``,
    ``snapshot_prev_labels_per_env``) let the trainer build one
    ``Trajectory`` per env from the shared batched state.
    """

    def __init__(
        self,
        model: CitrusTransformerBC,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
        num_envs: int,
    ):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        self.model = model
        self.device = device
        self.n = num_envs
        self.norm_mean = norm_mean.view(1, FEATURE_DIM).to(device)
        self.norm_std = norm_std.view(1, FEATURE_DIM).to(device)
        self.action_vocab = ACTION_VOCAB.to(device)

        self.kv_cache = torch.zeros(
            (TEMPORAL_LAYERS, 2, num_envs, SEQ_LEN - 1, TEMPORAL_DIM),
            dtype=torch.float32,
            device=device,
        )
        self.prev_labels = np.zeros((num_envs, PREV_ACTION_DIM), dtype=np.float32)

    # ----------------------------------------------------------------
    # Per-env state management
    # ----------------------------------------------------------------
    def reset_env(self, i: int) -> None:
        """Zero env ``i``'s KV slot + prev_labels (on reset_context)."""
        self.kv_cache[:, :, i].zero_()
        self.prev_labels[i].fill(0.0)

    def reset_all(self) -> None:
        self.kv_cache.zero_()
        self.prev_labels.fill(0.0)

    def snapshot_kv_per_env(self) -> list[np.ndarray]:
        """One ``[L, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]`` numpy copy per env,
        matching ``Trajectory.initial_kv``.  Take BEFORE a rollout starts."""
        kv = self.kv_cache.detach().cpu().numpy()
        return [kv[:, :, i : i + 1].copy() for i in range(self.n)]

    def snapshot_prev_labels_per_env(self) -> list[np.ndarray]:
        return [self.prev_labels[i].copy() for i in range(self.n)]

    @staticmethod
    def _gumbel_like(logits: torch.Tensor) -> torch.Tensor:
        return -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    def _build_features(self, states: list[StateFrame]) -> torch.Tensor:
        """Stack N (core_features ++ prev_labels) rows → normalized [N, FD].

        Per-row concatenate mirrors ``RLAgent.act`` exactly; np.stack will
        raise if any row isn't FEATURE_DIM wide rather than silently
        leaving uninitialized slots.
        """
        feats = np.stack(
            [
                np.concatenate([s.core_features, self.prev_labels[i]])
                for i, s in enumerate(states)
            ]
        ).astype(np.float32)                                # [N, FEATURE_DIM]
        feat_t = torch.from_numpy(feats).to(self.device)
        return (feat_t - self.norm_mean) / self.norm_std

    @torch.no_grad()
    def act_batch(self, states: list[StateFrame]) -> list[ActOutput]:
        if len(states) != self.n:
            raise ValueError(f"expected {self.n} states, got {len(states)}")

        # Reset KV/prev_labels for any env that just crossed a phase/match
        # boundary — must happen BEFORE we read prev_labels into features.
        for i, s in enumerate(states):
            if s.reset_context:
                self.reset_env(i)

        feat_t = self._build_features(states)               # [N, FD]
        entities = _flat_to_entities_onnx(feat_t)           # [N, n_ent, raw]
        frame_emb = self.model.entity_encoder(entities)     # [N, D]
        policy_t, kv_out = self.model.temporal.forward_cached(
            frame_emb, self.kv_cache
        )                                                   # [N, D], [L,2,N,S-1,D]

        # ── Action head: 12-class categorical, Gumbel-max per row ───────
        action_logits = self.model.ctrl_head.action_head(policy_t)   # [N, 12]
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        action_probs = action_log_probs.exp()
        action_idx = (action_logits + self._gumbel_like(action_logits)).argmax(dim=-1)  # [N]
        lp_action = action_log_probs.gather(1, action_idx.unsqueeze(1)).squeeze(1)      # [N]
        btn_flags = self.action_vocab[action_idx]            # [N, 7]

        # ── Stick heads: 21-bin per axis, conditioned on action probs ───
        stk_input = torch.cat([policy_t, action_probs], dim=-1)      # [N, D+12]
        stick_bin_idx = torch.empty((self.n, STICK_DIM), dtype=torch.long, device=self.device)
        stick_vals_t = torch.empty((self.n, STICK_DIM), dtype=torch.float32, device=self.device)
        lp_sticks = torch.zeros(self.n, device=self.device)
        for ax in range(STICK_DIM):
            stk_logits = self.model.ctrl_head.stk_heads[ax](stk_input)   # [N, 21]
            stk_log_probs = F.log_softmax(stk_logits, dim=-1)
            bins = (stk_logits + self._gumbel_like(stk_logits)).argmax(dim=-1)   # [N]
            stick_bin_idx[:, ax] = bins
            stick_vals_t[:, ax] = bins.float() / (STICK_BINS - 1) * 2.0 - 1.0
            lp_sticks += stk_log_probs.gather(1, bins.unsqueeze(1)).squeeze(1)

        value = self.model.value_head(policy_t).squeeze(-1)  # [N]

        # ── Commit internal state for next call ─────────────────────────
        self.kv_cache = kv_out
        btn_np = btn_flags.detach().cpu().numpy().astype(np.float32)     # [N, 7]
        stick_np = stick_vals_t.detach().cpu().numpy().astype(np.float32)  # [N, 4]
        self.prev_labels[:, :BUTTON_DIM] = btn_np
        self.prev_labels[:, BUTTON_DIM : BUTTON_DIM + STICK_DIM] = stick_np

        # ── Slice batched tensors back into per-env outputs ─────────────
        action_idx_np = action_idx.detach().cpu().numpy()
        stick_bin_np = stick_bin_idx.detach().cpu().numpy()
        lp_total = (lp_action + lp_sticks).detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()
        outs: list[ActOutput] = []
        for i in range(self.n):
            outs.append(ActOutput(
                action_idx=int(action_idx_np[i]),
                stick_bin_idx=stick_bin_np[i].astype(np.int64),
                btn_flags=btn_np[i],
                stick_vals=stick_np[i],
                log_prob=float(lp_total[i]),
                value=float(value_np[i]),
            ))
        return outs

    @torch.no_grad()
    def value_only_batch(self, states: list[StateFrame]) -> np.ndarray:
        """``V(s)`` for each env without mutating KV/prev_labels.  Used for
        the per-env ``last_value`` arg to ``compute_gae`` at rollout end."""
        if len(states) != self.n:
            raise ValueError(f"expected {self.n} states, got {len(states)}")
        feat_t = self._build_features(states)
        entities = _flat_to_entities_onnx(feat_t)
        frame_emb = self.model.entity_encoder(entities)
        policy_t, _ = self.model.temporal.forward_cached(frame_emb, self.kv_cache)
        return self.model.value_head(policy_t).squeeze(-1).detach().cpu().numpy()
