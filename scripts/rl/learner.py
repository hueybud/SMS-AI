"""PPO learner with teacher-KL regularization.

Port-from-memory of slippi-ai's ``slippi_ai/rl/learner.py`` adapted to our
transformer + per-env KV cache.  The shape of the loss is::

    L = - policy_gradient_weight * E[ min( ρ * A, clip(ρ, 1±ε) * A ) ]
        + value_loss_weight       * E[ (V_new(s) - return)² ]
        + kl_teacher_weight       * E[ KL(π_new(·|s) ‖ π_teacher(·|s)) ]
        - entropy_weight          * E[ H(π_new(·|s)) ]

where ρ = π_new / π_old uses log-probs stored at sample time (frozen
π_old / V_old in the trajectory).

Forward KL ``KL(policy ‖ teacher)`` is the slippi-ai default (see
their inline comment on learner.py:270): rewarding the policy for
*refining* the teacher rather than imitating its mistakes.

Replay strategy:
    Each PPO epoch replays the rollout sequentially through the
    policy-with-KV-cache (matching the rollout's evolution exactly).
    The teacher uses the SAME ``initial_kv`` snapshot as the policy and
    evolves its own KV cache during replay — frozen weights, no_grad.
    This is approximate for cross-rollout context but works for MVP
    (the alternative is tracking the teacher's KV during rollout, doubling
    forward cost).

Hyperparams default to slippi-ai's posted values, with ``kl_teacher_weight``
bumped tighter (1e-1, vs the example script's 3e-3) — we want the policy
to stay close to BC at first.

Configurable via ``PPOConfig``.  Stats returned per ``update()`` for logging.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_onnx_transformer import (  # noqa: E402
    BUTTON_DIM,
    CitrusTransformerBC,
    FEATURE_DIM,
    NUM_ACTIONS,
    STICK_BINS,
    STICK_DIM,
    _flat_to_entities_onnx,
)

from .trajectory import Trajectory


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO learner.  Values are slippi-ai defaults
    with two tweaks for our setup (see comments)."""

    # ── PPO objective ──
    epsilon: float = 1e-2          # clip range; slippi uses 1e-2 (very tight)
    policy_gradient_weight: float = 5.0  # loss weight on PPO objective
    value_loss_weight: float = 0.5

    # ── Teacher KL ──
    # Tighter than slippi's example (3e-3): we want the policy to stay
    # near BC during early training.  Dial down if updates feel choked.
    kl_teacher_weight: float = 1e-1

    # ── Entropy bonus ──
    # Slippi uses 0; the BC distribution already has reasonable entropy
    # so we don't need to push for exploration via this term.
    entropy_weight: float = 0.0

    # ── Optimization ──
    learning_rate: float = 3e-4
    grad_clip: float = 0.5
    num_epochs: int = 2            # PPO multi-epoch reuse of one rollout
    advantage_normalize: bool = True


class PPOLearner:
    """Owns the policy + teacher + optimizer; ``update(traj)`` runs PPO.

    Both policy and teacher are ``CitrusTransformerBC`` instances.  Caller
    constructs them, loads BC weights into both, freezes the teacher's
    parameters externally (or relies on this class doing it on init).
    """

    def __init__(
        self,
        policy: CitrusTransformerBC,
        teacher: CitrusTransformerBC,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
        config: PPOConfig = None,
    ):
        self.policy = policy
        self.teacher = teacher
        self.device = device
        self.config = config or PPOConfig()

        # Freeze the teacher.
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.norm_mean = norm_mean.view(1, FEATURE_DIM).to(device)
        self.norm_std = norm_std.view(1, FEATURE_DIM).to(device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )

    # ----------------------------------------------------------------
    # Per-frame forward (returns logits, value, KV-out — used in replay)
    # ----------------------------------------------------------------
    def _forward_one(
        self,
        model: CitrusTransformerBC,
        feat_t: torch.Tensor,         # [1, FEATURE_DIM] already normalized
        kv: torch.Tensor,             # [3, 2, 1, 127, 512]
    ):
        """Run one frame through ``model``, returning the components PPO
        needs.  Differentiable when ``model`` has grad enabled.
        """
        entities = _flat_to_entities_onnx(feat_t)
        frame_emb = model.entity_encoder(entities)
        policy_t, kv_out = model.temporal.forward_cached(frame_emb, kv)

        action_logits = model.ctrl_head.action_head(policy_t)        # [1, 12]
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        action_probs = action_log_probs.exp()
        stk_input = torch.cat([policy_t, action_probs], dim=-1)

        stick_log_probs: List[torch.Tensor] = []
        for i in range(STICK_DIM):
            stk_logits = model.ctrl_head.stk_heads[i](stk_input)     # [1, 21]
            stick_log_probs.append(F.log_softmax(stk_logits, dim=-1))

        # Detach the value head's input so its MSE-against-returns
        # gradient updates only the value_head Linear layer, not the
        # shared transformer encoder.  Without this, value loss perturbs
        # the encoder in random directions, which corrupts the action
        # distribution (observed: kl=2.5 to teacher on rollout 0,
        # policy mode-locked to one action by rollout 5).  See the
        # matching `model.value_head.weight.zero_()` in train._load_bc
        # for why the head starts at zero.
        value = model.value_head(policy_t.detach()).squeeze(-1).squeeze(-1)

        return action_log_probs, stick_log_probs, value, kv_out

    @staticmethod
    def _entropy(log_probs: torch.Tensor) -> torch.Tensor:
        """Categorical entropy ``H = -Σ p log p`` from log-probs."""
        return -(log_probs.exp() * log_probs).sum(dim=-1)

    @staticmethod
    def _kl(p_log_probs: torch.Tensor, q_log_probs: torch.Tensor) -> torch.Tensor:
        """``KL(p || q) = Σ p (log p - log q)`` from log-probs."""
        return (p_log_probs.exp() * (p_log_probs - q_log_probs)).sum(dim=-1)

    # ----------------------------------------------------------------
    # Replay one rollout, accumulating per-frame tensors
    # ----------------------------------------------------------------
    def _replay(self, traj: Trajectory):
        """Sequentially run the trajectory through both policy and teacher.

        Returns dict of per-frame tensors of shape ``(T,)``::
            new_log_probs, new_values, entropies, teacher_kls
        """
        T = traj.length
        device = self.device

        if traj.initial_kv is None or traj.initial_prev_labels is None:
            raise RuntimeError(
                "trajectory missing initial_kv / initial_prev_labels — "
                "the trainer must snapshot agent state before each rollout"
            )

        kv_p = torch.from_numpy(traj.initial_kv).to(device).contiguous()
        kv_t = torch.from_numpy(traj.initial_kv).to(device).contiguous()
        prev_labels = traj.initial_prev_labels.copy()

        new_log_probs = []
        new_values = []
        entropies = []
        teacher_kls = []

        # action_idx and stick_bin_idx as tensors for indexing
        action_idx_all = torch.from_numpy(traj.action_idx).to(device)
        stick_bin_idx_all = torch.from_numpy(traj.stick_bin_idx).to(device)

        for t in range(T):
            # Reset cache + prev_labels at boundaries — mirrors RLAgent.act().
            if traj.is_resetting[t]:
                kv_p = torch.zeros_like(kv_p)
                kv_t = torch.zeros_like(kv_t)
                prev_labels = np.zeros_like(prev_labels)

            # Build 194-feature input.
            core = traj.states[t].core_features
            feat_full = np.concatenate([core, prev_labels], dtype=np.float32)
            feat_t_torch = (
                torch.from_numpy(feat_full).view(1, FEATURE_DIM).to(device)
            )
            feat_t_torch = (feat_t_torch - self.norm_mean) / self.norm_std

            # Policy forward (with grad).
            p_a_lp, p_s_lps, p_value, kv_p_out = self._forward_one(
                self.policy, feat_t_torch, kv_p
            )
            # Teacher forward (no grad).
            with torch.no_grad():
                t_a_lp, t_s_lps, _, kv_t_out = self._forward_one(
                    self.teacher, feat_t_torch, kv_t
                )

            # log π_new(joint sample): pick the per-component log_prob at the
            # sampled index, sum across components.
            a_idx = action_idx_all[t]                       # scalar
            new_lp = p_a_lp[0, a_idx]                       # [] tensor
            for i in range(STICK_DIM):
                bin_idx = stick_bin_idx_all[t, i]
                new_lp = new_lp + p_s_lps[i][0, bin_idx]

            # Entropy: sum over the 5 categorical components.
            ent = self._entropy(p_a_lp[0])
            for i in range(STICK_DIM):
                ent = ent + self._entropy(p_s_lps[i][0])

            # KL(policy || teacher), summed across components.
            kl = self._kl(p_a_lp[0], t_a_lp[0])
            for i in range(STICK_DIM):
                kl = kl + self._kl(p_s_lps[i][0], t_s_lps[i][0])

            new_log_probs.append(new_lp)
            new_values.append(p_value)
            entropies.append(ent)
            teacher_kls.append(kl)

            # Carry KV forward.  Use detach + new tensor so autograd doesn't
            # try to backprop through 240 frames of recurrence — we only
            # need gradients local to each frame's logits/value (PPO is
            # off-policy from the trajectory's perspective).
            kv_p = kv_p_out.detach()
            kv_t = kv_t_out.detach()

            # Carry prev_labels forward — use the SAMPLED action stored in
            # the trajectory (same values that were sent to the env).
            prev_labels[:BUTTON_DIM] = traj.btn_flags[t]
            prev_labels[BUTTON_DIM : BUTTON_DIM + STICK_DIM] = traj.stick_vals[t]

        return (
            torch.stack(new_log_probs),
            torch.stack(new_values),
            torch.stack(entropies),
            torch.stack(teacher_kls),
        )

    # ----------------------------------------------------------------
    # PPO update over a BATCH of trajectories
    # ----------------------------------------------------------------
    def update(self, trajs: List[Trajectory]) -> Dict[str, float]:
        """Run ``num_epochs`` PPO updates over a batch of trajectories.

        Concatenates per-frame outputs across all trajectories so the
        gradient is computed from the union of frames (slippi-ai's
        ``ppo.num_batches`` pattern).  Advantage normalization is over
        the whole batch — outlier events (a single goal in one rollout)
        get diluted across the batch instead of dominating the gradient.

        Returns summary metrics from the last epoch.
        """
        if not trajs:
            raise RuntimeError("update called with empty trajectory list")
        for traj in trajs:
            if traj.advantages is None or traj.returns is None or traj.rewards is None:
                raise RuntimeError(
                    "trajectory missing rewards / advantages / returns — "
                    "call reward.compute() and compute_gae() first"
                )

        device = self.device
        # Concatenate the per-frame frozen tensors that don't change
        # epoch-to-epoch (old log probs, advantages, returns).
        old_log_probs = torch.cat([
            torch.from_numpy(t.log_probs).to(device) for t in trajs
        ])
        advantages = torch.cat([
            torch.from_numpy(t.advantages).to(device) for t in trajs
        ])
        returns = torch.cat([
            torch.from_numpy(t.returns).to(device) for t in trajs
        ])

        if self.config.advantage_normalize and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        last_metrics: Dict[str, float] = {}
        for epoch in range(self.config.num_epochs):
            # Replay each trajectory through the (current) policy + teacher
            # and concatenate per-frame outputs.
            per_traj_outs = [self._replay(t) for t in trajs]
            new_log_probs = torch.cat([o[0] for o in per_traj_outs])
            new_values    = torch.cat([o[1] for o in per_traj_outs])
            entropies     = torch.cat([o[2] for o in per_traj_outs])
            teacher_kls   = torch.cat([o[3] for o in per_traj_outs])

            log_rho = new_log_probs - old_log_probs
            rho = log_rho.exp()
            clipped_rho = torch.clamp(
                rho, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon
            )
            ppo_obj = torch.min(rho * advantages, clipped_rho * advantages).mean()

            value_loss = F.mse_loss(new_values, returns)
            kl_loss = teacher_kls.mean()
            entropy = entropies.mean()

            total_loss = (
                -self.config.policy_gradient_weight * ppo_obj
                + self.config.value_loss_weight * value_loss
                + self.config.kl_teacher_weight * kl_loss
                - self.config.entropy_weight * entropy
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

            last_metrics = {
                "loss/total": float(total_loss.item()),
                "loss/ppo_obj": float(ppo_obj.item()),
                "loss/value": float(value_loss.item()),
                "loss/kl_teacher": float(kl_loss.item()),
                "loss/entropy": float(entropy.item()),
                "log_rho/mean": float(log_rho.mean().item()),
                "log_rho/abs_max": float(log_rho.abs().max().item()),
                "rho/mean": float(rho.mean().item()),
                "grad_norm": float(grad_norm.item()),
                "epoch": epoch,
                "n_trajs": len(trajs),
                "n_frames": int(advantages.numel()),
            }

        return last_metrics
