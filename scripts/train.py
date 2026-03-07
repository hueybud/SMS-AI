#!/usr/bin/env python3
"""train.py -- Behavioral cloning LSTM for Citrus Strikers.

Architecture:
  Linear(440→256) → LSTM(256, hidden=512, layers=2)
  → policy: Linear(512→256) → ARControllerHead(buttons=6 BCE, sticks=4×21-way CE)
  → value:  Linear(512→1)   — unused during IL, kept for future RL (PPO)

Training:
  Sequence length T=64 (~1 sec of game time at 60fps).
  Batch: B TBPTT streams × T frames.
  Contextual sampler (default): mixed targeted+natural segment sampling with
  configurable offense/defense/transition and event ratios.
  Legacy sampler: keep all action segments + neutral_keep fraction of neutral segments.
  Temporal split 70/15/15 by match order.

Usage:
    python train.py <X.npy> <y.npy> <ms.npy> <seg.npy> <output_dir>
    python train.py <X.npy> <y.npy> <ms.npy> <seg.npy> <output_dir> --epochs 30 --batch 64
"""

import argparse
import math
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# --- Constants ---------------------------------------------------------------

SEQ_LEN     = 64    # LSTM unroll length (frames per sequence)
FEATURE_DIM = 440   # features per frame  (must match build_dataset.py: 430 core + 10 prev action)
BUTTON_DIM  = 6     # labels 0-5: A, B, X, Y, L, R
STICK_DIM   = 4     # labels 6-9: stick_x, stick_y, cstick_x, cstick_y
LABEL_DIM   = BUTTON_DIM + STICK_DIM
STICK_BINS  = 21    # discrete stick bins per axis [-1,1] → 0..20 (Change 2)

BUTTON_NAMES = ["A", "B", "X", "Y", "L", "R"]
STICK_NAMES  = ["stick_x", "stick_y", "cstick_x", "cstick_y"]

# Offline metric targets
F1_TARGETS       = {"A": 0.45, "B": 0.55, "X": 0.40, "Y": 0.35, "L": 0.30, "R": 0.50}
STICK_ERR_TARGET = 25.0
IDLE_ACC_TARGET  = 0.90

# Contextual sampler labels
CTX_OFFENSE    = "offense"
CTX_DEFENSE    = "defense"
CTX_TRANSITION = "transition"
CTX_NAMES      = [CTX_OFFENSE, CTX_DEFENSE, CTX_TRANSITION]

EVT_SHOT    = "shot"
EVT_PASS    = "pass"
EVT_DEF_HIT = "def_hit"
EVT_ITEM    = "item"
EVT_MOVE    = "move"
EVT_NAMES   = [EVT_SHOT, EVT_PASS, EVT_DEF_HIT, EVT_ITEM, EVT_MOVE]

# build_dataset.py feature index constants (430 core + 10 prev action)
# Carrier flags in the 430-core block:
#   self carrier: 66
#   friendly striker carriers: 102, 138, 174
#   enemy striker carriers: 240, 276, 312, 348
OWN_CARRIER_IDXS   = [66, 102, 138, 174]
ENEMY_CARRIER_IDXS = [240, 276, 312, 348]


# --- Discretization helpers --------------------------------------------------

def float_to_bin(v: torch.Tensor, bins: int = STICK_BINS) -> torch.Tensor:
    """Convert continuous stick values in [-1, 1] to discrete bin indices [0, bins-1].

    bin_index = round((v + 1) / 2 * (bins - 1))
    """
    scaled = (v.clamp(-1.0, 1.0) + 1.0) / 2.0 * (bins - 1)
    return scaled.round().long()


def bin_to_float(b: torch.Tensor, bins: int = STICK_BINS) -> torch.Tensor:
    """Convert bin index [0, bins-1] back to float value in [-1, 1]."""
    return b.float() / (bins - 1) * 2.0 - 1.0


# --- Model -------------------------------------------------------------------

class ARControllerHead(nn.Module):
    """Autoregressive controller head.

    Predicts A, B, X, Y, L, R buttons (single logit → BCE) then
    stick_x, stick_y, cstick_x, cstick_y (stick_bins-way categorical → CE)
    in sequence.  A shared residual vector is updated after each prediction
    using a teacher-forced (training) or argmax (inference) embedding of the
    previous component in the sequence.
    """

    RESIDUAL_DIM = 128

    def __init__(self, input_dim: int, stick_bins: int = STICK_BINS):
        super().__init__()
        self.stick_bins = stick_bins
        R = self.RESIDUAL_DIM

        # Project LSTM policy output to initial residual
        self.residual_proj = nn.Linear(input_dim, R)

        # Button heads: single scalar logit each (→ BCE with pos_weights)
        # Input: [residual (R), prev-button one-hot (2)]
        self.btn_heads  = nn.ModuleList([nn.Linear(R + 2, 1) for _ in range(BUTTON_DIM)])
        self.btn_embeds = nn.ModuleList([nn.Linear(2, R)     for _ in range(BUTTON_DIM)])

        # Stick heads: stick_bins logits each (→ CrossEntropyLoss)
        # Input: [residual (R), prev-stick one-hot (stick_bins)]
        self.stk_heads  = nn.ModuleList([nn.Linear(R + stick_bins, stick_bins)
                                         for _ in range(STICK_DIM)])
        self.stk_embeds = nn.ModuleList([nn.Linear(stick_bins, R)
                                         for _ in range(STICK_DIM)])

    @staticmethod
    def _btn_emb(x: torch.Tensor) -> torch.Tensor:
        """Binary one-hot: x (float 0/1 or bool) → [..., 2]."""
        xf = x.float()
        return torch.stack([1.0 - xf, xf], dim=-1)

    def _stk_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Categorical one-hot: x (int64 [0, stick_bins)) → [..., stick_bins]."""
        return torch.nn.functional.one_hot(x.long(), self.stick_bins).float()

    def forward_train(self, policy: torch.Tensor,
                      btn_targets: torch.Tensor,
                      stk_targets: torch.Tensor):
        """Teacher-forced forward for training.

        policy:      [B, T, input_dim]
        btn_targets: [B, T, BUTTON_DIM] float (0 or 1)
        stk_targets: [B, T, STICK_DIM]  int64 [0, stick_bins)

        Returns:
            btn_logits:    [B, T, BUTTON_DIM] — stacked single-logit outputs
            stk_logit_list: list of STICK_DIM tensors each [B, T, stick_bins]
        """
        B, T, _ = policy.shape
        R = self.residual_proj(policy)   # [B, T, RESIDUAL_DIM]

        btn_logit_list = []
        for i in range(BUTTON_DIM):
            prev_emb = (self._btn_emb(btn_targets[:, :, i - 1])
                        if i > 0 else torch.zeros(B, T, 2, device=policy.device))
            logit = self.btn_heads[i](torch.cat([R, prev_emb], dim=-1))  # [B, T, 1]
            btn_logit_list.append(logit.squeeze(-1))                      # [B, T]
            R = R + self.btn_embeds[i](self._btn_emb(btn_targets[:, :, i]))

        btn_logits = torch.stack(btn_logit_list, dim=-1)   # [B, T, BUTTON_DIM]

        stk_logit_list = []
        for i in range(STICK_DIM):
            prev_emb = (self._stk_emb(stk_targets[:, :, i - 1])
                        if i > 0 else torch.zeros(B, T, self.stick_bins, device=policy.device))
            logits = self.stk_heads[i](torch.cat([R, prev_emb], dim=-1))  # [B, T, bins]
            stk_logit_list.append(logits)
            R = R + self.stk_embeds[i](self._stk_emb(stk_targets[:, :, i]))

        return btn_logits, stk_logit_list

    def forward_infer(self, policy_t: torch.Tensor):
        """Autoregressive inference (argmax, no teacher forcing).

        policy_t: [B, input_dim]  — single time step (time dim already squeezed)

        Returns:
            btn_probs:  [B, BUTTON_DIM] float — sigmoid probability of each button press
            stick_bins: [B, STICK_DIM]  int64 — argmax bin index per axis
        """
        B = policy_t.shape[0]
        R = self.residual_proj(policy_t)   # [B, RESIDUAL_DIM]

        btn_probs_list = []
        prev_emb = torch.zeros(B, 2, device=policy_t.device)
        for i in range(BUTTON_DIM):
            logit = self.btn_heads[i](torch.cat([R, prev_emb], dim=-1))  # [B, 1]
            prob  = torch.sigmoid(logit.squeeze(-1))                      # [B]
            btn_probs_list.append(prob)
            prev_emb = self._btn_emb(prob > 0.5)
            R = R + self.btn_embeds[i](prev_emb)

        btn_probs = torch.stack(btn_probs_list, dim=-1)   # [B, BUTTON_DIM]

        stk_bins_list = []
        prev_emb = torch.zeros(B, self.stick_bins, device=policy_t.device)
        for i in range(STICK_DIM):
            logits = self.stk_heads[i](torch.cat([R, prev_emb], dim=-1))  # [B, bins]
            bins_i = torch.argmax(logits, dim=-1)                          # [B]
            stk_bins_list.append(bins_i)
            prev_emb = self._stk_emb(bins_i)
            R = R + self.stk_embeds[i](prev_emb)

        stick_bins = torch.stack(stk_bins_list, dim=-1)   # [B, STICK_DIM]
        return btn_probs, stick_bins


class StrikersLSTM(nn.Module):
    """2-layer LSTM behavioral cloning model with autoregressive controller head.

    Inputs (training):
        x            [B, T, FEATURE_DIM]
        btn_targets  [B, T, BUTTON_DIM] float  — teacher forcing for AR head
        stk_targets  [B, T, STICK_DIM]  int64  — bin indices for AR head
    Outputs (training):
        btn_logits     [B, T, BUTTON_DIM]         — single logits → BCE
        stk_logit_list list[STICK_DIM × [B,T,BINS]] — categorical logits → CE
        value          [B, T, 1], h_out, c_out

    At inference (ONNX wrapper, T=1, no targets):
        features [1,440], h_in [2,1,512], c_in [2,1,512]
        → btn_probs [1,6], stick_vals [1,4] (bins→float), h_out, c_out
    """

    HIDDEN = 512
    LAYERS = 2
    PROJ   = 256

    def __init__(self, feature_dim: int = FEATURE_DIM, stick_bins: int = STICK_BINS):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, self.PROJ)
        self.lstm       = nn.LSTM(self.PROJ, self.HIDDEN,
                                  num_layers=self.LAYERS, batch_first=True)
        self.policy_fc  = nn.Linear(self.HIDDEN, self.PROJ)
        self.ar_head    = ARControllerHead(self.PROJ, stick_bins)
        self.value_head = nn.Linear(self.HIDDEN, 1)   # for future PPO

    def forward(self, x: torch.Tensor,
                h: Optional[torch.Tensor] = None,
                c: Optional[torch.Tensor] = None,
                btn_targets: Optional[torch.Tensor] = None,
                stk_targets: Optional[torch.Tensor] = None):
        """
        x:           [B, T, FEATURE_DIM]
        h, c:        [LAYERS, B, HIDDEN]  (None → zero-initialized by LSTM)
        btn_targets: [B, T, BUTTON_DIM] float — provide for teacher-forced training
        stk_targets: [B, T, STICK_DIM]  int64 — provide for teacher-forced training

        Training (targets provided):
            returns btn_logits [B,T,BUTTON_DIM], stk_logit_list, value [B,T,1], h_out, c_out
        Inference (targets=None, T=1):
            returns btn_probs [B,BUTTON_DIM], stick_bins [B,STICK_DIM], value, h_out, c_out
        """
        proj = torch.relu(self.input_proj(x))           # [B, T, PROJ]
        hc   = (h, c) if (h is not None and c is not None) else None
        out, (h_out, c_out) = self.lstm(proj, hc)       # out: [B, T, HIDDEN]
        policy = torch.relu(self.policy_fc(out))         # [B, T, PROJ]
        value  = self.value_head(out)                    # [B, T, 1]

        if btn_targets is not None:
            btn_logits, stk_logit_list = self.ar_head.forward_train(
                policy, btn_targets, stk_targets)
            return btn_logits, stk_logit_list, value, h_out, c_out
        else:
            # Inference path: single frame (T=1), squeeze time dim
            btn_probs, stick_bins = self.ar_head.forward_infer(policy[:, 0, :])
            return btn_probs, stick_bins, value, h_out, c_out


# --- Sequence index building -------------------------------------------------

def build_seq_indices(seg: np.ndarray, T: int) -> np.ndarray:
    """Return array of sequence start frame indices.

    Enumerates non-overlapping T-frame windows within each contiguous segment.
    Windows that would span a segment boundary are excluded.

    seg: (N,) int32 — segment ID per frame (monotonically increasing)
    T:   sequence length
    Returns: (S,) int64 — starting frame index of each valid sequence
    """
    n = len(seg)
    if n < T:
        return np.array([], dtype=np.int64)

    # Segment boundaries: positions where seg ID changes
    boundaries = np.where(np.diff(seg) != 0)[0] + 1   # index of first frame in new seg
    seg_starts = np.concatenate([[0], boundaries])
    seg_ends   = np.concatenate([boundaries, [n]])      # exclusive

    starts = []
    for s_start, s_end in zip(seg_starts, seg_ends):
        seg_len = int(s_end - s_start)
        # Non-overlapping windows; drop the tail if it's < T frames
        for seq_start in range(int(s_start), int(s_end) - T + 1, T):
            starts.append(seq_start)

    return np.array(starts, dtype=np.int64)


def temporal_split_seqs(seq_starts: np.ndarray, match_starts: np.ndarray,
                        total_frames: int, T: int,
                        train_frac: float = 0.70, val_frac: float = 0.15):
    """Split sequence start indices by temporal match order (70/15/15)."""
    n   = len(match_starts)
    i_v = int(n * train_frac)
    i_t = int(n * (train_frac + val_frac))

    train_end = int(match_starts[i_v]) if i_v < n else total_frames
    val_end   = int(match_starts[i_t]) if i_t < n else total_frames

    # A sequence belongs to a split if its entire window [s, s+T) lies within
    train_idx = seq_starts[seq_starts + T <= train_end]
    val_idx   = seq_starts[(seq_starts >= train_end) & (seq_starts + T <= val_end)]
    test_idx  = seq_starts[seq_starts >= val_end]
    return train_idx, val_idx, test_idx, train_end


def compute_norm_stats(X_mm: np.ndarray, n_frames: int,
                       chunk_size: int = 200_000) -> tuple:
    """Compute per-feature mean and std over the first n_frames rows of X_mm.

    Uses float64 accumulators for numerical stability.
    Features with std < 1e-6 (constants, e.g. always-zero one-hot dims) get std=1.0
    so division is a no-op rather than blowing up.
    """
    F  = X_mm.shape[1]
    s1 = np.zeros(F, dtype=np.float64)
    s2 = np.zeros(F, dtype=np.float64)
    for start in range(0, n_frames, chunk_size):
        chunk = X_mm[start:min(start + chunk_size, n_frames)].astype(np.float64)
        s1 += chunk.sum(axis=0)
        s2 += (chunk ** 2).sum(axis=0)
    mean = s1 / n_frames
    var  = np.maximum(s2 / n_frames - mean ** 2, 0.0)
    std  = np.sqrt(var)
    std  = np.where(std < 1e-6, 1.0, std)   # constant dims → scale 1
    return mean.astype(np.float32), std.astype(np.float32)


def oversample_seqs(seq_starts: np.ndarray, y: np.ndarray, T: int,
                    neutral_keep: float, seed: int) -> np.ndarray:
    """Sequence-level oversampling.

    Keep all sequences that contain ≥1 frame with any button pressed (A-R, cols 0-5)
    OR any significant c-stick deflection (cols 8-9, |val| > 0.15 — captures dekes).
    Keep neutral_keep fraction of fully-neutral sequences.
    """
    n = len(seq_starts)
    action_mask = np.zeros(n, dtype=bool)

    # Vectorised check in chunks to avoid OOM on large datasets
    CHUNK = 50_000
    for i in range(0, n, CHUNK):
        batch = seq_starts[i:i + CHUNK]
        # idx: [chunk, T] → y_chunk: [chunk, T, 6]
        idx      = batch[:, None] + np.arange(T, dtype=np.int64)[None, :]
        idx_flat = idx.ravel()
        y_chunk  = y[idx_flat, :6].reshape(len(batch), T, 6)
        btn_mask = np.any(y_chunk > 0.5, axis=(1, 2))
        # Also flag sequences with c-stick deflection (deke inputs, cols 8-9)
        cstick   = y[idx_flat, 8:10].reshape(len(batch), T, 2)
        deke_mask = np.any(np.abs(cstick) > 0.15, axis=(1, 2))
        action_mask[i:i + len(batch)] = btn_mask | deke_mask

    action_seqs  = seq_starts[action_mask]
    neutral_seqs = seq_starts[~action_mask]

    rng    = np.random.default_rng(seed)
    n_keep = int(len(neutral_seqs) * neutral_keep)
    neutral_sample = (rng.choice(neutral_seqs, size=n_keep, replace=False)
                      if n_keep > 0 else np.array([], dtype=np.int64))

    combined = np.concatenate([action_seqs, neutral_sample])
    return np.sort(combined)


def compute_pos_weights(y_train: np.ndarray):
    """Compute BCE positive-class weights from training set button rates."""
    rates   = y_train[:, :BUTTON_DIM].mean(axis=0)
    weights = np.clip((1.0 - rates) / np.maximum(rates, 1e-6), 1.0, 20.0)
    weights[4] = min((1.0 - rates[4]) / max(rates[4], 1e-6), 60.0)  # L: higher cap
    print("Button rates and pos_weights:")
    for name, rate, w in zip(BUTTON_NAMES, rates, weights):
        print(f"  {name:<3s}  rate={rate:.4f}  pos_weight={w:.2f}")
    return torch.tensor(weights, dtype=torch.float32), rates


# --- Segment-level helpers for stateful TBPTT --------------------------------

def build_seg_ranges(seg: np.ndarray, end: int) -> list:
    """Return (start, end) frame index pairs for each segment in seg[:end]."""
    seg_slice = seg[:end]
    if len(seg_slice) == 0:
        return []
    boundaries = np.where(np.diff(seg_slice) != 0)[0] + 1
    seg_starts = np.concatenate([[0], boundaries])
    seg_ends   = np.concatenate([boundaries, [end]])
    return [(int(s), int(e)) for s, e in zip(seg_starts, seg_ends)]


def oversample_segments(seg_ranges: list, y: np.ndarray,
                        neutral_keep: float, seed: int) -> list:
    """Segment-level oversampling.

    Keeps all segments containing >=1 button press (cols 0-5) or c-stick
    deflection >0.15 (cols 8-9, deke inputs).  Keeps neutral_keep fraction
    of fully neutral segments.  Returns a sorted list of (start, end) pairs.
    """
    rng = np.random.default_rng(seed)
    action, neutral = [], []
    for s, e in seg_ranges:
        labels   = y[s:e]
        has_btn  = bool(np.any(labels[:, :6] > 0.5))
        has_deke = bool(np.any(np.abs(labels[:, 8:10]) > 0.15))
        (action if (has_btn or has_deke) else neutral).append((s, e))
    n_keep   = int(len(neutral) * neutral_keep)
    keep_idx = rng.choice(len(neutral), size=min(n_keep, len(neutral)), replace=False)
    combined = action + [neutral[i] for i in keep_idx]
    combined.sort(key=lambda x: x[0])
    return combined


def classify_segment_context(x_seg: np.ndarray) -> str:
    """Classify a segment as offense / defense / transition from possession cues."""
    if len(x_seg) == 0:
        return CTX_TRANSITION

    own_has_ball   = np.any(x_seg[:, OWN_CARRIER_IDXS] > 0.5, axis=1)
    enemy_has_ball = np.any(x_seg[:, ENEMY_CARRIER_IDXS] > 0.5, axis=1)
    own_frac       = float(own_has_ball.mean())
    enemy_frac     = float(enemy_has_ball.mean())

    if own_frac >= 0.35 and own_frac > enemy_frac + 0.05:
        return CTX_OFFENSE
    if enemy_frac >= 0.35 and enemy_frac > own_frac + 0.05:
        return CTX_DEFENSE
    return CTX_TRANSITION


def classify_segment_event(context: str, y_seg: np.ndarray) -> str:
    """Primary event label for one segment (used by contextual sampler)."""
    has_a     = bool(np.any(y_seg[:, 0] > 0.5))
    has_b     = bool(np.any(y_seg[:, 1] > 0.5))
    has_x     = bool(np.any(y_seg[:, 2] > 0.5))
    has_y     = bool(np.any(y_seg[:, 3] > 0.5))
    has_cdeke = bool(np.any(np.abs(y_seg[:, 8:10]) > 0.15))

    if context == CTX_OFFENSE and has_b:
        return EVT_SHOT
    if context == CTX_OFFENSE and has_a:
        return EVT_PASS
    if context == CTX_DEFENSE and (has_y or has_cdeke):
        return EVT_DEF_HIT
    if has_x:
        return EVT_ITEM
    return EVT_MOVE


def build_segment_metadata(seg_ranges: list, X_mm: np.ndarray, y: np.ndarray) -> list:
    """Build context/event tags for every trainable segment range."""
    meta = []
    for s, e in seg_ranges:
        x_seg = X_mm[s:e, :430]  # core features only; ignore previous-action tail
        y_seg = y[s:e]
        context = classify_segment_context(x_seg)
        event   = classify_segment_event(context, y_seg)
        meta.append({"range": (s, e), "context": context, "event": event})
    return meta


def print_segment_distribution(tag: str, meta: list) -> None:
    """Print context/event counts for a segment metadata list."""
    ctx_counts = Counter(m["context"] for m in meta)
    evt_counts = Counter(m["event"] for m in meta)
    total = max(len(meta), 1)
    print(f"\n{tag} segments: {len(meta):,}")
    print("  Context mix:")
    for c in CTX_NAMES:
        n = ctx_counts.get(c, 0)
        print(f"    {c:<10s} {n:>8,}  ({100*n/total:5.1f}%)")
    print("  Event mix:")
    for e in EVT_NAMES:
        n = evt_counts.get(e, 0)
        print(f"    {e:<10s} {n:>8,}  ({100*n/total:5.1f}%)")


def _norm_weights(weights: dict, keys: list) -> dict:
    vals = [max(0.0, float(weights.get(k, 0.0))) for k in keys]
    s = sum(vals)
    if s <= 0:
        u = 1.0 / max(len(keys), 1)
        return {k: u for k in keys}
    return {k: v / s for k, v in zip(keys, vals)}


def _allocate_counts(total: int, weights: dict, keys: list) -> dict:
    """Largest-remainder integer allocation that sums exactly to total."""
    if total <= 0 or len(keys) == 0:
        return {k: 0 for k in keys}

    w = _norm_weights(weights, keys)
    raw = [w[k] * total for k in keys]
    base = [int(math.floor(x)) for x in raw]
    rem = total - sum(base)
    frac_order = np.argsort([x - b for x, b in zip(raw, base)])[::-1]
    for i in range(rem):
        base[int(frac_order[i % len(keys)])] += 1
    return {k: b for k, b in zip(keys, base)}


def sample_contextual_segments(seg_meta: list,
                               target_count: int,
                               targeted_mix: float,
                               context_weights: dict,
                               event_weights: dict,
                               rng: np.random.Generator) -> list:
    """Sample train segments with mixed natural + targeted contextual distribution."""
    if not seg_meta:
        return []

    n_total = max(1, int(target_count))
    n_targeted = int(round(n_total * max(0.0, min(1.0, targeted_mix))))
    n_natural  = max(0, n_total - n_targeted)

    # Natural component: unbiased snapshot of real distribution
    natural = []
    if n_natural > 0:
        natural_idx = rng.choice(len(seg_meta), size=min(n_natural, len(seg_meta)),
                                 replace=False)
        natural = [seg_meta[int(i)]["range"] for i in natural_idx]

    # Targeted component: context quotas, then event-weighted draws within context
    targeted = []
    ctx_quota = _allocate_counts(n_targeted, context_weights, CTX_NAMES)
    for ctx in CTX_NAMES:
        need = ctx_quota[ctx]
        if need <= 0:
            continue

        pool_ctx = [m for m in seg_meta if m["context"] == ctx]
        if not pool_ctx:
            pool_ctx = seg_meta

        evt_pools = {e: [m for m in pool_ctx if m["event"] == e] for e in EVT_NAMES}
        evt_keys  = [e for e in EVT_NAMES if evt_pools[e]]
        if not evt_keys:
            evt_keys = EVT_NAMES
            evt_pools = {e: pool_ctx for e in EVT_NAMES}

        evt_quota = _allocate_counts(need, event_weights, evt_keys)
        for evt in evt_keys:
            n_evt = evt_quota.get(evt, 0)
            if n_evt <= 0:
                continue
            choices = evt_pools[evt]
            idxs = rng.integers(0, len(choices), size=n_evt)
            targeted.extend(choices[int(i)]["range"] for i in idxs)

    # Backfill (can happen due empty pools with strict quotas)
    if len(targeted) < n_targeted:
        need = n_targeted - len(targeted)
        idxs = rng.integers(0, len(seg_meta), size=need)
        targeted.extend(seg_meta[int(i)]["range"] for i in idxs)

    combined = natural + targeted[:n_targeted]
    if not combined:
        combined = [m["range"] for m in seg_meta]
    rng.shuffle(combined)
    return combined


def sampled_distribution(sampled_ranges: list, seg_lookup: dict) -> tuple:
    ctx = Counter()
    evt = Counter()
    for r in sampled_ranges:
        c, e = seg_lookup.get(r, (CTX_TRANSITION, EVT_MOVE))
        ctx[c] += 1
        evt[e] += 1
    return ctx, evt


def sample_labels_for_pos_weights(seg_ranges: list, y: np.ndarray,
                                  max_frames: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """Collect up to max_frames labels from shuffled segment ranges."""
    if not seg_ranges:
        return y[:1]

    order = rng.permutation(len(seg_ranges))
    chunks = []
    n = 0
    for i in order:
        s, e = seg_ranges[int(i)]
        if e <= s:
            continue
        seg_y = y[s:e]
        take = min(len(seg_y), max_frames - n)
        if take <= 0:
            break
        chunks.append(seg_y[:take])
        n += take
        if n >= max_frames:
            break
    if not chunks:
        s, e = seg_ranges[0]
        return y[s:min(e, s + 1)]
    return np.concatenate(chunks, axis=0)


def iter_tbptt_batched(X_mm: np.ndarray, y: np.ndarray, seg_ranges: list,
                       T: int, B: int, rng: np.random.Generator,
                       device: torch.device, norm: tuple = None):
    """Stateful TBPTT iterator with B parallel segment streams.

    Maintains B active segments processed in shuffled order.  When a segment
    is exhausted, the next one replaces it and its h/c slot is reset.

    Yields: (X [B,T,F], y_b [B,T,L], reset_mask np.ndarray[B], live_mask np.ndarray[B])
      reset_mask[i] — stream i began a new segment this step; caller zeros h[:,i,:]
      live_mask[i]  — stream i has valid data; exclude dead streams from loss
    """
    order = rng.permutation(len(seg_ranges)).tolist()
    queue = [seg_ranges[i] for i in order]

    pos  = np.zeros(B, dtype=np.int64)
    end_ = np.zeros(B, dtype=np.int64)
    live = np.zeros(B, dtype=bool)

    def _init(i):
        while queue:
            s, e = queue.pop(0)
            if e - s >= T:
                pos[i], end_[i], live[i] = s, e, True
                return
        live[i] = False

    for i in range(B):
        _init(i)

    F = X_mm.shape[1]
    L = y.shape[1]

    while live.any():
        X_buf = np.zeros((B, T, F), dtype=np.float32)
        y_buf = np.zeros((B, T, L), dtype=np.float32)
        reset = np.zeros(B, dtype=bool)

        for i in range(B):
            if not live[i]:
                continue
            if pos[i] + T > end_[i]:
                reset[i] = True
                _init(i)
                if not live[i]:
                    continue
            X_buf[i] = X_mm[pos[i]:pos[i] + T]
            y_buf[i] = y[pos[i]:pos[i] + T]
            pos[i]  += T

        if not live.any():
            break

        X_t = torch.from_numpy(X_buf.copy()).to(device)
        if norm is not None:
            X_t = (X_t - norm[0]) / norm[1]

        yield (X_t,
               torch.from_numpy(y_buf.copy()).to(device),
               reset,
               live.copy())


# --- Data iteration ----------------------------------------------------------

def iter_sequences(X_mm: np.ndarray, y: np.ndarray, seq_starts: np.ndarray,
                   T: int, batch_size: int, shuffle: bool,
                   rng: np.random.Generator, device: torch.device,
                   norm: tuple = None):
    """Yield (X_batch [B,T,F], y_batch [B,T,L]) sequence tensor pairs.

    Sequences are shuffled when shuffle=True; within each mini-batch they are
    sorted by start index for sequential SSD reads.
    If norm=(mean, std) tensors are provided, X is z-score normalized before yielding.
    """
    order = rng.permutation(len(seq_starts)) if shuffle else np.arange(len(seq_starts))

    for b in range(0, len(order), batch_size):
        batch_order  = order[b:b + batch_size]
        batch_starts = seq_starts[batch_order]
        sorted_order = np.argsort(batch_starts)          # sort for sequential reads
        batch_starts = batch_starts[sorted_order]

        # Load B×T frames — each sequence is a contiguous slice
        X_batch = np.stack([X_mm[s:s + T] for s in batch_starts])   # [B, T, FEAT]
        y_batch = np.stack([y[s:s + T]    for s in batch_starts])    # [B, T, LABEL]

        X_t = torch.from_numpy(X_batch.copy()).to(device)
        if norm is not None:
            X_t = (X_t - norm[0]) / norm[1]

        yield X_t, torch.from_numpy(y_batch.copy()).to(device)


# --- Evaluation --------------------------------------------------------------

def evaluate(X_mm: np.ndarray, y: np.ndarray, seq_starts: np.ndarray,
             T: int, model: nn.Module, pos_weights: torch.Tensor,
             batch_size: int, device: torch.device,
             norm: tuple = None) -> dict:
    model.eval()
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    ce_fn  = nn.CrossEntropyLoss()

    rng = np.random.default_rng(0)
    total_btn_loss = total_stk_loss = n_batches = 0

    all_btn_pred, all_btn_true = [], []
    all_stk_pred, all_stk_true = [], []

    with torch.no_grad():
        for X_b, y_b in iter_sequences(X_mm, y, seq_starts, T, batch_size,
                                        shuffle=False, rng=rng, device=device,
                                        norm=norm):
            # X_b: [B, T, FEAT],  y_b: [B, T, LABEL]
            btn_true_b = y_b[:, :, :BUTTON_DIM]       # [B, T, 6] float
            stk_true_b = y_b[:, :, BUTTON_DIM:]       # [B, T, 4] float
            stk_bins_b = float_to_bin(stk_true_b)     # [B, T, 4] int64

            btn_logits, stk_logit_list, _, _, _ = model(
                X_b, btn_targets=btn_true_b, stk_targets=stk_bins_b)

            BT = X_b.shape[0] * T
            btn_loss = bce_fn(btn_logits.reshape(BT, BUTTON_DIM),
                              btn_true_b.reshape(BT, BUTTON_DIM))
            stk_loss = sum(
                ce_fn(stk_logit_list[i].reshape(-1, STICK_BINS),
                      stk_bins_b[:, :, i].reshape(-1))
                for i in range(STICK_DIM)
            ) / STICK_DIM

            total_btn_loss += btn_loss.item()
            total_stk_loss += stk_loss.item()
            n_batches += 1

            all_btn_pred.append(
                (torch.sigmoid(btn_logits) > 0.5).reshape(BT, BUTTON_DIM).cpu().numpy())
            all_btn_true.append(btn_true_b.reshape(BT, BUTTON_DIM).cpu().numpy())

            # Stick predictions: argmax over bins → normalized float for metrics
            stk_bins_pred = torch.stack(
                [l.argmax(dim=-1) for l in stk_logit_list], dim=-1)  # [B, T, 4]
            stk_pred_f = bin_to_float(stk_bins_pred).reshape(BT, STICK_DIM).cpu().numpy()
            all_stk_pred.append(stk_pred_f)
            all_stk_true.append(stk_true_b.reshape(BT, STICK_DIM).cpu().numpy())

    if n_batches == 0:
        return {
            "loss": 0.0, "bce": 0.0, "huber": 0.0,
            "f1": {k: 0.0 for k in BUTTON_NAMES},
            "stick_err_deg": 0.0, "idle_acc": 1.0,
        }

    avg_bce    = total_btn_loss / n_batches
    avg_hub    = total_stk_loss / n_batches
    total_loss = avg_bce + avg_hub

    btn_pred = np.concatenate(all_btn_pred)
    btn_true = np.concatenate(all_btn_true)
    stk_pred = np.concatenate(all_stk_pred)
    stk_true = np.concatenate(all_stk_true)

    f1 = {}
    for i, name in enumerate(BUTTON_NAMES):
        tp = ((btn_pred[:, i] == 1) & (btn_true[:, i] == 1)).sum()
        fp = ((btn_pred[:, i] == 1) & (btn_true[:, i] == 0)).sum()
        fn = ((btn_pred[:, i] == 0) & (btn_true[:, i] == 1)).sum()
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1[name] = 2 * p * r / max(p + r, 1e-8)

    pa  = np.arctan2(stk_pred[:, 1], stk_pred[:, 0])
    ta  = np.arctan2(stk_true[:, 1], stk_true[:, 0])
    d   = np.abs(pa - ta)
    d   = np.where(d > math.pi, 2 * math.pi - d, d)
    stick_err_deg = float(np.degrees(d.mean()))

    idle_mask = np.all(btn_true < 0.5, axis=1)
    idle_acc  = float(np.all(btn_pred[idle_mask] == 0, axis=1).mean()) \
                if idle_mask.any() else 1.0

    return {
        "loss": total_loss, "bce": avg_bce, "huber": avg_hub,
        "f1": f1, "stick_err_deg": stick_err_deg, "idle_acc": idle_acc,
    }


def composite_score(m: dict) -> float:
    return sum(m["f1"].values()) / len(m["f1"])


def print_eval(tag: str, m: dict) -> None:
    f1_str = "  ".join(f"{k}={v:.3f}" for k, v in m["f1"].items())
    print(f"  [{tag}] loss={m['loss']:.4f}  btn_bce={m['bce']:.4f}  stk_ce={m['huber']:.4f}")
    print(f"         F1: {f1_str}")
    print(f"         stick_err={m['stick_err_deg']:.1f}deg  idle_acc={m['idle_acc']:.3f}")


def print_targets(m: dict) -> None:
    print("\n  Target check:")
    all_pass = True
    for name, target in F1_TARGETS.items():
        got  = m["f1"][name]
        ok   = got >= target
        mark = "PASS" if ok else "FAIL"
        print(f"    [{mark}] {name} F1  {got:.3f}  (target >= {target})")
        all_pass = all_pass and ok

    ok = m["stick_err_deg"] <= STICK_ERR_TARGET
    print(f"    [{'PASS' if ok else 'FAIL'}] stick_err  {m['stick_err_deg']:.1f}deg"
          f"  (target <= {STICK_ERR_TARGET}deg)")
    all_pass = all_pass and ok

    ok = m["idle_acc"] >= IDLE_ACC_TARGET
    print(f"    [{'PASS' if ok else 'FAIL'}] idle_acc  {m['idle_acc']:.3f}"
          f"  (target >= {IDLE_ACC_TARGET})")
    all_pass = all_pass and ok

    print(f"\n  Overall: {'ALL TARGETS MET' if all_pass else 'some targets missed'}")


# --- Main --------------------------------------------------------------------

def train(x_path: str, y_path: str, ms_path: str, seg_path: str, out_dir: str,
          epochs: int, batch_size: int, lr: float, seed: int,
          neutral_keep: float,
          sampler_mode: str,
          targeted_mix: float,
          ctx_offense: float, ctx_defense: float, ctx_transition: float,
          evt_shot: float, evt_pass: float, evt_def_hit: float,
          evt_item: float, evt_move: float,
          eval_every: int = 1) -> None:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "training_history.csv"
    csv_cols = (["epoch", "train_loss", "train_bce", "val_loss", "val_bce", "gap",
                 "val_composite"]
                + [f"val_f1_{n}" for n in BUTTON_NAMES]
                + ["val_stick_err_deg", "val_idle_acc"])

    def _fmt(v):
        if v is None:
            return ""
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    csv_file = open(csv_path, "w", buffering=1)
    csv_file.write(",".join(csv_cols) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(seed)

    # Load y, match_starts, seg (all fit in RAM)
    print("\nLoading y, match_starts, seg...", flush=True)
    y   = np.load(y_path).astype(np.float32)
    ms  = np.load(ms_path)
    seg = np.load(seg_path).astype(np.int32)
    N   = y.shape[0]
    print(f"  Frames: {N:,}  Matches: {len(ms):,}  "
          f"Segments: {int(seg.max()) if len(seg) else 0:,}", flush=True)

    # Build all valid sequence start indices from segment structure
    all_seq_starts = build_seq_indices(seg, SEQ_LEN)
    print(f"  Valid sequences (T={SEQ_LEN}, non-overlapping): {len(all_seq_starts):,}",
          flush=True)

    # Temporal split
    train_seq, val_seq, test_seq, train_end = temporal_split_seqs(
        all_seq_starts, ms, N, SEQ_LEN)

    print(f"\nSplit (temporal, by match order):")
    print(f"  Train: {len(train_seq):>8,} sequences")
    print(f"  Val:   {len(val_seq):>8,} sequences")
    print(f"  Test:  {len(test_seq):>8,} sequences")

    # Open X memmap
    print(f"\nOpening X memmap: {x_path}", flush=True)
    X_mm = np.lib.format.open_memmap(x_path, mode="r")

    all_seg_ranges = build_seg_ranges(seg, N)
    train_seg_base = [(s, e) for s, e in all_seg_ranges if e <= train_end]

    context_weights = {
        CTX_OFFENSE: ctx_offense,
        CTX_DEFENSE: ctx_defense,
        CTX_TRANSITION: ctx_transition,
    }
    event_weights = {
        EVT_SHOT: evt_shot,
        EVT_PASS: evt_pass,
        EVT_DEF_HIT: evt_def_hit,
        EVT_ITEM: evt_item,
        EVT_MOVE: evt_move,
    }

    seg_lookup = {}
    train_seg_meta = []
    if sampler_mode == "legacy":
        # Legacy sequence-level oversampling (retained for A/B comparison)
        train_seq_s = oversample_seqs(train_seq, y, SEQ_LEN,
                                      neutral_keep=neutral_keep, seed=seed)
        action_n  = int(np.sum([
            np.any(y[s:s + SEQ_LEN, :6] > 0.5) for s in train_seq]))
        neutral_n = len(train_seq) - action_n
        print(f"\nLegacy oversampling (neutral_keep={neutral_keep:.0%}):")
        print(f"  Action sequences:        {action_n:>8,}")
        print(f"  Neutral sequences (kept):{int(neutral_n * neutral_keep):>8,}")
        print(f"  Training set after OS:   {len(train_seq_s):>8,} sequences")

        train_seg_ranges_base = oversample_segments(
            train_seg_base, y, neutral_keep=neutral_keep, seed=seed)
        seg_lookup = {(s, e): (CTX_TRANSITION, EVT_MOVE) for s, e in train_seg_ranges_base}

        # Gather labels for pos_weight computation (sample up to 500K frames)
        print()
        sample_starts = train_seq_s[:min(len(train_seq_s), 500_000 // SEQ_LEN)]
        if len(sample_starts):
            y_sample = np.concatenate([y[s:s + SEQ_LEN] for s in sample_starts])
        else:
            y_sample = y[:SEQ_LEN]
    else:
        print("\nContextual sampler configuration:")
        print(f"  targeted_mix={targeted_mix:.0%}  natural_mix={1.0-targeted_mix:.0%}")
        print("  context weights:"
              f" offense={ctx_offense:.2f} defense={ctx_defense:.2f} transition={ctx_transition:.2f}")
        print("  event weights:"
              f" shot={evt_shot:.2f} pass={evt_pass:.2f} def_hit={evt_def_hit:.2f}"
              f" item={evt_item:.2f} move={evt_move:.2f}")

        train_seg_meta = build_segment_metadata(train_seg_base, X_mm, y)
        print_segment_distribution("Base train", train_seg_meta)
        seg_lookup = {m["range"]: (m["context"], m["event"]) for m in train_seg_meta}

        rng_init = np.random.default_rng(seed)
        train_seg_ranges_base = sample_contextual_segments(
            train_seg_meta, len(train_seg_meta), targeted_mix,
            context_weights, event_weights, rng_init
        )
        ctx_c, evt_c = sampled_distribution(train_seg_ranges_base, seg_lookup)
        total = max(len(train_seg_ranges_base), 1)
        print("\nInitial sampled segment mix:")
        print("  Context:")
        for c in CTX_NAMES:
            n = ctx_c.get(c, 0)
            print(f"    {c:<10s} {n:>8,}  ({100*n/total:5.1f}%)")
        print("  Events:")
        for e_name in EVT_NAMES:
            n = evt_c.get(e_name, 0)
            print(f"    {e_name:<10s} {n:>8,}  ({100*n/total:5.1f}%)")

        # Gather labels for pos_weight computation from sampled train segments
        print()
        y_sample = sample_labels_for_pos_weights(
            train_seg_ranges_base, y, max_frames=500_000, rng=np.random.default_rng(seed + 1)
        )

    pos_weights, _ = compute_pos_weights(y_sample)
    del y_sample

    n_chunks_total = sum(max(0, (e - s) // SEQ_LEN) for s, e in train_seg_ranges_base)
    print(f"\nStateful TBPTT segments: {len(train_seg_ranges_base):,}"
          f"  (~{n_chunks_total:,} chunks of T={SEQ_LEN})")

    # Normalization stats — computed from training portion only (no val/test leakage)
    print(f"\nComputing normalization stats over {train_end:,} training frames...",
          flush=True)
    norm_mean, norm_std = compute_norm_stats(X_mm, train_end)
    np.savez(out / "norm_stats.npz", mean=norm_mean, std=norm_std)
    print(f"  Saved: {out / 'norm_stats.npz'}", flush=True)
    norm_mean_t = torch.from_numpy(norm_mean).to(device)
    norm_std_t  = torch.from_numpy(norm_std).to(device)
    norm = (norm_mean_t, norm_std_t)

    # Model
    model    = StrikersLSTM(feature_dim=FEATURE_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: StrikersLSTM — {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    ce_fn  = nn.CrossEntropyLoss()

    n_batches_est = max(1, n_chunks_total // batch_size)
    print(f"\nStarting training: {epochs} epochs × ~{n_batches_est:,} batches/epoch")
    print("-" * 70)

    best_composite = -float("inf")
    best_epoch     = -1

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_bce = epoch_hub = 0.0
        n_seen = 0
        t0 = time.time()

        rng = np.random.default_rng(seed + epoch)
        if sampler_mode == "contextual":
            train_seg_ranges = sample_contextual_segments(
                train_seg_meta, len(train_seg_meta), targeted_mix,
                context_weights, event_weights, rng
            )
            if epoch == 1 or epoch % 5 == 0:
                ctx_c, evt_c = sampled_distribution(train_seg_ranges, seg_lookup)
                total = max(len(train_seg_ranges), 1)
                ctx_str = "  ".join(f"{c}={100*ctx_c.get(c,0)/total:.1f}%"
                                    for c in CTX_NAMES)
                evt_str = "  ".join(f"{e}={100*evt_c.get(e,0)/total:.1f}%"
                                    for e in EVT_NAMES)
                print(f"\n  Sampler mix (epoch {epoch}):")
                print(f"    Context: {ctx_str}")
                print(f"    Events:  {evt_str}")
        else:
            train_seg_ranges = train_seg_ranges_base

        # Stateful TBPTT: h/c carried across chunks within each segment,
        # reset only when a stream starts a new segment.
        h = torch.zeros(StrikersLSTM.LAYERS, batch_size, StrikersLSTM.HIDDEN,
                        device=device)
        c = torch.zeros(StrikersLSTM.LAYERS, batch_size, StrikersLSTM.HIDDEN,
                        device=device)

        batch_i = -1
        for batch_i, (X_b, y_b, reset_mask, live_mask) in enumerate(
            iter_tbptt_batched(X_mm, y, train_seg_ranges, SEQ_LEN, batch_size,
                               rng, device, norm=norm)
        ):
            # Zero h/c for streams that started a new segment this step
            reset_idx = np.where(reset_mask)[0]
            if len(reset_idx):
                h[:, reset_idx, :] = 0.0
                c[:, reset_idx, :] = 0.0

            # Pass targets to LSTM for AR head teacher forcing
            btn_logits, stk_logit_list, _, h_out, c_out = model(
                X_b, h, c,
                btn_targets=y_b[:, :, :BUTTON_DIM],
                stk_targets=float_to_bin(y_b[:, :, BUTTON_DIM:]))

            # Detach state — BPTT gradient flows within the T-frame chunk only,
            # but the hidden state VALUES carry context across chunks.
            h = h_out.detach()
            c = c_out.detach()

            # Compute loss only over live (active) streams
            n_live = int(live_mask.sum())
            if n_live == 0:
                continue

            live_t   = torch.from_numpy(live_mask).to(device)
            BT_live  = n_live * SEQ_LEN
            btn_true = y_b[live_t, :, :BUTTON_DIM]
            stk_bins = float_to_bin(y_b[live_t, :, BUTTON_DIM:])   # [n_live, T, 4] int64

            bce    = bce_fn(btn_logits[live_t].reshape(BT_live, BUTTON_DIM),
                            btn_true.reshape(BT_live, BUTTON_DIM))
            stk_ce = sum(
                ce_fn(stk_logit_list[i][live_t].reshape(-1, STICK_BINS),
                      stk_bins[:, :, i].reshape(-1))
                for i in range(STICK_DIM)
            ) / STICK_DIM
            loss = bce + stk_ce

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch} batch {batch_i}: "
                    f"bce={bce.item():.4f} stk_ce={stk_ce.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_bce += bce.item()
            epoch_hub += stk_ce.item()
            n_seen    += n_live

            if (batch_i + 1) % 200 == 0:
                pct = (batch_i + 1) / n_batches_est * 100
                print(f"  epoch {epoch}  [{pct:5.1f}%  {n_seen:,} seqs]  "
                      f"btn_bce={epoch_bce/(batch_i+1):.4f}  "
                      f"stk_ce={epoch_hub/(batch_i+1):.4f}",
                      flush=True)

        n_batches  = max(1, batch_i + 1)
        train_bce  = epoch_bce / n_batches
        train_hub  = epoch_hub / n_batches
        train_loss = train_bce + train_hub
        elapsed    = time.time() - t0

        print(f"\nEpoch {epoch:3d}/{epochs}  ({elapsed:.0f}s)")
        print(f"  [train] loss={train_loss:.4f}  btn_bce={train_bce:.4f}  stk_ce={train_hub:.4f}")

        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        if do_eval:
            val_m = evaluate(X_mm, y, val_seq, SEQ_LEN, model, pos_weights,
                             batch_size, device, norm=norm)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            scheduler.step(val_m["loss"])

            score   = composite_score(val_m)
            is_best = score > best_composite
            if is_best:
                best_composite = score
                best_epoch     = epoch
                torch.save(model.state_dict(), out / "best_model.pt")

            gap = val_m["loss"] - train_loss
            print_eval("val", val_m)
            gap_flag = "  << overfitting" if gap > 0.05 else ""
            best_tag = " <- best" if is_best else ""
            print(f"  gap(val-train)={gap:+.4f}{gap_flag}  "
                  f"best composite: {best_composite:.4f} (epoch {best_epoch}){best_tag}")

            f1 = val_m["f1"]
            csv_row = ([epoch, train_loss, train_bce,
                        val_m["loss"], val_m["bce"], gap, score]
                       + [f1.get(n) for n in BUTTON_NAMES]
                       + [val_m["stick_err_deg"], val_m["idle_acc"]])
        else:
            print(f"  [val skipped — eval_every={eval_every}]")
            csv_row = ([epoch, train_loss, train_bce,
                        None, None, None, None]
                       + [None] * len(BUTTON_NAMES)
                       + [None, None])

        csv_file.write(",".join(_fmt(v) for v in csv_row) + "\n")
        print("-" * 70)

    torch.save(model.state_dict(), out / "final_model.pt")
    csv_file.close()
    print(f"\nSaved: {out / 'best_model.pt'}  (epoch {best_epoch})")
    print(f"Saved: {out / 'final_model.pt'}")
    print(f"Saved: {csv_path}")

    print(f"\n{'=' * 70}")
    print("TEST SET EVALUATION (best model)")
    print("=" * 70)
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    test_m = evaluate(X_mm, y, test_seq, SEQ_LEN, model, pos_weights,
                      batch_size, device, norm=norm)
    print_eval("test", test_m)
    print_targets(test_m)


# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM behavioral cloning model on Strikers CITF dataset"
    )
    parser.add_argument("x_path",   help="Path to *_X.npy (memory-mapped features)")
    parser.add_argument("y_path",   help="Path to *_y.npy (labels)")
    parser.add_argument("ms_path",  help="Path to *_ms.npy (match start indices)")
    parser.add_argument("seg_path", help="Path to *_seg.npy (segment IDs)")
    parser.add_argument("out_dir",  help="Output directory for saved models and logs")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch",        type=int,   default=64,
                        help="Sequences per batch (default 64 → 64×64=4096 frames/step)")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--neutral-keep", type=float, default=0.20,
                        help="Legacy sampler only: fraction of neutral sequences/segments kept")
    parser.add_argument("--sampler",      choices=["contextual", "legacy"],
                        default="contextual",
                        help="Sampling strategy for training segments (default contextual)")
    parser.add_argument("--targeted-mix", type=float, default=0.60,
                        help="Contextual sampler: fraction of targeted samples per epoch")
    parser.add_argument("--ctx-offense",  type=float, default=0.40,
                        help="Contextual sampler weight for offense segments")
    parser.add_argument("--ctx-defense",  type=float, default=0.40,
                        help="Contextual sampler weight for defense segments")
    parser.add_argument("--ctx-transition", type=float, default=0.20,
                        help="Contextual sampler weight for transition segments")
    parser.add_argument("--evt-shot",     type=float, default=0.25,
                        help="Contextual sampler event weight: shot")
    parser.add_argument("--evt-pass",     type=float, default=0.20,
                        help="Contextual sampler event weight: pass")
    parser.add_argument("--evt-def-hit",  type=float, default=0.20,
                        help="Contextual sampler event weight: defensive hit/deke")
    parser.add_argument("--evt-item",     type=float, default=0.10,
                        help="Contextual sampler event weight: item usage")
    parser.add_argument("--evt-move",     type=float, default=0.25,
                        help="Contextual sampler event weight: movement/positioning")
    parser.add_argument("--eval-every",   type=int,   default=1,
                        help="Run validation every N epochs (default 1)")
    args = parser.parse_args()

    train(args.x_path, args.y_path, args.ms_path, args.seg_path, args.out_dir,
          args.epochs, args.batch, args.lr, args.seed, args.neutral_keep,
          args.sampler, args.targeted_mix,
          args.ctx_offense, args.ctx_defense, args.ctx_transition,
          args.evt_shot, args.evt_pass, args.evt_def_hit, args.evt_item, args.evt_move,
          eval_every=args.eval_every)
