#!/usr/bin/env python3
"""train_transformer.py — Behavioral cloning transformer for Citrus Strikers.

Architecture:
  EntityEncoder (2L, 256d, 4h):           per-frame spatial attention over 16 entity tokens
  CausalTemporalTransformer (3L, 512d, 4h): causal attention over T=64 frame window
  ARControllerHead:                        autoregressive buttons→stick-bins output
  Total: ~7.55M parameters

Key differences from train.py (LSTM):
  - No TBPTT: sequences are independent; PyTorch DataLoader handles batching
  - Both X and y opened as numpy memmaps so datasets exceeding RAM are supported
  - Flat 440-float vectors sliced into 16 entity tokens at batch time (on GPU)
  - No LSTM h/c state — KV cache is ONNX inference-only (see forward_infer)
  - AMP (bf16/fp16) training for GPU throughput
  - AdamW + cosine LR with linear warmup

Requires PyTorch >= 2.0 (F.scaled_dot_product_attention with is_causal=True).

Usage:
    python train_transformer.py <X.npy> <y.npy> <ms.npy> <seg.npy> <out_dir>
    python train_transformer.py <X.npy> <y.npy> <ms.npy> <seg.npy> <out_dir> \\
        --epochs 20 --batch 256 --amp --workers 4

Entity token layout (16 tokens, ENTITY_RAW_DIM=64):
    Derived from build_dataset.py feature layout — must stay in sync.
    Token  Slice      Raw dim  Entity type
     0     [0:19]       19     ball: pos(3)+vel(3)+charge(1)+perfect_pass(1)+owner_oh(11)
     1     [19:67]      48     self character
     2     [67:103]     36     friendly striker 0
     3     [103:139]    36     friendly striker 1
     4     [139:175]    36     friendly striker 2
     5     [175:205]    30     friendly goalie
     6     [205:241]    36     enemy striker 0
     7     [241:277]    36     enemy striker 1
     8     [277:313]    36     enemy striker 2
     9     [313:349]    36     enemy striker 3
    10     [349:379]    30     enemy goalie
    11     [379:390]    11     own inventory slot 0
    12     [390:401]    11     own inventory slot 1
    13     [401:412]    11     enemy inventory slot 0
    14     [412:423]    11     enemy inventory slot 1
    15     [423:442]    19     context: tactical(5)+score/time(2)+possession(2)+prev_action(10)

ONNX inference interface (export_onnx_transformer.py — separate script):
    inputs:  entities [1,16,64], kv_cache [3,2,1,63,512]
    outputs: btn_probs [1,6], stk_bins [1,4], kv_cache_out [3,2,1,63,512]
    AIController.cpp: replace h/c with kv_cache; call flat_to_entities C++ port.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN     = 64      # frames per training sequence (~1 s at 60fps)
FEATURE_DIM = 442     # flat features per frame (must match build_dataset.py)
BUTTON_DIM      = 6       # labels 0-5: A, B, X, Y, L, R
STICK_DIM       = 4       # labels 6-9: stick_x, stick_y, cstick_x, cstick_y
LABEL_DIM       = BUTTON_DIM + STICK_DIM
STICK_BINS      = 21      # discrete stick bins per axis [-1,1] → 0..20
PREV_ACTION_DIM = 10      # last 10 features are prev-frame labels (zeroed at train+infer)

ENTITY_RAW_DIM  = 64   # zero-padded token width fed into EntityEncoder
N_ENTITY_TYPES  = 9    # ball / self / fr_str / fr_gk / en_str / en_gk / inv_own / inv_enemy / ctx
N_ENTITIES      = 16

ENTITY_DIM      = 256  # EntityEncoder hidden dim
ENTITY_LAYERS   = 2
ENTITY_HEADS    = 4

TEMPORAL_DIM    = 512  # TemporalTransformer hidden dim
TEMPORAL_LAYERS = 3
TEMPORAL_HEADS  = 4
FF_MULT         = 2

BUTTON_NAMES = ["A", "B", "X", "Y", "L", "R"]
STICK_NAMES  = ["stick_x", "stick_y", "cstick_x", "cstick_y"]

F1_TARGETS       = {"A": 0.45, "B": 0.55, "X": 0.40, "Y": 0.35, "L": 0.30, "R": 0.50}
STICK_ERR_TARGET = 25.0
IDLE_ACC_TARGET  = 0.90

# ── Entity token layout ───────────────────────────────────────────────────────
# (slice_start, slice_end_exclusive, entity_type_id)
# Slice offsets derived from build_dataset.py feature layout comment block.
# If FEATURE_DIM changes (e.g. remove game_time, add attacking bools → 441),
# update the slices here to match and set FEATURE_DIM accordingly.

_ENTITY_DEFS: List[Tuple[int, int, int]] = [
    (  0,  19, 0),  # ball: pos×3 vel×3 charge perfect_pass + owner_oh×11
    ( 19,  67, 1),  # self: pos_delta×3 state_oh×30 heading×2 goal×3 effect×5 spd×3 timer carrier
    ( 67, 103, 2),  # friendly striker 0: pos_delta×3 state_oh×30 heading×2 carrier
    (103, 139, 2),  # friendly striker 1
    (139, 175, 2),  # friendly striker 2
    (175, 205, 3),  # friendly goalie: pos_delta×3 goalie_state_oh×27
    (205, 241, 4),  # enemy striker 0
    (241, 277, 4),  # enemy striker 1
    (277, 313, 4),  # enemy striker 2
    (313, 349, 4),  # enemy striker 3
    (349, 379, 5),  # enemy goalie
    (379, 390, 6),  # own inventory 0: powerup_oh×10 charge
    (390, 401, 6),  # own inventory 1
    (401, 412, 7),  # enemy inventory 0
    (412, 423, 7),  # enemy inventory 1
    (423, 442, 8),  # context: tactical×5 score_diff time possession×2 prev_action×10
]

assert len(_ENTITY_DEFS) == N_ENTITIES, "N_ENTITIES mismatch"
assert all(0 <= s and e <= FEATURE_DIM and (e - s) <= ENTITY_RAW_DIM
           for s, e, _ in _ENTITY_DEFS), "Entity slice out of bounds or exceeds ENTITY_RAW_DIM"
assert sum(e - s for s, e, _ in _ENTITY_DEFS) == FEATURE_DIM, (
    f"Entity slices do not cover all {FEATURE_DIM} features")

_ENTITY_TYPE_IDS = [t for _, _, t in _ENTITY_DEFS]   # [16] — used in EntityEncoder


# ── Discretisation helpers ─────────────────────────────────────────────────────

def float_to_bin(v: torch.Tensor, bins: int = STICK_BINS) -> torch.Tensor:
    """Continuous stick value [-1,1] → discrete bin index [0, bins-1]."""
    return ((v.clamp(-1.0, 1.0) + 1.0) / 2.0 * (bins - 1)).round().long()


def bin_to_float(b: torch.Tensor, bins: int = STICK_BINS) -> torch.Tensor:
    """Bin index [0, bins-1] → float in [-1, 1]."""
    return b.float() / (bins - 1) * 2.0 - 1.0


# ── Entity tokenisation ────────────────────────────────────────────────────────

def flat_to_entities(x: torch.Tensor) -> torch.Tensor:
    """Slice flat feature vector into zero-padded entity token matrix.

    x: [..., FEATURE_DIM]
    returns: [..., N_ENTITIES, ENTITY_RAW_DIM]
    """
    *leading, _ = x.shape
    out = x.new_zeros(*leading, N_ENTITIES, ENTITY_RAW_DIM)
    for i, (s, e, _) in enumerate(_ENTITY_DEFS):
        out[..., i, : e - s] = x[..., s:e]
    return out


# ── Model ──────────────────────────────────────────────────────────────────────

class _FFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * FF_MULT),
            nn.GELU(),
            nn.Linear(dim * FF_MULT, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EntityTransformerLayer(nn.Module):
    """Non-causal full attention within a single frame's entity tokens."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn   = _FFN(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 for attention to avoid bf16 overflow (softmax of large
        # QK^T values overflows bf16 → NaN). Cast back to original dtype after.
        orig_dtype = x.dtype
        x32 = x.float()
        a, _ = self.attn(x32, x32, x32)
        a = a.to(orig_dtype)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class EntityEncoder(nn.Module):
    """Per-frame spatial encoder: entity tokens → single frame embedding.

    Input:  [B, N_ENTITIES, ENTITY_RAW_DIM]
    Output: [B, TEMPORAL_DIM]

    A learned entity-type embedding is added after projection so the model
    learns to distinguish ball / strikers / goalies / inventory / context
    without relying solely on their positional order.
    """

    def __init__(self):
        super().__init__()
        self.proj       = nn.Linear(ENTITY_RAW_DIM, ENTITY_DIM)
        self.type_embed = nn.Embedding(N_ENTITY_TYPES, ENTITY_DIM)
        self.layers     = nn.ModuleList([
            EntityTransformerLayer(ENTITY_DIM, ENTITY_HEADS)
            for _ in range(ENTITY_LAYERS)
        ])
        self.out = nn.Linear(ENTITY_DIM, TEMPORAL_DIM)

        type_ids = torch.tensor(_ENTITY_TYPE_IDS, dtype=torch.long)
        self.register_buffer("type_ids", type_ids)  # [N_ENTITIES]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, raw_dim]
        x = self.proj(x) + self.type_embed(self.type_ids)  # [B, N, entity_dim]
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)    # mean pool over entity tokens → [B, entity_dim]
        return self.out(x)   # [B, temporal_dim]


class TemporalLayer(nn.Module):
    """Transformer layer with two forward paths that share weights:

    forward_causal(x [B,T,D]) — training: full sequence, causal mask, FlashAttention
    forward_cached(x [B,1,D], k_cache, v_cache) — ONNX inference: KV cache eviction
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads    = heads
        self.head_dim = dim // heads
        self.dim      = dim

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn      = _FFN(dim)
        self.norm1    = nn.LayerNorm(dim)
        self.norm2    = nn.LayerNorm(dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.heads, self.head_dim).transpose(1, 2)  # [B,H,S,D/H]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.dim)

    def forward_causal(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]. Uses F.scaled_dot_product_attention (FlashAttention)."""
        orig_dtype = x.dtype
        q = self._split_heads(self.q_proj(x)).float()
        k = self._split_heads(self.k_proj(x)).float()
        v = self._split_heads(self.v_proj(x)).float()
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True).to(orig_dtype)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x

    def forward_cached(self,
                       x: torch.Tensor,
                       k_cache: torch.Tensor,
                       v_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incremental inference.

        x:       [B, 1, D]   — new frame
        k_cache: [B, S, D]   — past S frames' keys   (S = SEQ_LEN-1 at steady state)
        v_cache: [B, S, D]
        Returns: out [B, 1, D], k_cache_new [B, S, D], v_cache_new [B, S, D]
        """
        q = self._split_heads(self.q_proj(x))   # [B, H, 1, D/H]
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        k_all = torch.cat([self._split_heads(k_cache), k], dim=2)  # [B, H, S+1, D/H]
        v_all = torch.cat([self._split_heads(v_cache), v], dim=2)

        scale = math.sqrt(self.head_dim)
        attn  = (q @ k_all.transpose(-2, -1)) / scale   # [B, H, 1, S+1]
        attn  = attn.softmax(dim=-1)
        out   = self._merge_heads(attn @ v_all)          # [B, 1, D]
        out   = self.out_proj(out)
        x     = self.norm1(x + out)
        x     = self.norm2(x + self.ffn(x))

        # Evict the oldest cached frame so the cache stays at S entries
        k_full = self._merge_heads(k_all)   # [B, S+1, D]
        v_full = self._merge_heads(v_all)
        return x, k_full[:, 1:, :], v_full[:, 1:, :]


class CausalTemporalTransformer(nn.Module):
    """3-layer causal transformer over the temporal frame sequence.

    Training path: forward(x [B,T,D]) — processes full sequence in parallel.
    Inference path: forward_cached(frame_emb, kv_cache) — per-frame, O(1).
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalLayer(TEMPORAL_DIM, TEMPORAL_HEADS)
            for _ in range(TEMPORAL_LAYERS)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward_causal(x)
        return x

    def forward_cached(self,
                       frame_emb: torch.Tensor,
                       kv_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """frame_emb: [B, D]. kv_cache: [LAYERS, 2, B, S, D]. Returns (out [B,D], kv_out)."""
        x = frame_emb.unsqueeze(1)  # [B, 1, D]
        new_kv = []
        for i, layer in enumerate(self.layers):
            x, k_new, v_new = layer.forward_cached(x, kv_cache[i, 0], kv_cache[i, 1])
            new_kv.append(torch.stack([k_new, v_new], dim=0))  # [2, B, S, D]
        return x.squeeze(1), torch.stack(new_kv, dim=0)        # [B,D], [L,2,B,S,D]


class IndependentControllerHead(nn.Module):
    """Independent controller head — all outputs predicted directly from policy.

    Each button and each stick axis is predicted independently with a single
    linear layer.  No autoregressive conditioning between outputs, so there is
    no teacher-forcing gap at inference time.
    """

    def __init__(self, input_dim: int, stick_bins: int = STICK_BINS):
        super().__init__()
        self.stick_bins = stick_bins
        self.btn_head   = nn.Linear(input_dim, BUTTON_DIM)
        self.stk_heads  = nn.ModuleList([nn.Linear(input_dim, stick_bins) for _ in range(STICK_DIM)])

    def forward_train(self,
                      policy: torch.Tensor,
                      btn_targets: torch.Tensor,
                      stk_targets: torch.Tensor):
        """Training forward (btn_targets/stk_targets unused — kept for API compatibility).

        policy: [B, T, input_dim]
        Returns:
            btn_logits: [B, T, BUTTON_DIM]
            stk_logits: list of STICK_DIM tensors, each [B, T, STICK_BINS]
        """
        btn_logits = self.btn_head(policy)  # [B, T, BUTTON_DIM]
        stk_logits = [self.stk_heads[i](policy) for i in range(STICK_DIM)]
        return btn_logits, stk_logits

    def forward_infer(self, policy_t: torch.Tensor):
        """Inference forward — identical computation to forward_train, no gap.

        policy_t: [B, input_dim]
        Returns:
            btn_probs [B, BUTTON_DIM] — sigmoid probabilities
            stk_bins  [B, STICK_DIM] — argmax bin per axis
        """
        btn_probs = torch.sigmoid(self.btn_head(policy_t))  # [B, BUTTON_DIM]
        stk_bins  = torch.stack(
            [self.stk_heads[i](policy_t).argmax(dim=-1) for i in range(STICK_DIM)],
            dim=-1)  # [B, STICK_DIM]
        return btn_probs, stk_bins


class CitrusTransformerBC(nn.Module):
    """Full transformer behavioral cloning model.

    Training:  forward(x, btn_targets, stk_targets) — teacher-forced
    Inference: forward_infer(entities, kv_cache)     — ONNX-exportable
    """

    def __init__(self):
        super().__init__()
        self.entity_encoder = EntityEncoder()
        self.temporal       = CausalTemporalTransformer()
        self.ctrl_head      = IndependentControllerHead(TEMPORAL_DIM)
        self.value_head     = nn.Linear(TEMPORAL_DIM, 1)  # reserved for future RL/PPO

    def forward(self,
                x: torch.Tensor,
                btn_targets: torch.Tensor,
                stk_targets: torch.Tensor):
        """Training forward.

        x:           [B, T, FEATURE_DIM]
        btn_targets: [B, T, BUTTON_DIM] float
        stk_targets: [B, T, STICK_DIM]  int64
        Returns: btn_logits [B,T,6], stk_logits list[B,T,21], value [B,T,1]
        """
        B, T, _ = x.shape
        entities  = flat_to_entities(x)                           # [B, T, N, raw]
        frame_emb = self.entity_encoder(
            entities.view(B * T, N_ENTITIES, ENTITY_RAW_DIM))    # [B*T, D]
        frame_emb = frame_emb.view(B, T, TEMPORAL_DIM)            # [B, T, D]
        temporal  = self.temporal(frame_emb)                       # [B, T, D]
        value     = self.value_head(temporal)                      # [B, T, 1]
        btn_logits, stk_logits = self.ctrl_head.forward_train(temporal, btn_targets, stk_targets)
        return btn_logits, stk_logits, value

    def forward_infer(self,
                      entities: torch.Tensor,
                      kv_cache: torch.Tensor):
        """ONNX inference path — one frame at a time.

        entities:  [1, N_ENTITIES, ENTITY_RAW_DIM]
        kv_cache:  [TEMPORAL_LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]
        Returns:
            btn_probs    [1, BUTTON_DIM]         — sigmoid probabilities
            stk_bins     [1, STICK_DIM]          — argmax bin indices (int64)
            kv_cache_out [TEMPORAL_LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]
        """
        frame_emb           = self.entity_encoder(entities)          # [1, D]
        out, kv_cache_out   = self.temporal.forward_cached(frame_emb, kv_cache)
        btn_probs, stk_bins = self.ctrl_head.forward_infer(out)
        return btn_probs, stk_bins, kv_cache_out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Dataset ───────────────────────────────────────────────────────────────────

class StrikersDataset(Dataset):
    """Memory-mapped sequence dataset.

    Opens X and y as numpy memmaps inside each DataLoader worker so the full
    dataset never needs to fit in RAM.  norm_mean/std are applied in __getitem__
    so all normalisation happens on CPU before the tensor hits the GPU.
    """

    def __init__(self,
                 x_path: str,
                 y_path: str,
                 seq_starts: np.ndarray,
                 T: int,
                 norm_mean: Optional[torch.Tensor] = None,
                 norm_std:  Optional[torch.Tensor] = None):
        self.x_path    = str(x_path)
        self.y_path    = str(y_path)
        self.starts    = seq_starts
        self.T         = T
        self.norm_mean = norm_mean  # [F] on CPU, or None
        self.norm_std  = norm_std
        # Lazily opened per worker to avoid cross-process descriptor sharing
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def _ensure_open(self) -> None:
        if self._X is None:
            self._X = np.lib.format.open_memmap(self.x_path, mode="r")
            self._y = np.lib.format.open_memmap(self.y_path, mode="r")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        self._ensure_open()
        s  = int(self.starts[i])
        X  = torch.from_numpy(self._X[s : s + self.T].copy()).float()  # [T, F] float32
        y  = torch.from_numpy(self._y[s : s + self.T].copy())  # [T, L] float32
        if self.norm_mean is not None:
            X = (X - self.norm_mean) / self.norm_std
        return X, y


def make_loader(dataset: StrikersDataset, batch_size: int,
                shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


# ── Index & split utilities ────────────────────────────────────────────────────

def build_seq_indices(seg: np.ndarray, T: int) -> np.ndarray:
    """Non-overlapping T-frame windows within each contiguous segment.

    Windows that would cross a segment boundary are excluded.
    Returns sorted array of sequence start frame indices.
    """
    if len(seg) < T:
        return np.array([], dtype=np.int64)
    boundaries = np.where(np.diff(seg) != 0)[0] + 1
    seg_starts = np.concatenate([[0], boundaries])
    seg_ends   = np.concatenate([boundaries, [len(seg)]])
    starts = []
    for ss, se in zip(seg_starts, seg_ends):
        for st in range(int(ss), int(se) - T + 1, T):
            starts.append(st)
    return np.array(starts, dtype=np.int64)


def temporal_split_seqs(seq_starts: np.ndarray,
                         match_starts: np.ndarray,
                         total_frames: int,
                         T: int,
                         train_frac: float = 0.70,
                         val_frac:   float = 0.15):
    """Split sequence indices by temporal match order (70 / 15 / 15)."""
    n   = len(match_starts)
    i_v = int(n * train_frac)
    i_t = int(n * (train_frac + val_frac))
    train_end = int(match_starts[i_v]) if i_v < n else total_frames
    val_end   = int(match_starts[i_t]) if i_t < n else total_frames
    train_idx = seq_starts[seq_starts + T <= train_end]
    val_idx   = seq_starts[(seq_starts >= train_end) & (seq_starts + T <= val_end)]
    test_idx  = seq_starts[seq_starts >= val_end]
    return train_idx, val_idx, test_idx, train_end


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_norm_stats(x_path: str, n_frames: int,
                       chunk: int = 500_000) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std over the first n_frames rows.

    Uses chunked sequential reads so the full file need not be loaded at once.
    Features with std < 1e-6 (constant dims) get std=1 so division is a no-op.
    """
    X   = np.lib.format.open_memmap(x_path, mode="r")
    F   = X.shape[1]
    s1  = np.zeros(F, dtype=np.float64)
    s2  = np.zeros(F, dtype=np.float64)
    pos = 0
    while pos < n_frames:
        end   = min(pos + chunk, n_frames)
        blk   = X[pos:end].astype(np.float64)
        s1   += blk.sum(axis=0)
        s2   += (blk ** 2).sum(axis=0)
        pos   = end
    mean = s1 / n_frames
    var  = np.maximum(s2 / n_frames - mean ** 2, 0.0)
    std  = np.where(np.sqrt(var) < 1e-6, 1.0, np.sqrt(var))
    return mean.astype(np.float32), std.astype(np.float32)


# ── Oversampling ──────────────────────────────────────────────────────────────

def build_action_frame_mask(y_path: str, n_frames: int,
                             chunk: int = 1_000_000) -> np.ndarray:
    """Sequential scan: True for frames with any button press or c-stick deflection.

    Sequential reads avoid the random-access penalty of large memmapped files.
    """
    y    = np.lib.format.open_memmap(y_path, mode="r")
    mask = np.zeros(n_frames, dtype=bool)
    pos  = 0
    while pos < n_frames:
        end  = min(pos + chunk, n_frames)
        blk  = y[pos:end]
        mask[pos:end] = (
            np.any(blk[:, :BUTTON_DIM] > 0.5, axis=1) |
            np.any(np.abs(blk[:, BUTTON_DIM + 2 : BUTTON_DIM + 4]) > 0.15, axis=1)
        )
        pos = end
    return mask


def oversample_seqs(seq_starts: np.ndarray,
                    action_frame_mask: np.ndarray,
                    T: int,
                    neutral_keep: float,
                    seed: int) -> np.ndarray:
    """Sequence-level neutral undersampling using pre-built per-frame action mask.

    Uses a prefix-sum for O(N) vectorised range queries — no per-sequence loops.
    Keep all sequences containing ≥1 action frame; keep neutral_keep fraction of
    fully-neutral sequences.
    """
    # Prefix sum for O(1) range queries: action_in_seq[i] = #action frames in seq i
    cumsum         = np.concatenate([[0], action_frame_mask.cumsum()])
    action_in_seq  = cumsum[seq_starts + T] - cumsum[seq_starts]
    action_mask    = action_in_seq > 0

    action_seqs  = seq_starts[action_mask]
    neutral_seqs = seq_starts[~action_mask]

    rng    = np.random.default_rng(seed)
    n_keep = int(len(neutral_seqs) * neutral_keep)
    neutral_sample = (rng.choice(neutral_seqs, size=n_keep, replace=False)
                      if n_keep > 0 else np.array([], dtype=np.int64))
    return np.sort(np.concatenate([action_seqs, neutral_sample]))


# ── Positive-class weights ─────────────────────────────────────────────────────

def compute_pos_weights(y_path: str, train_end: int,
                         chunk: int = 1_000_000) -> Tuple[torch.Tensor, np.ndarray]:
    """Button press rates and pos_weights from sequential scan of training labels."""
    y    = np.lib.format.open_memmap(y_path, mode="r")
    sums = np.zeros(BUTTON_DIM, dtype=np.float64)
    pos  = 0
    while pos < train_end:
        end   = min(pos + chunk, train_end)
        sums += (y[pos:end, :BUTTON_DIM] > 0.5).sum(axis=0)
        pos   = end
    rates   = sums / train_end
    weights = np.clip((1.0 - rates) / np.maximum(rates, 1e-6), 1.0, 20.0)
    weights[4] = min((1.0 - rates[4]) / max(rates[4], 1e-6), 20.0)  # L: capped at 20 to prevent gradient explosion

    print("Button press rates and pos_weights:")
    for name, r, w in zip(BUTTON_NAMES, rates, weights):
        print(f"  {name:<3s}  rate={r:.4f}  pos_weight={w:.2f}")
    return torch.tensor(weights, dtype=torch.float32), rates


# ── Stick bin class weights ────────────────────────────────────────────────────

def compute_stick_bin_weights(y_path: str, train_end: int,
                               max_weight: float = 10.0,
                               chunk: int = 1_000_000) -> torch.Tensor:
    """Per-axis inverse-frequency class weights for the 21-bin stick CE loss.

    Returns a [STICK_DIM, STICK_BINS] tensor — one weight vector per axis.
    Each axis is weighted independently so that the very different neutral-bin
    frequencies of stick_x/y (~15%) vs cstick_x/y (~80-93%) don't dilute
    each other.

    Weight formula:  w_i = (total / (STICK_BINS * count_i)).clip(1, max_weight)
    """
    AXIS_NAMES = ["stick_x", "stick_y", "cstick_x", "cstick_y"]
    y      = np.lib.format.open_memmap(y_path, mode="r")
    counts = np.zeros((STICK_DIM, STICK_BINS), dtype=np.float64)
    pos    = 0
    while pos < train_end:
        end      = min(pos + chunk, train_end)
        stk_cont = y[pos:end, BUTTON_DIM:].astype(np.float32)
        stk_bins = np.round(
            (stk_cont.clip(-1, 1) + 1) / 2 * (STICK_BINS - 1)
        ).astype(int)
        for axis in range(STICK_DIM):
            counts[axis] += np.bincount(stk_bins[:, axis], minlength=STICK_BINS)
        pos = end

    total       = train_end
    bin_centers = [f"{b / (STICK_BINS - 1) * 2 - 1:.2f}" for b in range(STICK_BINS)]
    weights     = np.zeros((STICK_DIM, STICK_BINS), dtype=np.float32)

    print(f"Per-axis stick bin weights (max_weight={max_weight}):")
    for axis in range(STICK_DIM):
        w = np.clip(total / (STICK_BINS * np.maximum(counts[axis], 1.0)), 1.0, max_weight)
        weights[axis] = w
        neutral_frac = counts[axis, STICK_BINS // 2] / total
        print(f"  {AXIS_NAMES[axis]}: neutral={neutral_frac:.1%}  "
              f"w[neutral]={w[STICK_BINS//2]:.2f}  "
              f"w[bin0]={w[0]:.2f}  w[bin20]={w[20]:.2f}")

    return torch.tensor(weights, dtype=torch.float32)  # [STICK_DIM, STICK_BINS]


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                 warmup_steps: int,
                                 total_steps: int,
                                 last_epoch: int = -1):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(loader: DataLoader,
             model: CitrusTransformerBC,
             pos_weights: torch.Tensor,
             stk_weights: torch.Tensor,
             device: torch.device,
             use_amp: bool) -> dict:
    model.eval()
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    ce_fns = [nn.CrossEntropyLoss(weight=stk_weights[i].to(device)) for i in range(STICK_DIM)]

    total_bce = total_ce = n_batches = 0
    all_btn_pred, all_btn_true = [], []
    all_stk_pred, all_stk_true = [], []
    all_stk_pred_bins = []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device, non_blocking=True)   # [B, T, F]
            y_b = y_b.to(device, non_blocking=True)   # [B, T, L]
            X_b[:, :, -PREV_ACTION_DIM:] = 0.0        # zero prev_labels — matches inference
            btn_true = y_b[:, :, :BUTTON_DIM]
            stk_bins = float_to_bin(y_b[:, :, BUTTON_DIM:])

            with torch.amp.autocast("cuda", enabled=use_amp):
                btn_logits, stk_logits, _ = model(X_b, btn_true, stk_bins)

            B, T = X_b.shape[:2]
            bce = bce_fn(btn_logits.float().reshape(B * T, BUTTON_DIM),
                         btn_true.reshape(B * T, BUTTON_DIM))
            ce  = sum(
                ce_fns[i](stk_logits[i].float().reshape(-1, STICK_BINS),
                           stk_bins[:, :, i].reshape(-1))
                for i in range(STICK_DIM)
            ) / STICK_DIM

            if not (math.isfinite(bce.item()) and math.isfinite(ce.item())):
                continue
            total_bce += bce.item()
            total_ce  += ce.item()
            n_batches += 1

            btn_pred = (torch.sigmoid(btn_logits) > 0.5).reshape(B * T, BUTTON_DIM).cpu().numpy()
            all_btn_pred.append(btn_pred)
            all_btn_true.append(btn_true.reshape(B * T, BUTTON_DIM).cpu().numpy())

            stk_pred_bins = torch.stack([l.argmax(-1) for l in stk_logits], dim=-1)  # [B,T,4]
            stk_pred_f = bin_to_float(stk_pred_bins).reshape(B * T, STICK_DIM).cpu().numpy()
            all_stk_pred.append(stk_pred_f)
            all_stk_true.append(y_b[:, :, BUTTON_DIM:].reshape(B * T, STICK_DIM).cpu().numpy())
            all_stk_pred_bins.append(stk_pred_bins.reshape(B * T, STICK_DIM).cpu().numpy())

    avg_bce = total_bce / max(n_batches, 1)
    avg_ce  = total_ce  / max(n_batches, 1)

    btn_pred = np.concatenate(all_btn_pred)
    btn_true = np.concatenate(all_btn_true)
    stk_pred = np.concatenate(all_stk_pred)
    stk_true = np.concatenate(all_stk_true)

    f1 = {}
    for i, name in enumerate(BUTTON_NAMES):
        tp = int(((btn_pred[:, i] == 1) & (btn_true[:, i] == 1)).sum())
        fp = int(((btn_pred[:, i] == 1) & (btn_true[:, i] == 0)).sum())
        fn = int(((btn_pred[:, i] == 0) & (btn_true[:, i] == 1)).sum())
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1[name] = 2 * p * r / max(p + r, 1e-8)

    pa = np.arctan2(stk_pred[:, 1], stk_pred[:, 0])
    ta = np.arctan2(stk_true[:, 1], stk_true[:, 0])
    d  = np.abs(pa - ta)
    d  = np.where(d > math.pi, 2 * math.pi - d, d)
    stick_err_deg = float(np.degrees(d.mean()))

    cpa = np.arctan2(stk_pred[:, 3], stk_pred[:, 2])
    cta = np.arctan2(stk_true[:, 3], stk_true[:, 2])
    cd  = np.abs(cpa - cta)
    cd  = np.where(cd > math.pi, 2 * math.pi - cd, cd)
    cstick_err_deg = float(np.degrees(cd.mean()))

    idle_mask = np.all(btn_true < 0.5, axis=1)
    idle_acc  = float(np.all(btn_pred[idle_mask] == 0, axis=1).mean()) if idle_mask.any() else 1.0

    neutral_bin = STICK_BINS // 2  # bin 10 for STICK_BINS=21
    stk_pred_bins = np.concatenate(all_stk_pred_bins)  # [N, 4]
    nonneut_pct = [(stk_pred_bins[:, i] != neutral_bin).mean() for i in range(STICK_DIM)]

    return {
        "loss": avg_bce + avg_ce, "bce": avg_bce, "ce": avg_ce,
        "f1": f1, "stick_err_deg": stick_err_deg, "cstick_err_deg": cstick_err_deg,
        "idle_acc": idle_acc, "nonneut_pct": nonneut_pct,
    }


def composite_score(m: dict) -> float:
    return sum(m["f1"].values()) / len(m["f1"])


def print_eval(tag: str, m: dict) -> None:
    f1s = "  ".join(f"{k}={v:.3f}" for k, v in m["f1"].items())
    nn = m.get("nonneut_pct", [0, 0, 0, 0])
    print(f"  [{tag}] loss={m['loss']:.4f}  bce={m['bce']:.4f}  stk_ce={m['ce']:.4f}")
    print(f"         F1: {f1s}")
    print(f"         stick_err={m['stick_err_deg']:.1f}°  cstick_err={m['cstick_err_deg']:.1f}°  idle_acc={m['idle_acc']:.3f}")
    print(f"         non-neutral: sx={nn[0]:.1%}  sy={nn[1]:.1%}  cx={nn[2]:.1%}  cy={nn[3]:.1%}")


def print_targets(m: dict) -> None:
    print("\n  Target check:")
    all_pass = True
    for name, target in F1_TARGETS.items():
        ok   = m["f1"][name] >= target
        print(f"    [{'PASS' if ok else 'FAIL'}] {name} F1  {m['f1'][name]:.3f}  (>= {target})")
        all_pass = all_pass and ok
    ok = m["stick_err_deg"] <= STICK_ERR_TARGET
    print(f"    [{'PASS' if ok else 'FAIL'}] stick_err  {m['stick_err_deg']:.1f}°  (<= {STICK_ERR_TARGET}°)")
    all_pass = all_pass and ok
    ok = m["idle_acc"] >= IDLE_ACC_TARGET
    print(f"    [{'PASS' if ok else 'FAIL'}] idle_acc  {m['idle_acc']:.3f}  (>= {IDLE_ACC_TARGET})")
    all_pass = all_pass and ok
    print(f"\n  Overall: {'ALL TARGETS MET' if all_pass else 'some targets missed'}")


# ── Training curves plot ───────────────────────────────────────────────────────

def _save_training_plot(history: dict, out_dir: Path) -> None:
    """Save training_curves.png to out_dir. Silently skips on any error."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
        import matplotlib.pyplot as plt

        epochs = history["epoch"]
        if not epochs:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Training Curves", fontsize=13)

        # ── Loss ──────────────────────────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], label="train loss", color="steelblue")
        val_epochs = [e for e, v in zip(epochs, history["val_loss"]) if v is not None]
        val_losses  = [v for v in history["val_loss"] if v is not None]
        if val_losses:
            ax.plot(val_epochs, val_losses, label="val loss", color="tomato")
        ax.set_title("Loss (BCE + stk CE)")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Composite score ────────────────────────────────────────────────────
        ax = axes[0, 1]
        comp_epochs = [e for e, v in zip(epochs, history["val_composite"]) if v is not None]
        comp_vals   = [v for v in history["val_composite"] if v is not None]
        if comp_vals:
            ax.plot(comp_epochs, comp_vals, color="mediumseagreen", label="composite (mean F1)")
        ax.set_title("Val Composite Score")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Per-button F1 ──────────────────────────────────────────────────────
        ax = axes[1, 0]
        colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
        for name, color in zip(BUTTON_NAMES, colors):
            f1_vals   = history["val_f1"][name]
            f1_epochs = [e for e, v in zip(epochs, f1_vals) if v is not None]
            f1_nonull = [v for v in f1_vals if v is not None]
            if f1_nonull:
                ax.plot(f1_epochs, f1_nonull, label=name, color=color)
        ax.set_title("Val F1 per Button")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

        # ── Stick error ────────────────────────────────────────────────────────
        ax = axes[1, 1]
        se_epochs = [e for e, v in zip(epochs, history["val_stick_err"]) if v is not None]
        se_vals   = [v for v in history["val_stick_err"] if v is not None]
        if se_vals:
            ax.plot(se_epochs, se_vals, color="darkorange", label="stick err (°)")
            ax.axhline(STICK_ERR_TARGET, color="red", linestyle="--",
                       linewidth=0.8, label=f"target {STICK_ERR_TARGET}°")
        ax.set_title("Val Stick Error (degrees)")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "training_curves.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  [plot] warning: could not save training_curves.png — {exc}")


# ── Training ──────────────────────────────────────────────────────────────────

def train(x_path: str, y_path: str, ms_path: str, seg_path: str, out_dir: str,
          epochs: int, batch_size: int, lr: float, weight_decay: float,
          seed: int, neutral_keep: float, warmup_steps: int, eval_every: int,
          num_workers: int, use_amp: bool, grad_accum: int,
          resume: str | None = None, start_epoch: int = 1) -> None:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if use_amp and device.type != "cuda":
        print("  WARNING: --amp has no effect without CUDA; ignoring.")
        use_amp = False
    torch.manual_seed(seed)

    # ── Load small arrays into RAM ───────────────────────────────────────────
    print("\nLoading seg / ms arrays...", flush=True)
    seg = np.load(seg_path).astype(np.int32)
    ms  = np.load(ms_path)
    N   = len(seg)
    print(f"  Frames: {N:,}  Matches: {len(ms):,}  "
          f"Segments: {int(seg[-1]) + 1 if len(seg) else 0:,}")

    # ── Sequence indices ─────────────────────────────────────────────────────
    all_seq = build_seq_indices(seg, SEQ_LEN)
    print(f"  Valid sequences (T={SEQ_LEN}): {len(all_seq):,}")

    train_seq, val_seq, test_seq, train_end = temporal_split_seqs(all_seq, ms, N, SEQ_LEN)
    print(f"\nTemporal split (70/15/15 by match order):")
    print(f"  Train: {len(train_seq):>9,} sequences  ({train_end:,} frames)")
    print(f"  Val:   {len(val_seq):>9,} sequences")
    print(f"  Test:  {len(test_seq):>9,} sequences")

    # ── Action-frame mask (sequential scan of y) ─────────────────────────────
    print(f"\nScanning labels for action frames (train portion)...", flush=True)
    t0 = time.time()
    action_mask = build_action_frame_mask(y_path, train_end)
    print(f"  Done in {time.time() - t0:.1f}s  "
          f"({100 * action_mask.mean():.1f}% action frames in training data)")

    # ── Oversampling ─────────────────────────────────────────────────────────
    train_seq_s = oversample_seqs(train_seq, action_mask, SEQ_LEN, neutral_keep, seed)
    action_n    = int((action_mask[train_seq + SEQ_LEN // 2] > 0).sum())
    print(f"\nOversampling (neutral_keep={neutral_keep:.0%}):")
    print(f"  Training set after oversampling: {len(train_seq_s):,} sequences")

    # ── Normalization stats ───────────────────────────────────────────────────
    norm_path = out / "norm_stats.npz"
    if norm_path.exists():
        print(f"\nLoading cached norm stats: {norm_path}")
        nz = np.load(norm_path)
        norm_mean, norm_std = nz["mean"], nz["std"]
    else:
        print(f"\nComputing norm stats over {train_end:,} training frames...", flush=True)
        t0 = time.time()
        norm_mean, norm_std = compute_norm_stats(x_path, train_end)
        np.savez(norm_path, mean=norm_mean, std=norm_std)
        print(f"  Done in {time.time() - t0:.1f}s  saved: {norm_path}")

    norm_mean_t = torch.from_numpy(norm_mean)  # keep on CPU; dataset applies per-item
    norm_std_t  = torch.from_numpy(norm_std)

    # ── Positive-class weights and stick bin weights ──────────────────────────
    print()
    pos_weights, _ = compute_pos_weights(y_path, train_end)
    print()
    stk_weights = compute_stick_bin_weights(y_path, train_end)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = StrikersDataset(x_path, y_path, train_seq_s, SEQ_LEN, norm_mean_t, norm_std_t)
    val_ds   = StrikersDataset(x_path, y_path, val_seq,     SEQ_LEN, norm_mean_t, norm_std_t)
    test_ds  = StrikersDataset(x_path, y_path, test_seq,    SEQ_LEN, norm_mean_t, norm_std_t)

    train_loader = make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = make_loader(test_ds,  batch_size, shuffle=False, num_workers=num_workers)

    # ── Model ────────────────────────────────────────────────────────────────
    model    = CitrusTransformerBC().to(device)
    n_params = count_params(model)
    print(f"\nModel: CitrusTransformerBC — {n_params:,} parameters")
    print(f"  EntityEncoder:    {count_params(model.entity_encoder):>10,}")
    print(f"  TemporalTransf:   {count_params(model.temporal):>10,}")
    print(f"  CtrlHead:         {count_params(model.ctrl_head):>10,}")

    if resume:
        print(f"\nResuming from: {resume}  (starting at epoch {start_epoch})")
        model.load_state_dict(torch.load(resume, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader) // grad_accum
    total_opt_steps = steps_per_epoch * epochs
    # Fast-forward scheduler to where we left off so LR continues the cosine curve.
    # initial_lr must be set in param_groups manually when resuming without optimizer state.
    steps_already_done = steps_per_epoch * (start_epoch - 1)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = lr
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps,
                                             last_epoch=steps_already_done - 1)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    ce_fns = [nn.CrossEntropyLoss(weight=stk_weights[i].to(device)) for i in range(STICK_DIM)]

    print(f"\nTraining: {epochs} epochs × {len(train_loader):,} batches/epoch"
          f"  (batch={batch_size} × T={SEQ_LEN} = {batch_size * SEQ_LEN:,} frames/step)")
    print(f"  Optimiser: AdamW  lr={lr}  wd={weight_decay}"
          f"  warmup={warmup_steps} steps  grad_accum={grad_accum}")
    print(f"  AMP: {'enabled (bf16)' if use_amp else 'disabled'}")
    print("-" * 70)

    # CSV log
    csv_path = out / "training_history.csv"
    csv_cols  = (["epoch", "train_loss", "train_bce", "val_loss", "val_bce", "gap",
                  "val_composite"]
                 + [f"val_f1_{n}" for n in BUTTON_NAMES]
                 + ["val_stick_err_deg", "val_idle_acc"])

    def _fmt(v):
        return f"{v:.6f}" if isinstance(v, float) else ("" if v is None else str(v))

    best_val_loss  = float("inf")
    best_epoch     = -1
    opt_step       = steps_already_done

    # ── Plotting history ─────────────────────────────────────────────────────
    plot_history: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_composite": [],
        "val_stick_err": [],
        "val_f1": {name: [] for name in BUTTON_NAMES},
    }

    # When resuming, seed plot_history and best_val_loss from the existing CSV
    # so the training curve and best-model tracking include prior epochs.
    resuming = start_epoch > 1 and csv_path.exists()
    if resuming:
        import csv as _csv
        with open(csv_path, newline="") as _f:
            for row in _csv.DictReader(_f):
                ep = int(row["epoch"])
                if ep >= start_epoch:
                    break  # don't load rows we're about to rewrite
                plot_history["epoch"].append(ep)
                plot_history["train_loss"].append(float(row["train_loss"]) if row["train_loss"] else None)
                plot_history["val_loss"].append(float(row["val_loss"]) if row["val_loss"] else None)
                plot_history["val_composite"].append(float(row["val_composite"]) if row["val_composite"] else None)
                plot_history["val_stick_err"].append(float(row["val_stick_err_deg"]) if row["val_stick_err_deg"] else None)
                for n in BUTTON_NAMES:
                    v = row.get(f"val_f1_{n}", "")
                    plot_history["val_f1"][n].append(float(v) if v else None)
                vl = float(row["val_loss"]) if row["val_loss"] else float("inf")
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_epoch    = ep

    # Append to CSV when resuming (keep prior rows); write fresh otherwise.
    csv_file = open(csv_path, "a" if resuming else "w", buffering=1)
    if not resuming:
        csv_file.write(",".join(csv_cols) + "\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_bce = epoch_ce = 0.0
        n_batches = 0
        t0 = time.time()

        optimizer.zero_grad()

        for batch_i, (X_b, y_b) in enumerate(train_loader):
            X_b = X_b.to(device, non_blocking=True)   # [B, T, F]
            y_b = y_b.to(device, non_blocking=True)   # [B, T, L]
            X_b[:, :, -PREV_ACTION_DIM:] = 0.0        # zero prev_labels — matches inference

            # Skip batches with corrupt input data before doing any GPU work
            if not torch.isfinite(X_b).all():
                print(f"  WARN: non-finite input at epoch {epoch} batch {batch_i} — skipping",
                      flush=True)
                continue

            btn_true = y_b[:, :, :BUTTON_DIM]
            stk_bins = float_to_bin(y_b[:, :, BUTTON_DIM:])

            with torch.amp.autocast("cuda", enabled=use_amp):
                btn_logits, stk_logits, _ = model(X_b, btn_true, stk_bins)
                B, T = X_b.shape[:2]
                # Cast logits to fp32 before loss — bf16 logits can overflow in exp()
                # inside CrossEntropyLoss/BCEWithLogitsLoss, producing NaN.
                bce = bce_fn(btn_logits.float().reshape(B * T, BUTTON_DIM),
                             btn_true.reshape(B * T, BUTTON_DIM))
                ce  = sum(
                    ce_fns[i](stk_logits[i].float().reshape(-1, STICK_BINS),
                               stk_bins[:, :, i].reshape(-1))
                    for i in range(STICK_DIM)
                ) / STICK_DIM
                loss = (bce + ce) / grad_accum

            # Check for NaN/Inf BEFORE backward — accumulating NaN gradients
            # poisons model weights even when the optimizer step is skipped.
            if not torch.isfinite(loss * grad_accum):
                print(f"  WARN: non-finite loss at epoch {epoch} batch {batch_i} "
                      f"(bce={bce.item():.4f} stk_ce={ce.item():.4f}) — skipping",
                      flush=True)
                optimizer.zero_grad()
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_bce += bce.item()
            epoch_ce  += ce.item()
            n_batches += 1

            # Gradient step every grad_accum batches
            if (batch_i + 1) % grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                opt_step += 1

            if (batch_i + 1) % 500 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                pct    = (batch_i + 1) / len(train_loader) * 100
                print(f"  epoch {epoch}  [{pct:5.1f}%]  "
                      f"bce={epoch_bce / n_batches:.4f}  "
                      f"stk_ce={epoch_ce / n_batches:.4f}  "
                      f"lr={cur_lr:.2e}",
                      flush=True)

        train_bce  = epoch_bce / max(n_batches, 1)
        train_ce   = epoch_ce  / max(n_batches, 1)
        train_loss = train_bce + train_ce
        elapsed    = time.time() - t0
        print(f"\nEpoch {epoch:3d}/{epochs}  ({elapsed:.0f}s)")
        print(f"  [train] loss={train_loss:.4f}  bce={train_bce:.4f}  stk_ce={train_ce:.4f}")

        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        if do_eval:
            val_m   = evaluate(val_loader, model, pos_weights, stk_weights, device, use_amp)
            score   = composite_score(val_m)
            is_best = val_m["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_m["loss"]
                best_epoch    = epoch
                torch.save(model.state_dict(), out / "best_model.pt")
            gap      = val_m["loss"] - train_loss
            gap_flag = "  << overfitting" if gap > 0.05 else ""
            print_eval("val", val_m)
            print(f"  gap(val-train)={gap:+.4f}{gap_flag}  "
                  f"composite={score:.4f}  "
                  f"best val_loss={best_val_loss:.4f} (epoch {best_epoch})"
                  + (" <- best" if is_best else ""))
            f1 = val_m["f1"]
            csv_row = ([epoch, train_loss, train_bce,
                        val_m["loss"], val_m["bce"], gap, score]
                       + [f1.get(n) for n in BUTTON_NAMES]
                       + [val_m["stick_err_deg"], val_m["idle_acc"]])
        else:
            print(f"  [val skipped — eval_every={eval_every}]")
            csv_row = ([epoch, train_loss, train_bce,
                        None, None, None, None]
                       + [None] * len(BUTTON_NAMES) + [None, None])

        csv_file.write(",".join(_fmt(v) for v in csv_row) + "\n")

        # ── Update plot history & redraw ──────────────────────────────────────
        plot_history["epoch"].append(epoch)
        plot_history["train_loss"].append(train_loss)
        if do_eval:
            plot_history["val_loss"].append(val_m["loss"])
            plot_history["val_composite"].append(score)
            plot_history["val_stick_err"].append(val_m["stick_err_deg"])
            for name in BUTTON_NAMES:
                plot_history["val_f1"][name].append(val_m["f1"][name])
        else:
            plot_history["val_loss"].append(None)
            plot_history["val_composite"].append(None)
            plot_history["val_stick_err"].append(None)
            for name in BUTTON_NAMES:
                plot_history["val_f1"][name].append(None)
        _save_training_plot(plot_history, out)

        print("-" * 70)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), out / "final_model.pt")
    csv_file.close()
    print(f"\nSaved: {out / 'best_model.pt'}  (epoch {best_epoch})")
    print(f"Saved: {out / 'final_model.pt'}")
    print(f"Saved: {csv_path}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("TEST SET EVALUATION (best model)")
    print("=" * 70)
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    test_m = evaluate(test_loader, model, pos_weights, stk_weights, device, use_amp)
    print_eval("test", test_m)
    print_targets(test_m)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CitrusTransformerBC on Strikers CITF dataset"
    )
    parser.add_argument("x_path",   help="*_X.npy features (memory-mapped)")
    parser.add_argument("y_path",   help="*_y.npy labels")
    parser.add_argument("ms_path",  help="*_ms.npy match start indices")
    parser.add_argument("seg_path", help="*_seg.npy segment IDs")
    parser.add_argument("out_dir",  help="Output directory (models, logs, norm stats)")

    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch",        type=int,   default=256,
                        help="Sequences per batch (default 256; ≈16K frames/step on GPU)")
    parser.add_argument("--lr",           type=float, default=3e-4,
                        help="Peak learning rate for AdamW (default 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--neutral-keep", type=float, default=1.0,
                        help="Fraction of all-neutral sequences to keep (default 1.0)")
    parser.add_argument("--warmup-steps", type=int,   default=1000,
                        help="Linear LR warmup steps before cosine decay (default 1000)")
    parser.add_argument("--eval-every",   type=int,   default=1,
                        help="Run validation every N epochs (default 1)")
    parser.add_argument("--workers",      type=int,   default=4,
                        help="DataLoader worker processes (default 4; set 0 for debugging)")
    parser.add_argument("--amp",          action="store_true",
                        help="Enable bf16 AMP training (recommended on A100/A6000)")
    parser.add_argument("--grad-accum",   type=int,   default=1,
                        help="Gradient accumulation steps (default 1; increase if OOM)")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to model weights (.pt) to resume from")
    parser.add_argument("--start-epoch",  type=int,   default=1,
                        help="Epoch to start from when resuming (1-based; used to fast-forward LR schedule)")

    args = parser.parse_args()

    train(
        x_path       = args.x_path,
        y_path       = args.y_path,
        ms_path      = args.ms_path,
        seg_path     = args.seg_path,
        out_dir      = args.out_dir,
        epochs       = args.epochs,
        batch_size   = args.batch,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        seed         = args.seed,
        neutral_keep = args.neutral_keep,
        warmup_steps = args.warmup_steps,
        eval_every   = args.eval_every,
        num_workers  = args.workers,
        use_amp      = args.amp,
        grad_accum   = args.grad_accum,
        resume       = args.resume,
        start_epoch  = args.start_epoch,
    )
