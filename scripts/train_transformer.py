#!/usr/bin/env python3
"""train_transformer.py — Behavioral cloning transformer for Citrus Strikers (v6).

Architecture:
  EntityEncoder (2L, 256d, 4h):             per-frame spatial attention over 20 entity tokens
                                            with learned action-state embeddings
  CausalTemporalTransformer (3L, 512d, 4h): causal attention over T=128 frame window
  ConditionalControllerHead:                buttons (independent) → sticks (conditioned on buttons)
  Focal loss for buttons, weighted CE for sticks

v6 changes from v5:
  - Action states: one-hot (30/27 dims) → integer index + learned embedding (8 dims)
  - Composite lob labels: L removed, lob_pass (L+A) and chip_shot (L+B) added → 7 buttons
  - Conditional stick heads: sticks conditioned on button probabilities
  - Focal loss replaces pos_weight BCE for buttons
  - Rare-action sequence oversampling (X, lob_pass, chip_shot, Y, c-stick deke)
  - Kickoff-containing sequences force-included in training
  - Active field items as 4 entity tokens (top-K nearest)
  - Ball-to-goal geometry features
  - Removed: score_diff, time_fraction
  - Added: is_kickoff, goalie_has_ball booleans
  - prev_labels re-enabled (no longer zeroed)
  - SEQ_LEN=128, full checkpointing

Requires PyTorch >= 2.0 (F.scaled_dot_product_attention with is_causal=True).

Usage:
    python train_transformer.py <X.npy> <y.npy> <ms.npy> <seg.npy> <out_dir>
    python train_transformer.py <X.npy> <y.npy> <ms.npy> <seg.npy> <out_dir> \\
        --epochs 20 --batch 256 --workers 4

Entity token layout (20 tokens, ENTITY_RAW_DIM=32):
    Derived from build_dataset.py v6 feature layout — must stay in sync.
    Token  Slice       Raw dim  Entity type   Notes
     0     [0:22]        22     ball          pos(3)+vel(3)+charge+pp+b2g(3)+owner_oh(11)
     1     [22:41]       19     self          pos_delta(3)+state_idx(1)+heading(2)+goal(3)+effect(5)+speed(3)+timer+carrier
     2     [41:48]        7     friend str 0  pos_delta(3)+state_idx(1)+heading(2)+carrier
     3     [48:55]        7     friend str 1
     4     [55:62]        7     friend str 2
     5     [62:66]        4     friend goalie pos_delta(3)+state_idx(1)
     6     [66:73]        7     enemy str 0
     7     [73:80]        7     enemy str 1
     8     [80:87]        7     enemy str 2
     9     [87:94]        7     enemy str 3
    10     [94:98]        4     enemy goalie
    11     [98:109]      11     own inv 0     powerup_oh(10)+charge(1)
    12     [109:120]     11     own inv 1
    13     [120:131]     11     enemy inv 0
    14     [131:142]     11     enemy inv 1
    15     [142:150]      8     field item 0  type_idx(1)+pos_delta(3)+vel(3)+strength(1)
    16     [150:158]      8     field item 1
    17     [158:166]      8     field item 2
    18     [166:174]      8     field item 3
    19     [174:194]     20     context       tactical(5)+possession(2)+phase(2)+prev_action(11)
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

SEQ_LEN     = 128     # frames per training sequence (~2.1 s at 60fps)
FEATURE_DIM = 194     # flat features per frame (must match build_dataset.py v6)
BUTTON_DIM      = 7       # labels 0-6: A, B, X, Y, lob_pass, chip_shot, R
STICK_DIM       = 4       # labels 7-10: stick_x, stick_y, cstick_x, cstick_y
LABEL_DIM       = BUTTON_DIM + STICK_DIM
STICK_BINS      = 21      # discrete stick bins per axis [-1,1] → 0..20
PREV_ACTION_DIM = LABEL_DIM  # last 11 features are prev-frame labels (re-enabled in v6)
PREV_ACTION_OFFSET = 183     # flat feature index where prev_action starts (174+5+2+2)

# ── Action vocabulary (replaces independent sigmoid buttons) ──────────────────
# Each row maps an action index to 7 binary button flags [A B X Y lob chip R].
NUM_ACTIONS = 12
ACTION_VOCAB = torch.tensor([
    # A  B  X  Y  lob chip R
    [0, 0, 0, 0, 0, 0, 0],   #  0: (none)
    [0, 0, 0, 0, 0, 0, 1],   #  1: R
    [1, 0, 0, 0, 0, 0, 0],   #  2: A
    [1, 0, 0, 0, 0, 0, 1],   #  3: A+R
    [0, 1, 0, 0, 0, 0, 0],   #  4: B
    [0, 1, 0, 0, 0, 0, 1],   #  5: B+R
    [0, 0, 0, 1, 0, 0, 0],   #  6: Y
    [0, 0, 0, 1, 0, 0, 1],   #  7: Y+R
    [0, 0, 1, 0, 0, 0, 0],   #  8: X
    [0, 0, 1, 0, 0, 0, 1],   #  9: X+R
    [1, 0, 0, 0, 1, 0, 0],   # 10: L+A (lob pass)
    [0, 1, 0, 0, 0, 1, 0],   # 11: L+B (chip shot)
], dtype=torch.float32)
ACTION_NAMES = [
    "(none)", "R", "A", "A+R", "B", "B+R",
    "Y", "Y+R", "X", "X+R", "L+A", "L+B",
]

# Build a 128-entry lookup table: 7-bit button key → action index (0 for unlisted).
# This allows GPU-vectorized O(1) conversion from 7 binary labels to action indices.
_ACTION_KEY_TO_IDX = torch.zeros(128, dtype=torch.long)
for _i, _row in enumerate(ACTION_VOCAB):
    _key = int(sum(int(b) << j for j, b in enumerate(_row.tolist())))
    _ACTION_KEY_TO_IDX[_key] = _i


def labels_to_action_idx(btn_labels: torch.Tensor) -> torch.Tensor:
    """Convert binary button labels [..., 7] to action indices [...].

    Uses a 128-entry lookup table indexed by a 7-bit key.
    Unlisted combos map to action 0 (none).
    """
    bits = (btn_labels > 0.5).long()
    keys = torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
    for j in range(BUTTON_DIM):
        keys = keys + (bits[..., j] << j)
    lut = _ACTION_KEY_TO_IDX.to(bits.device)
    return lut[keys]

ENTITY_RAW_DIM  = 32   # zero-padded token width fed into EntityEncoder
N_ENTITY_TYPES  = 11   # ball / self / fr_str / fr_gk / en_str / en_gk / inv_own / inv_enemy / field_item / ctx / (reserved)
N_ENTITIES      = 20

# Action state embedding dims (states stored as integer indices in features)
STRIKER_STATE_VOCAB = 30   # 29 known + 1 other
GOALIE_STATE_VOCAB  = 27   # 26 known + 1 other
STATE_EMBED_DIM     = 8    # learned embedding dimension per state
FIELD_ITEM_VOCAB    = 10   # 9 powerup types + 1 padding
ITEM_EMBED_DIM      = 4    # learned embedding dimension per item type

ENTITY_DIM      = 256  # EntityEncoder hidden dim
ENTITY_LAYERS   = 2
ENTITY_HEADS    = 4

TEMPORAL_DIM    = 512  # TemporalTransformer hidden dim
TEMPORAL_LAYERS = 3
TEMPORAL_HEADS  = 4
FF_MULT         = 2

BUTTON_NAMES = ["A", "B", "X", "Y", "lob_pass", "chip_shot", "R"]
STICK_NAMES  = ["stick_x", "stick_y", "cstick_x", "cstick_y"]

F1_TARGETS       = {"A": 0.45, "B": 0.55, "X": 0.30, "Y": 0.30,
                     "lob_pass": 0.20, "chip_shot": 0.20, "R": 0.50}
STICK_ERR_TARGET = 25.0
IDLE_ACC_TARGET  = 0.90

# ── Entity token layout (v6) ──────────────────────────────────────────────────
# (slice_start, slice_end_exclusive, entity_type_id)
# Slice offsets derived from build_dataset.py v6 feature layout.
# Action states are stored as single float (integer index) — the model's
# EntityEncoder embeds them via nn.Embedding.

_ENTITY_DEFS: List[Tuple[int, int, int]] = [
    (  0,  22, 0),  # ball: pos×3 vel×3 charge pp b2g×3 + owner_oh×11
    ( 22,  41, 1),  # self: pos_delta×3 state_idx×1 heading×2 goal×3 effect×5 spd×3 timer carrier
    ( 41,  48, 2),  # friendly striker 0: pos_delta×3 state_idx×1 heading×2 carrier
    ( 48,  55, 2),  # friendly striker 1
    ( 55,  62, 2),  # friendly striker 2
    ( 62,  66, 3),  # friendly goalie: pos_delta×3 state_idx×1
    ( 66,  73, 4),  # enemy striker 0
    ( 73,  80, 4),  # enemy striker 1
    ( 80,  87, 4),  # enemy striker 2
    ( 87,  94, 4),  # enemy striker 3
    ( 94,  98, 5),  # enemy goalie
    ( 98, 109, 6),  # own inventory 0: powerup_oh×10 charge
    (109, 120, 6),  # own inventory 1
    (120, 131, 7),  # enemy inventory 0
    (131, 142, 7),  # enemy inventory 1
    (142, 150, 8),  # field item 0: type_idx×1 pos_delta×3 vel×3 strength×1
    (150, 158, 8),  # field item 1
    (158, 166, 8),  # field item 2
    (166, 174, 8),  # field item 3
    (174, 194, 9),  # context: tactical×5 possession×2 phase×2 prev_action×11
]

assert len(_ENTITY_DEFS) == N_ENTITIES, "N_ENTITIES mismatch"
assert all(0 <= s and e <= FEATURE_DIM and (e - s) <= ENTITY_RAW_DIM
           for s, e, _ in _ENTITY_DEFS), "Entity slice out of bounds or exceeds ENTITY_RAW_DIM"
assert sum(e - s for s, e, _ in _ENTITY_DEFS) == FEATURE_DIM, (
    f"Entity slices do not cover all {FEATURE_DIM} features")

_ENTITY_TYPE_IDS = [t for _, _, t in _ENTITY_DEFS]   # [20] — used in EntityEncoder

# Offsets within entity tokens where the action state index lives.
# Used by EntityEncoder to extract, embed, and replace the integer index.
# These are relative to the token's raw dims (after slicing from flat features).
_SELF_STATE_IDX_OFFSET    = 3   # self token: [pos_delta×3, STATE_IDX, ...]
_STRIKER_STATE_IDX_OFFSET = 3   # friendly/enemy striker tokens: [pos_delta×3, STATE_IDX, ...]
_GOALIE_STATE_IDX_OFFSET  = 3   # goalie tokens: [pos_delta×3, STATE_IDX]
_ITEM_TYPE_IDX_OFFSET     = 0   # field item tokens: [TYPE_IDX, pos_delta×3, vel×3, strength]


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
    """Per-frame spatial encoder: entity tokens → single frame embedding (v6).

    Input:  [B, N_ENTITIES, ENTITY_RAW_DIM]
    Output: [B, TEMPORAL_DIM]

    v6: Action states and field item types are stored as integer indices in the
    feature vector. This encoder extracts them, looks up learned embeddings,
    and concatenates them with the remaining continuous features before projection.

    A learned entity-type embedding is added after projection so the model
    learns to distinguish ball / strikers / goalies / inventory / items / context.
    """

    def __init__(self):
        super().__init__()
        # Action state embeddings (shared across all characters of same type)
        self.striker_state_embed = nn.Embedding(STRIKER_STATE_VOCAB, STATE_EMBED_DIM)
        self.goalie_state_embed  = nn.Embedding(GOALIE_STATE_VOCAB,  STATE_EMBED_DIM)
        self.item_type_embed     = nn.Embedding(FIELD_ITEM_VOCAB,    ITEM_EMBED_DIM)

        # After embedding replacement, the effective token width changes:
        # - Self token: 19 raw → 18 continuous + STATE_EMBED_DIM(8) = 26
        # - Striker token: 7 raw → 6 continuous + STATE_EMBED_DIM(8) = 14
        # - Goalie token: 4 raw → 3 continuous + STATE_EMBED_DIM(8) = 11
        # - Item token: 8 raw → 7 continuous + ITEM_EMBED_DIM(4) = 11
        # - Others unchanged (ball=22, inv=11, context=20)
        # Max effective width = 26 (self). Project from padded max.
        self._effective_max = ENTITY_RAW_DIM + STATE_EMBED_DIM  # generous upper bound
        self.proj       = nn.Linear(self._effective_max, ENTITY_DIM)
        self.type_embed = nn.Embedding(N_ENTITY_TYPES, ENTITY_DIM)
        self.layers     = nn.ModuleList([
            EntityTransformerLayer(ENTITY_DIM, ENTITY_HEADS)
            for _ in range(ENTITY_LAYERS)
        ])
        self.out = nn.Linear(ENTITY_DIM, TEMPORAL_DIM)

        type_ids = torch.tensor(_ENTITY_TYPE_IDS, dtype=torch.long)
        self.register_buffer("type_ids", type_ids)  # [N_ENTITIES]

    def _embed_state(self, token: torch.Tensor, idx_offset: int,
                     embed: nn.Embedding) -> torch.Tensor:
        """Extract integer index at idx_offset, look up embedding, replace it.

        token: [B, raw_dim]
        Returns: [B, raw_dim - 1 + embed_dim]  (index removed, embedding inserted)
        """
        idx = token[:, idx_offset].long().clamp(0, embed.num_embeddings - 1)
        emb = embed(idx)  # [B, embed_dim]
        before = token[:, :idx_offset]
        after  = token[:, idx_offset + 1:]
        return torch.cat([before, emb, after], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, raw_dim] — zero-padded entity tokens
        B, N, _ = x.shape

        # Process each entity token: extract index, embed, pad to uniform width
        processed = []
        for i, (s, e, etype) in enumerate(_ENTITY_DEFS):
            token = x[:, i, :e - s]  # [B, token_raw_dim]

            if etype == 1:  # self (striker or goalie state at offset 3)
                token = self._embed_state(token, _SELF_STATE_IDX_OFFSET,
                                           self.striker_state_embed)
            elif etype == 2 or etype == 4:  # friendly/enemy striker
                token = self._embed_state(token, _STRIKER_STATE_IDX_OFFSET,
                                           self.striker_state_embed)
            elif etype == 3 or etype == 5:  # friendly/enemy goalie
                token = self._embed_state(token, _GOALIE_STATE_IDX_OFFSET,
                                           self.goalie_state_embed)
            elif etype == 8:  # field item
                token = self._embed_state(token, _ITEM_TYPE_IDX_OFFSET,
                                           self.item_type_embed)

            # Pad to uniform width for the linear projection
            if token.shape[1] < self._effective_max:
                pad = token.new_zeros(B, self._effective_max - token.shape[1])
                token = torch.cat([token, pad], dim=1)
            processed.append(token)

        x = torch.stack(processed, dim=1)  # [B, N, effective_max]
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


class ConditionalControllerHead(nn.Module):
    """Conditional controller head — sticks conditioned on action probabilities.

    Actions are predicted as a single categorical over NUM_ACTIONS valid button
    combos.  Stick axes are predicted from the policy embedding concatenated
    with softmax(action_logits), so the stick head knows which action is active
    (e.g. R=sprint changes stick meaning, lob/chip affect aim direction).

    Always uses softmax(action_logits) as conditioning in both training and
    inference — no teacher-forcing gap.
    """

    def __init__(self, input_dim: int, stick_bins: int = STICK_BINS):
        super().__init__()
        self.stick_bins = stick_bins
        self.action_head = nn.Linear(input_dim, NUM_ACTIONS)
        # Stick heads receive policy + action probabilities
        stk_input_dim    = input_dim + NUM_ACTIONS
        self.stk_heads   = nn.ModuleList([nn.Linear(stk_input_dim, stick_bins)
                                           for _ in range(STICK_DIM)])

    def forward_train(self,
                      policy: torch.Tensor,
                      btn_targets: torch.Tensor,
                      stk_targets: torch.Tensor):
        """Training forward.

        policy: [B, T, input_dim]
        Returns:
            action_logits: [B, T, NUM_ACTIONS]
            stk_logits: list of STICK_DIM tensors, each [B, T, STICK_BINS]
        """
        action_logits = self.action_head(policy)                      # [B, T, NUM_ACTIONS]
        action_probs  = torch.softmax(action_logits, dim=-1)          # [B, T, NUM_ACTIONS]
        stk_input     = torch.cat([policy, action_probs], dim=-1)     # [B, T, input_dim + NUM_ACTIONS]
        stk_logits    = [self.stk_heads[i](stk_input) for i in range(STICK_DIM)]
        return action_logits, stk_logits

    def forward_infer(self, policy_t: torch.Tensor):
        """Inference forward — same computation as training, zero gap.

        policy_t: [B, input_dim]
        Returns:
            action_idx [B] — sampled action index
            stk_bins   [B, STICK_DIM] — argmax bin per axis
        """
        action_logits = self.action_head(policy_t)                    # [B, NUM_ACTIONS]
        action_probs  = torch.softmax(action_logits, dim=-1)          # [B, NUM_ACTIONS]
        stk_input     = torch.cat([policy_t, action_probs], dim=-1)   # [B, input_dim + NUM_ACTIONS]
        stk_bins      = torch.stack(
            [self.stk_heads[i](stk_input).argmax(dim=-1) for i in range(STICK_DIM)],
            dim=-1)  # [B, STICK_DIM]
        return action_logits.argmax(dim=-1), stk_bins


class CitrusTransformerBC(nn.Module):
    """Full transformer behavioral cloning model.

    Training:  forward(x, btn_targets, stk_targets) — teacher-forced
    Inference: forward_infer(entities, kv_cache)     — ONNX-exportable
    """

    def __init__(self):
        super().__init__()
        self.entity_encoder = EntityEncoder()
        self.temporal       = CausalTemporalTransformer()
        self.ctrl_head      = ConditionalControllerHead(TEMPORAL_DIM)
        self.value_head     = nn.Linear(TEMPORAL_DIM, 1)  # reserved for future RL/PPO

    def forward(self,
                x: torch.Tensor,
                btn_targets: torch.Tensor,
                stk_targets: torch.Tensor):
        """Training forward.

        x:           [B, T, FEATURE_DIM]
        btn_targets: [B, T, BUTTON_DIM] float  (unused by ctrl_head, kept for API compat)
        stk_targets: [B, T, STICK_DIM]  int64  (unused by ctrl_head, kept for API compat)
        Returns: action_logits [B,T,NUM_ACTIONS], stk_logits list[B,T,21], value [B,T,1]
        """
        B, T, _ = x.shape
        entities  = flat_to_entities(x)                           # [B, T, N, raw]
        frame_emb = self.entity_encoder(
            entities.view(B * T, N_ENTITIES, ENTITY_RAW_DIM))    # [B*T, D]
        frame_emb = frame_emb.view(B, T, TEMPORAL_DIM)            # [B, T, D]
        temporal  = self.temporal(frame_emb)                       # [B, T, D]
        value     = self.value_head(temporal)                      # [B, T, 1]
        action_logits, stk_logits = self.ctrl_head.forward_train(temporal, btn_targets, stk_targets)
        return action_logits, stk_logits, value

    def forward_infer(self,
                      entities: torch.Tensor,
                      kv_cache: torch.Tensor):
        """ONNX inference path — one frame at a time.

        entities:  [1, N_ENTITIES, ENTITY_RAW_DIM]
        kv_cache:  [TEMPORAL_LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]
        Returns:
            action_idx   [1]                     — sampled action index
            stk_bins     [1, STICK_DIM]          — argmax bin indices (int64)
            kv_cache_out [TEMPORAL_LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]
        """
        frame_emb             = self.entity_encoder(entities)          # [1, D]
        out, kv_cache_out     = self.temporal.forward_cached(frame_emb, kv_cache)
        action_idx, stk_bins  = self.ctrl_head.forward_infer(out)
        return action_idx, stk_bins, kv_cache_out


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


# ── Focal loss ────────────────────────────────────────────────────────────────

class FocalBCEWithLogitsLoss(nn.Module):
    """Focal loss for binary classification (replaces BCEWithLogitsLoss + pos_weight).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logit) if target=1, else 1-sigmoid(logit).
    alpha is per-button (inverse frequency), gamma controls focus on hard examples.
    No manual pos_weight cap needed — the (1-p_t)^gamma term self-regulates.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)  # [BUTTON_DIM]
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: [N, BUTTON_DIM]
        p = torch.sigmoid(logits)
        # Binary cross-entropy (numerically stable via logsigmoid)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = probability of the true class
        p_t = p * targets + (1 - p) * (1 - targets)
        # alpha_t: alpha for positive, (1 - alpha) for negative
        # For rare buttons alpha is high, so false negatives get large weight.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def compute_focal_alpha(y_path: str, train_end: int,
                        chunk: int = 1_000_000) -> torch.Tensor:
    """Compute per-button alpha for focal loss from training label frequencies.

    alpha_i = 1 - rate_i  (inverse frequency, no cap needed with focal loss).
    """
    y    = np.lib.format.open_memmap(y_path, mode="r")
    sums = np.zeros(BUTTON_DIM, dtype=np.float64)
    pos  = 0
    while pos < train_end:
        end   = min(pos + chunk, train_end)
        sums += (y[pos:end, :BUTTON_DIM] > 0.5).sum(axis=0)
        pos   = end
    rates = sums / train_end
    alpha = np.clip(1.0 - rates, 0.1, 0.99)  # keep in [0.1, 0.99]

    print("Button press rates and focal alpha:")
    for name, r, a in zip(BUTTON_NAMES, rates, alpha):
        print(f"  {name:<12s}  rate={r:.4f}  alpha={a:.4f}")
    return torch.tensor(alpha, dtype=torch.float32)


def compute_action_weights(y_path: str, train_end: int,
                           chunk: int = 1_000_000) -> torch.Tensor:
    """Compute per-action class weights for CrossEntropyLoss.

    Uses sqrt(1/freq) to avoid over-weighting extremely rare classes.
    Returns: [NUM_ACTIONS] float32 tensor.
    """
    y = np.lib.format.open_memmap(y_path, mode="r")
    counts = np.zeros(NUM_ACTIONS, dtype=np.float64)
    pos = 0
    while pos < train_end:
        end = min(pos + chunk, train_end)
        btns = torch.from_numpy((y[pos:end, :BUTTON_DIM] > 0.5).astype(np.float32))
        idxs = labels_to_action_idx(btns).numpy()
        for i in range(NUM_ACTIONS):
            counts[i] += (idxs == i).sum()
        pos = end
    freqs = counts / counts.sum()
    freqs = np.maximum(freqs, 1e-8)  # avoid div-by-zero for unused classes
    weights = np.sqrt(1.0 / freqs)
    weights /= weights.mean()  # normalize so mean weight = 1

    print("Action vocabulary weights:")
    for name, c, f, w in zip(ACTION_NAMES, counts, freqs, weights):
        print(f"  {name:<12s}  count={int(c):>12,}  freq={f:.4f}  weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


# ── Rare-action oversampling ─────────────────────────────────────────────────

def oversample_rare_actions(seq_starts: np.ndarray,
                             y_path: str,
                             T: int,
                             seed: int) -> np.ndarray:
    """Duplicate sequences containing rare actions to improve representation.

    Duplication rates derived from v5 button press rates relative to A/B (~8%):
      X (item use):             4x duplication
      lob_pass (L+A):           4x duplication
      chip_shot (L+B):          4x duplication
      Y:                        3x duplication
      c-stick deke:             4x duplication

    Also force-includes sequences overlapping kickoff phase. Since gamePhase
    is not in y, kickoff detection relies on the is_kickoff feature in X
    (which must be checked separately via oversample_kickoff_seqs).
    """
    y = np.lib.format.open_memmap(y_path, mode="r")

    # Scan each sequence's labels for rare actions
    # Button indices: 0=A, 1=B, 2=X, 3=Y, 4=lob_pass, 5=chip_shot, 6=R
    # Stick indices (in label): 7=sx, 8=sy, 9=cx, 10=cy
    rare_dups = []
    for start in seq_starts:
        end = start + T
        labels = y[start:end]
        max_dup = 1  # default: no duplication

        # Check buttons
        if np.any(labels[:, 2] > 0.5):  # X (item use)
            max_dup = max(max_dup, 4)
        if np.any(labels[:, 4] > 0.5):  # lob_pass
            max_dup = max(max_dup, 4)
        if np.any(labels[:, 5] > 0.5):  # chip_shot
            max_dup = max(max_dup, 4)
        if np.any(labels[:, 3] > 0.5):  # Y
            max_dup = max(max_dup, 3)

        # Check c-stick deke (any non-neutral c-stick deflection)
        if np.any(np.abs(labels[:, 9]) > 0.15) or np.any(np.abs(labels[:, 10]) > 0.15):
            max_dup = max(max_dup, 4)

        rare_dups.append(max_dup)

    # Build duplicated index array
    result = []
    for start, dup in zip(seq_starts, rare_dups):
        result.extend([start] * dup)

    print(f"  Rare-action oversampling: {len(seq_starts):,} → {len(result):,} sequences "
          f"({len(result)/len(seq_starts):.2f}x)")
    return np.array(result, dtype=np.int64)


def oversample_kickoff_seqs(seq_starts: np.ndarray,
                             x_path: str,
                             T: int,
                             kickoff_feat_offset: int) -> np.ndarray:
    """Force-include and duplicate sequences containing kickoff frames (3x).

    kickoff_feat_offset: the index in the feature vector where is_kickoff lives.
    """
    X = np.lib.format.open_memmap(x_path, mode="r")
    extra = []
    for start in seq_starts:
        end = start + T
        if np.any(X[start:end, kickoff_feat_offset] > 0.5):
            extra.extend([start, start])  # +2 copies (total 3x including original)
    if extra:
        print(f"  Kickoff oversampling: +{len(extra):,} duplicate sequences")
    return np.concatenate([seq_starts, np.array(extra, dtype=np.int64)])


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
             action_ce_fn: nn.Module,
             stk_weights: torch.Tensor,
             device: torch.device,
             use_amp: bool) -> dict:
    model.eval()
    ce_fns = [nn.CrossEntropyLoss(weight=stk_weights[i].to(device)) for i in range(STICK_DIM)]
    vocab = ACTION_VOCAB.to(device)  # [NUM_ACTIONS, 7]

    total_act_ce = total_stk_ce = n_batches = 0
    all_btn_pred, all_btn_true = [], []
    all_stk_pred, all_stk_true = [], []
    all_stk_pred_bins = []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device, non_blocking=True)   # [B, T, F]
            y_b = y_b.to(device, non_blocking=True)   # [B, T, L]
            btn_true = y_b[:, :, :BUTTON_DIM]
            stk_bins = float_to_bin(y_b[:, :, BUTTON_DIM:])

            with torch.amp.autocast("cuda", enabled=use_amp):
                action_logits, stk_logits, _ = model(X_b, btn_true, stk_bins)

            B, T = X_b.shape[:2]
            action_true = labels_to_action_idx(btn_true)  # [B, T]
            act_ce = action_ce_fn(action_logits.float().reshape(B * T, NUM_ACTIONS),
                                  action_true.reshape(B * T))
            stk_ce = sum(
                ce_fns[i](stk_logits[i].float().reshape(-1, STICK_BINS),
                           stk_bins[:, :, i].reshape(-1))
                for i in range(STICK_DIM)
            ) / STICK_DIM

            if not (math.isfinite(act_ce.item()) and math.isfinite(stk_ce.item())):
                continue
            total_act_ce += act_ce.item()
            total_stk_ce += stk_ce.item()
            n_batches += 1

            # Predict buttons: argmax over action logits → lookup vocab for 7 binary flags
            action_pred_idx = action_logits.argmax(dim=-1)  # [B, T]
            btn_pred = vocab[action_pred_idx.reshape(-1)].cpu().numpy()  # [B*T, 7]
            all_btn_pred.append(btn_pred)
            all_btn_true.append(btn_true.reshape(B * T, BUTTON_DIM).cpu().numpy())

            stk_pred_bins = torch.stack([l.argmax(-1) for l in stk_logits], dim=-1)  # [B,T,4]
            stk_pred_f = bin_to_float(stk_pred_bins).reshape(B * T, STICK_DIM).cpu().numpy()
            all_stk_pred.append(stk_pred_f)
            all_stk_true.append(y_b[:, :, BUTTON_DIM:].reshape(B * T, STICK_DIM).cpu().numpy())
            all_stk_pred_bins.append(stk_pred_bins.reshape(B * T, STICK_DIM).cpu().numpy())

    avg_act_ce = total_act_ce / max(n_batches, 1)
    avg_stk_ce = total_stk_ce / max(n_batches, 1)

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
    idle_acc  = float(np.all(btn_pred[idle_mask] < 0.5, axis=1).mean()) if idle_mask.any() else 1.0

    neutral_bin = STICK_BINS // 2  # bin 10 for STICK_BINS=21
    stk_pred_bins = np.concatenate(all_stk_pred_bins)  # [N, 4]
    nonneut_pct = [(stk_pred_bins[:, i] != neutral_bin).mean() for i in range(STICK_DIM)]

    return {
        "loss": avg_act_ce + avg_stk_ce, "act_ce": avg_act_ce, "stk_ce": avg_stk_ce,
        "f1": f1, "stick_err_deg": stick_err_deg, "cstick_err_deg": cstick_err_deg,
        "idle_acc": idle_acc, "nonneut_pct": nonneut_pct,
    }


def composite_score(m: dict) -> float:
    return sum(m["f1"].values()) / len(m["f1"])


def print_eval(tag: str, m: dict) -> None:
    f1s = "  ".join(f"{k}={v:.3f}" for k, v in m["f1"].items())
    nn = m.get("nonneut_pct", [0, 0, 0, 0])
    print(f"  [{tag}] loss={m['loss']:.4f}  act_ce={m['act_ce']:.4f}  stk_ce={m['stk_ce']:.4f}")
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
        ax.set_title("Loss (act CE + stk CE)")
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
        colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]
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


# ── Scheduled Sampling ────────────────────────────────────────────────────────

def build_predicted_prev_labels(action_logits: torch.Tensor,
                                stk_logits: list,
                                ) -> torch.Tensor:
    """Convert model outputs into a prev_labels tensor suitable for replacing
    ground-truth prev_action features in the input.

    action_logits: [B, T, NUM_ACTIONS] — raw categorical logits
    stk_logits:    list of STICK_DIM tensors, each [B, T, STICK_BINS]

    Returns: [B, T, PREV_ACTION_DIM] where:
        [:, :, :BUTTON_DIM] = ACTION_VOCAB[sampled_action] (binary 0/1)
        [:, :, BUTTON_DIM:] = bin_to_float(argmax(stk_logits)) (continuous [-1,1])
    """
    # Gumbel-max sample an action index from the categorical
    gumbel = -torch.log(-torch.log(torch.rand_like(action_logits) + 1e-20) + 1e-20)
    action_idx = (action_logits + gumbel).argmax(dim=-1)             # [B, T]
    # Look up 7 binary button flags from the vocabulary
    vocab = ACTION_VOCAB.to(action_logits.device)                    # [12, 7]
    btn_binary = vocab[action_idx]                                   # [B, T, 7]
    stk_vals   = torch.stack(
        [bin_to_float(logit.argmax(dim=-1)) for logit in stk_logits],
        dim=-1)                                                      # [B, T, 4]
    return torch.cat([btn_binary, stk_vals], dim=-1)                 # [B, T, 11]


def apply_scheduled_sampling(X_b: torch.Tensor,
                             pred_prev: torch.Tensor,
                             ss_prob: float,
                             norm_mean: torch.Tensor,
                             norm_std: torch.Tensor,
                             ) -> torch.Tensor:
    """Replace ground-truth prev_labels in X_b with model predictions.

    For each frame t > 0 in each sequence, with probability ss_prob, overwrite
    the prev_action features (X_b[:, t, PREV_ACTION_OFFSET:]) with the model's
    prediction from frame t-1 (pred_prev[:, t-1, :]).

    pred_prev is in raw space (sigmoid probs / [-1,1] floats).  Since X_b is
    already normalised by the dataset, we normalise pred_prev the same way
    before inserting.

    Frame t=0 always keeps ground truth (no t-1 prediction available).

    Returns a new tensor (X_b is not modified in-place).
    """
    if ss_prob <= 0.0:
        return X_b
    X_mixed = X_b.clone()
    B, T, _ = X_b.shape
    # Normalise predicted prev_labels to match the dataset's normalisation
    pa_mean = norm_mean[PREV_ACTION_OFFSET:PREV_ACTION_OFFSET + PREV_ACTION_DIM].to(X_b.device)
    pa_std  = norm_std[PREV_ACTION_OFFSET:PREV_ACTION_OFFSET + PREV_ACTION_DIM].to(X_b.device)
    pred_normed = (pred_prev - pa_mean) / pa_std                     # [B, T, 11]
    # Bernoulli mask: [B, T-1] — which frames get predicted prev_labels
    mask = torch.rand(B, T - 1, device=X_b.device) < ss_prob        # [B, T-1]
    mask = mask.unsqueeze(-1).expand(-1, -1, PREV_ACTION_DIM)       # [B, T-1, 11]
    # pred_normed[:, :-1, :] = predictions at frames 0..T-2, used as prev_labels for frames 1..T-1
    X_mixed[:, 1:, PREV_ACTION_OFFSET:PREV_ACTION_OFFSET + PREV_ACTION_DIM] = torch.where(
        mask,
        pred_normed[:, :-1, :],
        X_b[:, 1:, PREV_ACTION_OFFSET:PREV_ACTION_OFFSET + PREV_ACTION_DIM],
    )
    return X_mixed


# ── Training ──────────────────────────────────────────────────────────────────

def train(x_path: str, y_path: str, ms_path: str, seg_path: str, out_dir: str,
          epochs: int, batch_size: int, lr: float, weight_decay: float,
          seed: int, neutral_keep: float, warmup_steps: int, eval_every: int,
          num_workers: int, use_amp: bool, grad_accum: int,
          resume: str | None = None, start_epoch: int = 1,
          ss_max: float = 0.5, ss_warmup: int = 3) -> None:

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
    print(f"\nOversampling (neutral_keep={neutral_keep:.0%}):")
    print(f"  After neutral undersampling: {len(train_seq_s):,} sequences")

    # Rare-action oversampling (v6): duplicate sequences with X, lob, chip, Y, deke
    train_seq_s = oversample_rare_actions(train_seq_s, y_path, SEQ_LEN, seed)

    # Kickoff oversampling (v6): force-include sequences overlapping kickoff phase
    # is_kickoff is at absolute offset 181 in the flat feature vector:
    # context token starts at 174, is_kickoff is at context[7] (after tactical×5 + possession×2)
    kickoff_offset = 174 + 5 + 2  # = 181
    train_seq_s = oversample_kickoff_seqs(train_seq_s, x_path, SEQ_LEN, kickoff_offset)

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

    # ── Action vocabulary weights and stick bin weights ─────────────────────
    print()
    action_weights = compute_action_weights(y_path, train_end)
    action_ce_fn = nn.CrossEntropyLoss(weight=action_weights.to(device))
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
        ckpt = torch.load(resume, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

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
    ce_fns    = [nn.CrossEntropyLoss(weight=stk_weights[i].to(device)) for i in range(STICK_DIM)]

    print(f"\nTraining: {epochs} epochs × {len(train_loader):,} batches/epoch"
          f"  (batch={batch_size} × T={SEQ_LEN} = {batch_size * SEQ_LEN:,} frames/step)")

    print(f"  Optimiser: AdamW  lr={lr}  wd={weight_decay}"
          f"  warmup={warmup_steps} steps  grad_accum={grad_accum}")
    print(f"  AMP: {'enabled (bf16)' if use_amp else 'disabled'}")
    print(f"  Loss: action CE ({NUM_ACTIONS}-class) + weighted CE sticks")
    print("-" * 70)

    # CSV log
    csv_path = out / "training_history.csv"
    csv_cols  = (["epoch", "train_loss", "train_act_ce", "val_loss", "val_act_ce", "gap",
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
        epoch_act_ce = epoch_stk_ce = 0.0
        n_batches = 0
        t0 = time.time()

        # ── Scheduled sampling probability for this epoch ─────────────────────
        # Ramps linearly from 0 to ss_max after ss_warmup epochs.
        # During warmup epochs, ss_prob stays 0 (pure teacher forcing).
        if ss_max > 0 and epoch > ss_warmup:
            ss_prob = min(ss_max, ss_max * (epoch - ss_warmup) / max(epochs - ss_warmup, 1))
        else:
            ss_prob = 0.0
        if epoch == start_epoch or ss_prob > 0:
            print(f"  scheduled_sampling: prob={ss_prob:.3f} (max={ss_max}, warmup={ss_warmup})",
                  flush=True)

        optimizer.zero_grad()

        for batch_i, (X_b, y_b) in enumerate(train_loader):
            X_b = X_b.to(device, non_blocking=True)   # [B, T, F]
            y_b = y_b.to(device, non_blocking=True)   # [B, T, L]
            # v6: prev_labels are re-enabled (no longer zeroed)

            # Skip batches with corrupt input data before doing any GPU work
            if not torch.isfinite(X_b).all():
                print(f"  WARN: non-finite input at epoch {epoch} batch {batch_i} — skipping",
                      flush=True)
                continue

            btn_true = y_b[:, :, :BUTTON_DIM]
            stk_bins = float_to_bin(y_b[:, :, BUTTON_DIM:])
            action_true = labels_to_action_idx(btn_true)          # [B, T] int64

            # ── Scheduled sampling: two-pass when active ──────────────────────
            # Pass 1 (no grad): get model predictions with ground-truth inputs.
            # Then mix predicted prev_labels into the input for pass 2.
            if ss_prob > 0:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    action_logits_p1, stk_logits_p1, _ = model(X_b, btn_true, stk_bins)
                pred_prev = build_predicted_prev_labels(action_logits_p1, stk_logits_p1)
                X_b = apply_scheduled_sampling(X_b, pred_prev, ss_prob,
                                               norm_mean_t, norm_std_t)

            with torch.amp.autocast("cuda", enabled=use_amp):
                action_logits, stk_logits, _ = model(X_b, btn_true, stk_bins)
                B, T = X_b.shape[:2]
                # Cast logits to fp32 before loss — bf16 logits can overflow in exp()
                # inside CrossEntropyLoss, producing NaN.
                act_ce = action_ce_fn(action_logits.float().reshape(B * T, NUM_ACTIONS),
                                      action_true.reshape(B * T))
                stk_ce = sum(
                    ce_fns[i](stk_logits[i].float().reshape(-1, STICK_BINS),
                               stk_bins[:, :, i].reshape(-1))
                    for i in range(STICK_DIM)
                ) / STICK_DIM
                loss = (act_ce + stk_ce) / grad_accum

            # Check for NaN/Inf BEFORE backward — accumulating NaN gradients
            # poisons model weights even when the optimizer step is skipped.
            if not torch.isfinite(loss * grad_accum):
                print(f"  WARN: non-finite loss at epoch {epoch} batch {batch_i} "
                      f"(act_ce={act_ce.item():.4f} stk_ce={stk_ce.item():.4f}) — skipping",
                      flush=True)
                optimizer.zero_grad()
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_act_ce += act_ce.item()
            epoch_stk_ce += stk_ce.item()
            n_batches    += 1

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
                      f"act_ce={epoch_act_ce / n_batches:.4f}  "
                      f"stk_ce={epoch_stk_ce / n_batches:.4f}  "
                      f"lr={cur_lr:.2e}",
                      flush=True)

        train_act_ce = epoch_act_ce / max(n_batches, 1)
        train_stk_ce = epoch_stk_ce / max(n_batches, 1)
        train_loss   = train_act_ce + train_stk_ce
        elapsed      = time.time() - t0
        print(f"\nEpoch {epoch:3d}/{epochs}  ({elapsed:.0f}s)")
        print(f"  [train] loss={train_loss:.4f}  act_ce={train_act_ce:.4f}  stk_ce={train_stk_ce:.4f}")

        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        if do_eval:
            val_m   = evaluate(val_loader, model, action_ce_fn, stk_weights, device, use_amp)
            score   = composite_score(val_m)
            is_best = val_m["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_m["loss"]
                best_epoch    = epoch
                torch.save(model.state_dict(), out / "best_model.pt")

            # Full checkpoint every 5 epochs (v6: includes optimizer + scheduler state)
            if epoch % 5 == 0 or epoch == epochs:
                ckpt = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                }
                ckpt_path = out / f"checkpoint_epoch{epoch}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")
            gap      = val_m["loss"] - train_loss
            gap_flag = "  << overfitting" if gap > 0.05 else ""
            print_eval("val", val_m)
            print(f"  gap(val-train)={gap:+.4f}{gap_flag}  "
                  f"composite={score:.4f}  "
                  f"best val_loss={best_val_loss:.4f} (epoch {best_epoch})"
                  + (" <- best" if is_best else ""))
            f1 = val_m["f1"]
            csv_row = ([epoch, train_loss, train_act_ce,
                        val_m["loss"], val_m["act_ce"], gap, score]
                       + [f1.get(n) for n in BUTTON_NAMES]
                       + [val_m["stick_err_deg"], val_m["idle_acc"]])
        else:
            print(f"  [val skipped — eval_every={eval_every}]")
            csv_row = ([epoch, train_loss, train_act_ce,
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
    test_m = evaluate(test_loader, model, action_ce_fn, stk_weights, device, use_amp)
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
    parser.add_argument("--ss-max",       type=float, default=0.5,
                        help="Scheduled sampling max probability (default 0.5; 0 to disable)")
    parser.add_argument("--ss-warmup",    type=int,   default=3,
                        help="Epochs of pure teacher forcing before scheduled sampling ramps up (default 3)")

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
        ss_max       = args.ss_max,
        ss_warmup    = args.ss_warmup,
    )
