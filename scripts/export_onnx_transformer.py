#!/usr/bin/env python3
"""export_onnx_transformer.py — Export trained CitrusTransformerBC (v6) to stateful ONNX.

The exported model processes one game frame at a time.  It accepts the KV cache
(past keys/values for all temporal transformer layers) as an explicit input and
returns the updated cache as an output.  AIController.cpp feeds this back on
every OnFrameEnd() call, maintaining temporal context across the match.

ONNX I/O:
    Inputs:
        features     [1, 194]              — single game frame (raw, unnormalized)
        kv_cache_in  [3, 2, 1, 127, 512]   — past KV state (layers, k/v, batch, seq, dim)
    Outputs:
        btn_probs    [1, 7]                — button flags (0.0 or 1.0)
                                             A B X Y lob_pass chip_shot R
        stick_vals   [1, 4]                — stick values in [-1,1] (Gumbel-max sampled)
        kv_cache_out [3, 2, 1, 127, 512]   — updated KV state (oldest frame evicted)

Usage:
    python export_onnx_transformer.py best_model.pt best_model.onnx
    python export_onnx_transformer.py best_model.pt best_model.onnx --norm-stats norm_stats.npz

Verify with:
    python -c "
    import onnxruntime as ort, numpy as np
    sess = ort.InferenceSession('best_model.onnx')
    feeds = {
        'features':    np.zeros((1, 194), dtype=np.float32),
        'kv_cache_in': np.zeros((3, 2, 1, 127, 512), dtype=np.float32),
    }
    btn, stk, kv = sess.run(None, feeds)
    print('btn:', btn.shape, 'stk:', stk.shape, 'kv:', kv.shape)
    "
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants (must stay in sync with train_transformer.py v6) ───────────────

SEQ_LEN        = 128
FEATURE_DIM    = 194
BUTTON_DIM     = 7
STICK_DIM      = 4
STICK_BINS     = 21
ENTITY_RAW_DIM = 32
N_ENTITIES     = 20
N_ENTITY_TYPES = 11

# Action state embedding dims
STRIKER_STATE_VOCAB = 30
GOALIE_STATE_VOCAB  = 27
STATE_EMBED_DIM     = 8
FIELD_ITEM_VOCAB    = 10
ITEM_EMBED_DIM      = 4

NUM_ACTIONS     = 12
ACTION_VOCAB    = torch.tensor([
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

ENTITY_DIM      = 256
ENTITY_LAYERS   = 2
ENTITY_HEADS    = 4
TEMPORAL_DIM    = 512
TEMPORAL_LAYERS = 3
TEMPORAL_HEADS  = 4
FF_MULT         = 2

# Entity token layout (v6)
_ENTITY_DEFS: List[Tuple[int, int, int]] = [
    (  0,  22, 0),  # ball: pos×3 vel×3 charge pp b2g×3 + owner_oh×11
    ( 22,  41, 1),  # self: pos_delta×3 state_idx×1 heading×2 goal×3 effect×5 spd×3 timer carrier
    ( 41,  48, 2),  # friendly striker 0
    ( 48,  55, 2),  # friendly striker 1
    ( 55,  62, 2),  # friendly striker 2
    ( 62,  66, 3),  # friendly goalie
    ( 66,  73, 4),  # enemy striker 0
    ( 73,  80, 4),  # enemy striker 1
    ( 80,  87, 4),  # enemy striker 2
    ( 87,  94, 4),  # enemy striker 3
    ( 94,  98, 5),  # enemy goalie
    ( 98, 109, 6),  # own inventory 0
    (109, 120, 6),  # own inventory 1
    (120, 131, 7),  # enemy inventory 0
    (131, 142, 7),  # enemy inventory 1
    (142, 150, 8),  # field item 0
    (150, 158, 8),  # field item 1
    (158, 166, 8),  # field item 2
    (166, 174, 8),  # field item 3
    (174, 194, 9),  # context: tactical×5 possession×2 phase×2 prev_action×11
]
_ENTITY_TYPE_IDS = [t for _, _, t in _ENTITY_DEFS]

# Offsets within entity tokens where the integer index lives
_SELF_STATE_IDX_OFFSET    = 3
_STRIKER_STATE_IDX_OFFSET = 3
_GOALIE_STATE_IDX_OFFSET  = 3
_ITEM_TYPE_IDX_OFFSET     = 0


# ── Architecture (replicated from train_transformer.py v6) ──────────────────

class _FFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * FF_MULT), nn.GELU(), nn.Linear(dim * FF_MULT, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EntityTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn   = _FFN(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        a, _ = self.attn(x32, x32, x32)
        a = a.to(orig_dtype)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class EntityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.striker_state_embed = nn.Embedding(STRIKER_STATE_VOCAB, STATE_EMBED_DIM)
        self.goalie_state_embed  = nn.Embedding(GOALIE_STATE_VOCAB,  STATE_EMBED_DIM)
        self.item_type_embed     = nn.Embedding(FIELD_ITEM_VOCAB,    ITEM_EMBED_DIM)

        self._effective_max = ENTITY_RAW_DIM + STATE_EMBED_DIM
        self.proj       = nn.Linear(self._effective_max, ENTITY_DIM)
        self.type_embed = nn.Embedding(N_ENTITY_TYPES, ENTITY_DIM)
        self.layers     = nn.ModuleList([
            EntityTransformerLayer(ENTITY_DIM, ENTITY_HEADS)
            for _ in range(ENTITY_LAYERS)
        ])
        self.out = nn.Linear(ENTITY_DIM, TEMPORAL_DIM)
        type_ids = torch.tensor(_ENTITY_TYPE_IDS, dtype=torch.long)
        self.register_buffer("type_ids", type_ids)

    def _embed_state(self, token: torch.Tensor, idx_offset: int,
                     embed: nn.Embedding) -> torch.Tensor:
        idx = token[:, idx_offset].long().clamp(0, embed.num_embeddings - 1)
        emb = embed(idx)
        before = token[:, :idx_offset]
        after  = token[:, idx_offset + 1:]
        return torch.cat([before, emb, after], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        processed = []
        for i, (s, e, etype) in enumerate(_ENTITY_DEFS):
            token = x[:, i, :e - s]

            if etype == 1:
                token = self._embed_state(token, _SELF_STATE_IDX_OFFSET,
                                           self.striker_state_embed)
            elif etype == 2 or etype == 4:
                token = self._embed_state(token, _STRIKER_STATE_IDX_OFFSET,
                                           self.striker_state_embed)
            elif etype == 3 or etype == 5:
                token = self._embed_state(token, _GOALIE_STATE_IDX_OFFSET,
                                           self.goalie_state_embed)
            elif etype == 8:
                token = self._embed_state(token, _ITEM_TYPE_IDX_OFFSET,
                                           self.item_type_embed)

            if token.shape[1] < self._effective_max:
                pad = token.new_zeros(B, self._effective_max - token.shape[1])
                token = torch.cat([token, pad], dim=1)
            processed.append(token)

        x = torch.stack(processed, dim=1)
        x = self.proj(x) + self.type_embed(self.type_ids)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.out(x)


class TemporalLayer(nn.Module):
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

    def _split_heads(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.dim)

    def forward_causal(self, x):
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

    def forward_cached(self, x, k_cache, v_cache):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        k_all = torch.cat([self._split_heads(k_cache), k], dim=2)
        v_all = torch.cat([self._split_heads(v_cache), v], dim=2)
        scale = math.sqrt(self.head_dim)
        attn  = (q @ k_all.transpose(-2, -1)) / scale
        attn  = attn.softmax(dim=-1)
        out   = self._merge_heads(attn @ v_all)
        out   = self.out_proj(out)
        x     = self.norm1(x + out)
        x     = self.norm2(x + self.ffn(x))
        k_full = self._merge_heads(k_all)
        v_full = self._merge_heads(v_all)
        return x, k_full[:, 1:, :], v_full[:, 1:, :]

    def forward_with_initial_kv(self, x, k_cache, v_cache):
        # Parallel forward over a contiguous segment with a prepended KV
        # cache.  Output for query t is mathematically identical to running
        # forward_cached() frame-by-frame: query t attends to the sliding
        # window [t, S_cache + t] of K_full = concat(k_cache, K_new), which
        # is exactly the 128-token cache that forward_cached would have at
        # frame t.
        S = k_cache.size(1)
        T = x.size(1)
        q = self._split_heads(self.q_proj(x))
        k_new = self._split_heads(self.k_proj(x))
        v_new = self._split_heads(self.v_proj(x))
        k_all = torch.cat([self._split_heads(k_cache), k_new], dim=2)
        v_all = torch.cat([self._split_heads(v_cache), v_new], dim=2)
        device = x.device
        key_idx = torch.arange(S + T, device=device)
        query_idx = torch.arange(T, device=device)
        attn_mask = (key_idx.unsqueeze(0) >= query_idx.unsqueeze(1)) & \
                    (key_idx.unsqueeze(0) <= S + query_idx.unsqueeze(1))
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class CausalTemporalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalLayer(TEMPORAL_DIM, TEMPORAL_HEADS) for _ in range(TEMPORAL_LAYERS)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward_causal(x)
        return x

    def forward_cached(self, frame_emb, kv_cache):
        x = frame_emb.unsqueeze(1)
        new_kv = []
        for i, layer in enumerate(self.layers):
            x, k_new, v_new = layer.forward_cached(x, kv_cache[i, 0], kv_cache[i, 1])
            new_kv.append(torch.stack([k_new, v_new], dim=0))
        return x.squeeze(1), torch.stack(new_kv, dim=0)

    def forward_with_initial_kv(self, emb, initial_kv):
        # emb: [1, T, D] segment of frame embeddings.
        # initial_kv: [L, 2, 1, S-1, D] KV state preceding the segment.
        # Returns: [1, T, D] policy outputs, equivalent to T sequential
        # forward_cached calls starting from initial_kv (no new cache
        # returned — replay doesn't carry context across rollouts).
        x = emb
        for i, layer in enumerate(self.layers):
            x = layer.forward_with_initial_kv(x, initial_kv[i, 0], initial_kv[i, 1])
        return x


class ConditionalControllerHead(nn.Module):
    def __init__(self, input_dim: int, stick_bins: int = STICK_BINS):
        super().__init__()
        self.stick_bins = stick_bins
        self.action_head = nn.Linear(input_dim, NUM_ACTIONS)
        stk_input_dim    = input_dim + NUM_ACTIONS
        self.stk_heads   = nn.ModuleList([nn.Linear(stk_input_dim, stick_bins)
                                           for _ in range(STICK_DIM)])

    def forward_infer(self, policy_t: torch.Tensor):
        action_logits = self.action_head(policy_t)        # [B, NUM_ACTIONS]
        action_probs  = torch.softmax(action_logits, dim=-1)

        # Gumbel-max to sample action index (ONNX-compatible)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(action_logits) + 1e-20) + 1e-20)
        action_idx   = (action_logits + gumbel_noise).argmax(dim=-1)  # [B]
        vocab        = ACTION_VOCAB.to(action_logits.device)
        btn_flags    = vocab[action_idx]                   # [B, 7] — 0/1 floats

        # Condition sticks on action softmax (differentiable; matches training)
        stk_input = torch.cat([policy_t, action_probs], dim=-1)
        stick_vals = torch.stack([
            self._gumbel_sample(self.stk_heads[i](stk_input))
            for i in range(STICK_DIM)
        ], dim=-1)  # [B, 4] continuous values in [-1, 1]
        return btn_flags, stick_vals

    @staticmethod
    def _gumbel_sample(logits: torch.Tensor) -> torch.Tensor:
        """Sample a bin index via Gumbel-max and convert to [-1, 1] float."""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        bin_idx = (logits + gumbel_noise).argmax(dim=-1)          # [B]
        return bin_idx.float() / (STICK_BINS - 1) * 2.0 - 1.0    # [B] in [-1, 1]


class CitrusTransformerBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.entity_encoder = EntityEncoder()
        self.temporal       = CausalTemporalTransformer()
        self.ctrl_head      = ConditionalControllerHead(TEMPORAL_DIM)
        self.value_head     = nn.Linear(TEMPORAL_DIM, 1)

    def forward_infer(self, entities, kv_cache):
        frame_emb             = self.entity_encoder(entities)
        out, kv_cache_out     = self.temporal.forward_cached(frame_emb, kv_cache)
        btn_probs, stick_vals = self.ctrl_head.forward_infer(out)
        return btn_probs, stick_vals, kv_cache_out


# ── ONNX-friendly flat→entities ───────────────────────────────────────────────

def _flat_to_entities_onnx(x: torch.Tensor) -> torch.Tensor:
    """Convert flat feature vector to entity token matrix without in-place ops.

    x: [1, FEATURE_DIM]
    returns: [1, N_ENTITIES, ENTITY_RAW_DIM]
    """
    tokens = []
    for s, e, _ in _ENTITY_DEFS:
        raw = x[:, s:e]                        # [1, width]
        pad = ENTITY_RAW_DIM - (e - s)
        if pad > 0:
            raw = F.pad(raw, (0, pad))         # [1, ENTITY_RAW_DIM]
        tokens.append(raw.unsqueeze(1))        # [1, 1, ENTITY_RAW_DIM]
    return torch.cat(tokens, dim=1)            # [1, N_ENTITIES, ENTITY_RAW_DIM]


# ── Single-frame ONNX wrapper ─────────────────────────────────────────────────

class _TransformerSingleFrameWrapper(nn.Module):
    """Wraps CitrusTransformerBC for single-frame stateful ONNX export.

    Takes flat features [1, FEATURE_DIM] and kv_cache [3, 2, 1, 127, 512].
    Internally runs flat_to_entities → entity_encoder → temporal (cached) → ctrl head.
    Normalization stats are baked in as constants if provided.
    """

    def __init__(self, model: CitrusTransformerBC,
                 norm_mean: torch.Tensor = None,
                 norm_std:  torch.Tensor = None):
        super().__init__()
        self.model = model
        if norm_mean is not None and norm_std is not None:
            self.register_buffer("norm_mean", norm_mean.view(1, -1))
            self.register_buffer("norm_std",  norm_std.view(1, -1))
        else:
            self.norm_mean = None
            self.norm_std  = None

    def forward(self, features: torch.Tensor, kv_cache_in: torch.Tensor):
        """
        features:    [1, FEATURE_DIM]           — raw (unnormalized) feature vector
        kv_cache_in: [TEMPORAL_LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]

        Returns:
            btn_probs    [1, BUTTON_DIM]                          — 0/1 button flags
            stick_vals   [1, STICK_DIM]                           — values in [-1, 1]
            kv_cache_out [TEMPORAL_LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]
        """
        if self.norm_mean is not None:
            features = (features - self.norm_mean) / self.norm_std

        entities = _flat_to_entities_onnx(features)   # [1, N_ENTITIES, ENTITY_RAW_DIM]
        btn_probs, stick_vals, kv_cache_out = self.model.forward_infer(entities, kv_cache_in)

        return btn_probs, stick_vals, kv_cache_out


# ── Export ────────────────────────────────────────────────────────────────────

def export(weights_path: str, output_path: str, norm_stats_path: str = None) -> None:
    device = torch.device("cpu")

    model = CitrusTransformerBC().to(device)
    state = torch.load(weights_path, map_location=device)
    # Support both raw state_dict (best_model.pt) and full checkpoint dicts
    if isinstance(state, dict) and "model" in state:
        print(f"Loading from full checkpoint (epoch {state.get('epoch', '?')})")
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {weights_path}")

    norm_mean = norm_std = None
    if norm_stats_path:
        stats = np.load(norm_stats_path)
        norm_mean = torch.from_numpy(stats["mean"].astype("float32"))
        norm_std  = torch.from_numpy(stats["std"].astype("float32"))
        print(f"Loaded norm stats from {norm_stats_path} — baking into ONNX graph")
    else:
        print("WARNING: no --norm-stats provided — exporting without normalization")

    wrapper = _TransformerSingleFrameWrapper(model, norm_mean, norm_std).to(device)
    wrapper.eval()

    dummy_features = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
    dummy_kv       = torch.zeros(TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM,
                                 dtype=torch.float32)

    with torch.no_grad():
        btn, stk, kv_out = wrapper(dummy_features, dummy_kv)

    print(f"Forward pass OK: btn={tuple(btn.shape)}  stk={tuple(stk.shape)}"
          f"  kv_out={tuple(kv_out.shape)}")

    assert btn.shape    == (1, BUTTON_DIM),                                   f"btn:    {btn.shape}"
    assert stk.shape    == (1, STICK_DIM),                                    f"stk:    {stk.shape}"
    assert kv_out.shape == (TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM), f"kv_out: {kv_out.shape}"
    assert float(stk.min()) >= -1.0 and float(stk.max()) <= 1.0, \
        f"stick_vals out of range: [{stk.min():.3f}, {stk.max():.3f}]"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_kv),
        output_path,
        export_params       = True,
        opset_version       = 17,
        do_constant_folding = True,
        input_names         = ["features", "kv_cache_in"],
        output_names        = ["btn_probs", "stick_vals", "kv_cache_out"],
    )
    print(f"Exported ONNX model to {output_path}")

    # Validate with onnxruntime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        in_names  = [inp.name for inp in sess.get_inputs()]
        out_names = [out.name for out in sess.get_outputs()]
        print(f"ORT inputs:  {in_names}")
        print(f"ORT outputs: {out_names}")
        assert in_names  == ["features", "kv_cache_in"],                      f"inputs:  {in_names}"
        assert out_names == ["btn_probs", "stick_vals", "kv_cache_out"],      f"outputs: {out_names}"

        feeds = {
            "features":    np.zeros((1, FEATURE_DIM), dtype=np.float32),
            "kv_cache_in": np.zeros((TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM),
                                    dtype=np.float32),
        }
        btn_out, stk_out, kv_out = sess.run(None, feeds)
        print("onnxruntime validation OK:")
        print(f"  btn_probs    {btn_out.shape}  range=[{btn_out.min():.4f}, {btn_out.max():.4f}]")
        print(f"  stick_vals   {stk_out.shape}  range=[{stk_out.min():.4f}, {stk_out.max():.4f}]")
        print(f"  kv_cache_out {kv_out.shape}")

        # Stateful check: cache changes after feeding non-zero input
        feeds2 = {
            "features":    np.random.randn(1, FEATURE_DIM).astype(np.float32),
            "kv_cache_in": kv_out,
        }
        btn2, stk2, kv2 = sess.run(None, feeds2)
        cache_changed = not np.allclose(kv2, kv_out)
        print(f"  Stateful check (kv_cache changes after step 2): "
              f"{'PASS' if cache_changed else 'FAIL'}")

    except ImportError:
        print("WARNING: onnxruntime not installed — skipping validation.")
        print("  pip install onnxruntime  # then re-run to validate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export CitrusTransformerBC (v6) to stateful ONNX")
    parser.add_argument("weights",      help="Path to best_model.pt (PyTorch state dict)")
    parser.add_argument("output",       help="Output path for .onnx file")
    parser.add_argument("--norm-stats", default=None, metavar="NPZ",
                        help="Path to norm_stats.npz from train_transformer.py; "
                             "bakes normalization into the ONNX graph")
    args = parser.parse_args()
    export(args.weights, args.output, args.norm_stats)
