#!/usr/bin/env python3
"""export_onnx_transformer.py — Export trained CitrusTransformerBC to stateful ONNX.

The exported model processes one game frame at a time.  It accepts the KV cache
(past keys/values for all temporal transformer layers) as an explicit input and
returns the updated cache as an output.  AIController.cpp feeds this back on
every OnFrameEnd() call, maintaining temporal context across the match.

ONNX I/O:
    Inputs:
        features     [1, 442]          — single game frame (raw, unnormalized)
        kv_cache_in  [3, 2, 1, 63, 512] — past KV state (layers, k/v, batch, seq, dim)
    Outputs:
        btn_probs    [1, 6]             — button probabilities (sigmoid, A B X Y L R)
        stick_vals   [1, 4]             — stick values in [-1,1] (AR bins→float)
        kv_cache_out [3, 2, 1, 63, 512] — updated KV state (oldest frame evicted)

Usage:
    python export_onnx_transformer.py best_model.pt best_model.onnx
    python export_onnx_transformer.py best_model.pt best_model.onnx --norm-stats norm_stats.npz

Verify with:
    python -c "
    import onnxruntime as ort, numpy as np
    sess = ort.InferenceSession('best_model.onnx')
    feeds = {
        'features':    np.zeros((1, 442), dtype=np.float32),
        'kv_cache_in': np.zeros((3, 2, 1, 63, 512), dtype=np.float32),
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

# ── Constants (must stay in sync with train_transformer.py) ───────────────────

SEQ_LEN        = 64
FEATURE_DIM    = 442
BUTTON_DIM     = 6
STICK_DIM      = 4
STICK_BINS     = 21
ENTITY_RAW_DIM = 64
N_ENTITIES     = 16
N_ENTITY_TYPES = 9
ENTITY_DIM     = 256
ENTITY_LAYERS  = 2
ENTITY_HEADS   = 4
TEMPORAL_DIM   = 512
TEMPORAL_LAYERS = 3
TEMPORAL_HEADS  = 4
FF_MULT        = 2

_ENTITY_DEFS: List[Tuple[int, int, int]] = [
    (  0,  19, 0),
    ( 19,  67, 1),
    ( 67, 103, 2),
    (103, 139, 2),
    (139, 175, 2),
    (175, 205, 3),
    (205, 241, 4),
    (241, 277, 4),
    (277, 313, 4),
    (313, 349, 4),
    (349, 379, 5),
    (379, 390, 6),
    (390, 401, 6),
    (401, 412, 7),
    (412, 423, 7),
    (423, 442, 8),
]
_ENTITY_TYPE_IDS = [t for _, _, t in _ENTITY_DEFS]


# ── Architecture (replicated from train_transformer.py) ───────────────────────

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
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class EntityEncoder(nn.Module):
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
        self.register_buffer("type_ids", type_ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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


class IndependentControllerHead(nn.Module):
    def __init__(self, input_dim: int, stick_bins: int = STICK_BINS):
        super().__init__()
        self.stick_bins = stick_bins
        self.btn_head   = nn.Linear(input_dim, BUTTON_DIM)
        self.stk_heads  = nn.ModuleList([nn.Linear(input_dim, stick_bins) for _ in range(STICK_DIM)])

    def forward_infer(self, policy_t):
        btn_probs = torch.sigmoid(self.btn_head(policy_t))  # [B, BUTTON_DIM]
        stk_bins  = torch.stack(
            [self.stk_heads[i](policy_t).argmax(dim=-1) for i in range(STICK_DIM)],
            dim=-1)  # [B, STICK_DIM]
        return btn_probs, stk_bins


class CitrusTransformerBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.entity_encoder = EntityEncoder()
        self.temporal       = CausalTemporalTransformer()
        self.ctrl_head      = IndependentControllerHead(TEMPORAL_DIM)
        self.value_head     = nn.Linear(TEMPORAL_DIM, 1)

    def forward_infer(self, entities, kv_cache):
        frame_emb           = self.entity_encoder(entities)
        out, kv_cache_out   = self.temporal.forward_cached(frame_emb, kv_cache)
        btn_probs, stk_bins = self.ctrl_head.forward_infer(out)
        return btn_probs, stk_bins, kv_cache_out


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

    Takes flat features [1, FEATURE_DIM] and kv_cache [3, 2, 1, 63, 512].
    Internally runs flat_to_entities → entity_encoder → temporal (cached) → AR head.
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
            btn_probs    [1, BUTTON_DIM]                          — sigmoid probabilities
            stick_vals   [1, STICK_DIM]                           — values in [-1, 1]
            kv_cache_out [TEMPORAL_LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]
        """
        if self.norm_mean is not None:
            features = (features - self.norm_mean) / self.norm_std

        entities = _flat_to_entities_onnx(features)   # [1, N_ENTITIES, ENTITY_RAW_DIM]
        btn_probs, stk_bins, kv_cache_out = self.model.forward_infer(entities, kv_cache_in)

        # Convert discrete bin indices → continuous float in [-1, 1]
        stick_vals = stk_bins.float() / (STICK_BINS - 1) * 2.0 - 1.0  # [1, STICK_DIM]

        return btn_probs, stick_vals, kv_cache_out


# ── Export ────────────────────────────────────────────────────────────────────

def export(weights_path: str, output_path: str, norm_stats_path: str = None) -> None:
    device = torch.device("cpu")

    model = CitrusTransformerBC().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {weights_path}")

    norm_mean = norm_std = None
    if norm_stats_path:
        stats = np.load(norm_stats_path)
        # norm_stats were computed on float32 X; load as float32
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
        print(f"  stick_vals   {stk_out.shape}  range=[{stk_out.min():.4f}, {stk_out.max():.4f}]"
              f"  ({STICK_BINS} bins)")
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
        description="Export CitrusTransformerBC to stateful ONNX")
    parser.add_argument("weights",      help="Path to best_model.pt (PyTorch state dict)")
    parser.add_argument("output",       help="Output path for .onnx file")
    parser.add_argument("--norm-stats", default=None, metavar="NPZ",
                        help="Path to norm_stats.npz from train_transformer.py; "
                             "bakes normalization into the ONNX graph")
    args = parser.parse_args()
    export(args.weights, args.output, args.norm_stats)
