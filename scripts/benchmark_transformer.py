#!/usr/bin/env python3
"""benchmark_transformer.py -- Measure inference latency for the proposed
two-level Citrus transformer architecture (entity encoder + temporal transformer).

Uses random weights — no training required. The ONNX Runtime number is the one
that matters for deployment in Dolphin.

Architecture (from proposed-citrus-ai-notes.md):
  EntityEncoder:       2 layers, 256 dim, 4 heads
                       Input:  [1, N_entities, ENTITY_RAW_DIM]
                       Output: [1, TEMPORAL_DIM]  (mean-pooled, projected up)
  TemporalTransformer: 3 layers, 512 dim, 4 heads, causal with KV cache
                       Input:  [1, TEMPORAL_DIM] (new frame) + KV cache
                       Output: [1, TEMPORAL_DIM] + updated KV cache
  ControllerHead:      6 buttons (logits) + 4 sticks × 21 bins (logits)

KV cache format: [LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]
  dim 0: layer index
  dim 1: 0=K, 1=V
  dim 2: batch (always 1 at inference)
  dim 3: cached sequence length (SEQ_LEN-1 frames in steady state)
  dim 4: TEMPORAL_DIM

Usage:
    python benchmark_transformer.py
    python benchmark_transformer.py --num-entities 40 --num-runs 200
    python benchmark_transformer.py --skip-onnx
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Architecture constants (from proposed-citrus-ai-notes.md)
# ---------------------------------------------------------------------------

ENTITY_RAW_DIM  = 64    # raw features per entity token before projection
                         # (actual dims vary per entity type; 64 is a safe upper bound)
ENTITY_DIM      = 256   # entity encoder hidden dim
ENTITY_LAYERS   = 2
ENTITY_HEADS    = 4

TEMPORAL_DIM    = 512   # temporal transformer hidden dim
TEMPORAL_LAYERS = 3
TEMPORAL_HEADS  = 4
FF_MULT         = 2

SEQ_LEN         = 64    # temporal sliding window (frames)
NUM_STICK_BINS  = 21
BUTTON_DIM      = 6
STICK_DIM       = 4

FRAME_MS        = 1000.0 / 60.0  # ~16.67 ms per frame at 60 fps


# ---------------------------------------------------------------------------
# Entity Encoder
# ---------------------------------------------------------------------------

class _FFN(nn.Module):
    def __init__(self, dim: int, mult: int = FF_MULT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EntityTransformerLayer(nn.Module):
    """Standard (non-causal) transformer layer for within-frame entity attention."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn   = _FFN(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + a)
        x    = self.norm2(x + self.ffn(x))
        return x


class EntityEncoder(nn.Module):
    """Per-frame spatial encoder over entity tokens.

    Input:  [B, N_entities, ENTITY_RAW_DIM]
    Output: [B, TEMPORAL_DIM]
    """

    def __init__(self,
                 raw_dim:      int = ENTITY_RAW_DIM,
                 entity_dim:   int = ENTITY_DIM,
                 temporal_dim: int = TEMPORAL_DIM,
                 layers:       int = ENTITY_LAYERS,
                 heads:        int = ENTITY_HEADS):
        super().__init__()
        self.proj    = nn.Linear(raw_dim, entity_dim)
        self.layers  = nn.ModuleList([
            EntityTransformerLayer(entity_dim, heads) for _ in range(layers)
        ])
        self.out     = nn.Linear(entity_dim, temporal_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)           # [B, N, entity_dim]
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)          # [B, entity_dim]  — mean pool over entities
        return self.out(x)         # [B, temporal_dim]


# ---------------------------------------------------------------------------
# Temporal Transformer with KV Cache
# ---------------------------------------------------------------------------

class CachedTemporalLayer(nn.Module):
    """Transformer layer with explicit KV cache for O(1) incremental inference.

    KV cache stored as flat tensors [B, S, dim] and reshaped to
    [B, heads, S, head_dim] internally for attention.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads    = heads
        self.head_dim = dim // heads
        self.dim      = dim
        self.scale    = math.sqrt(self.head_dim)

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn      = _FFN(dim)
        self.norm1    = nn.LayerNorm(dim)
        self.norm2    = nn.LayerNorm(dim)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, D] -> [B, H, S, D/H]"""
        B, S, _ = x.shape
        return x.view(B, S, self.heads, self.head_dim).transpose(1, 2)

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, S, D/H] -> [B, S, D]"""
        B, H, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.dim)

    def forward_cached(self,
                       x:       torch.Tensor,
                       k_cache: torch.Tensor,
                       v_cache: torch.Tensor):
        """Incremental single-frame inference.

        x:       [B, 1, dim]         — new frame
        k_cache: [B, S, dim]         — past S frames' keys   (S = SEQ_LEN-1 steady state)
        v_cache: [B, S, dim]         — past S frames' values

        Returns:
            out:         [B, 1, dim]
            k_cache_new: [B, S, dim]  — oldest frame evicted, new frame appended
            v_cache_new: [B, S, dim]
        """
        B = x.shape[0]

        q = self._to_heads(self.q_proj(x))          # [B, H, 1,   D/H]
        k = self._to_heads(self.k_proj(x))          # [B, H, 1,   D/H]
        v = self._to_heads(self.v_proj(x))          # [B, H, 1,   D/H]

        # Concat new K/V with cache -> [B, H, S+1, D/H]
        k_all = torch.cat([self._to_heads(k_cache), k], dim=2)
        v_all = torch.cat([self._to_heads(v_cache), v], dim=2)

        attn  = (q @ k_all.transpose(-2, -1)) / self.scale  # [B, H, 1, S+1]
        attn  = attn.softmax(dim=-1)
        out   = self._from_heads(attn @ v_all)               # [B, 1, dim]
        out   = self.out_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))

        # Evict oldest frame to keep cache at SEQ_LEN-1
        k_new_full = self._from_heads(k_all)                 # [B, S+1, dim]
        v_new_full = self._from_heads(v_all)                 # [B, S+1, dim]
        k_cache_new = k_new_full[:, 1:, :]                   # drop oldest [B, S, dim]
        v_cache_new = v_new_full[:, 1:, :]

        return x, k_cache_new, v_cache_new


class TemporalTransformer(nn.Module):
    """3-layer causal temporal transformer with KV cache.

    KV cache: [LAYERS, 2, B, SEQ_LEN-1, TEMPORAL_DIM]
    """

    def __init__(self,
                 dim:    int = TEMPORAL_DIM,
                 layers: int = TEMPORAL_LAYERS,
                 heads:  int = TEMPORAL_HEADS):
        super().__init__()
        self.dim     = dim
        self.n_layers = layers
        self.encoder = nn.ModuleList([
            CachedTemporalLayer(dim, heads) for _ in range(layers)
        ])

    def forward_cached(self,
                       frame_emb: torch.Tensor,
                       kv_cache:  torch.Tensor):
        """Incremental inference.

        frame_emb: [B, TEMPORAL_DIM]
        kv_cache:  [LAYERS, 2, B, SEQ_LEN-1, TEMPORAL_DIM]

        Returns:
            out:         [B, TEMPORAL_DIM]
            kv_cache_out [LAYERS, 2, B, SEQ_LEN-1, TEMPORAL_DIM]
        """
        x = frame_emb.unsqueeze(1)   # [B, 1, dim]

        new_kv_list = []
        for i, layer in enumerate(self.encoder):
            k_cache = kv_cache[i, 0]   # [B, S, dim]
            v_cache = kv_cache[i, 1]   # [B, S, dim]
            x, k_new, v_new = layer.forward_cached(x, k_cache, v_cache)
            new_kv_list.append(torch.stack([k_new, v_new], dim=0))  # [2, B, S, dim]

        kv_cache_out = torch.stack(new_kv_list, dim=0)   # [LAYERS, 2, B, S, dim]
        return x.squeeze(1), kv_cache_out


# ---------------------------------------------------------------------------
# ONNX-exportable wrapper
# ---------------------------------------------------------------------------

class CitrusInferenceModel(nn.Module):
    """Full inference model for ONNX export.

    Inputs:
        entities  [1, N_entities, ENTITY_RAW_DIM]
        kv_cache  [LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]

    Outputs:
        btn_logits   [1, BUTTON_DIM]
        stk_logits   [1, STICK_DIM * NUM_STICK_BINS]
        kv_cache_out [LAYERS, 2, 1, SEQ_LEN-1, TEMPORAL_DIM]
    """

    def __init__(self,
                 entity_encoder: EntityEncoder,
                 temporal:       TemporalTransformer,
                 temporal_dim:   int = TEMPORAL_DIM):
        super().__init__()
        self.entity_encoder = entity_encoder
        self.temporal       = temporal
        self.btn_head       = nn.Linear(temporal_dim, BUTTON_DIM)
        self.stk_head       = nn.Linear(temporal_dim, STICK_DIM * NUM_STICK_BINS)

    def forward(self,
                entities: torch.Tensor,
                kv_cache: torch.Tensor):
        frame_emb            = self.entity_encoder(entities)
        out, kv_cache_out    = self.temporal.forward_cached(frame_emb, kv_cache)
        btn_logits           = self.btn_head(out)
        stk_logits           = self.stk_head(out)
        return btn_logits, stk_logits, kv_cache_out


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def _make_fake_inputs(n_entities: int, device: torch.device):
    entities = torch.randn(1, n_entities, ENTITY_RAW_DIM, device=device)
    kv_cache = torch.zeros(
        TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM, device=device
    )
    return entities, kv_cache


def _benchmark_pytorch(model: CitrusInferenceModel,
                        n_entities: int,
                        device: torch.device,
                        num_warmup: int,
                        num_runs: int) -> float:
    entities, kv_cache = _make_fake_inputs(n_entities, device)
    model.eval()

    print(f"  Warming up ({num_warmup} runs)...", flush=True)
    with torch.no_grad():
        for _ in range(num_warmup):
            _, _, kv_cache = model(entities, kv_cache)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"  Timing ({num_runs} runs)...", flush=True)
    entities, kv_cache = _make_fake_inputs(n_entities, device)  # fresh cache

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _, _, kv_cache = model(entities, kv_cache)
            if device.type == "cuda":
                torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_runs * 1000.0


def _export_onnx(model: CitrusInferenceModel,
                 n_entities: int,
                 device: torch.device,
                 path: str) -> bool:
    try:
        import torch.onnx
        entities, kv_cache = _make_fake_inputs(n_entities, device)
        model.eval()

        torch.onnx.export(
            model,
            (entities, kv_cache),
            path,
            input_names=["entities", "kv_cache"],
            output_names=["btn_logits", "stk_logits", "kv_cache_out"],
            dynamic_axes={
                "entities": {1: "n_entities"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        return True
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        return False


def _benchmark_onnx(onnx_path: str,
                    n_entities: int,
                    num_warmup: int,
                    num_runs: int) -> float:
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"  ORT providers: {sess.get_providers()}", flush=True)

    entities = np.random.randn(1, n_entities, ENTITY_RAW_DIM).astype(np.float32)
    kv_cache = np.zeros(
        (TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM), dtype=np.float32
    )

    feed = {"entities": entities, "kv_cache": kv_cache}

    print(f"  Warming up ({num_warmup} runs)...", flush=True)
    for _ in range(num_warmup):
        _, _, kv_cache = sess.run(None, feed)
        feed["kv_cache"] = kv_cache

    print(f"  Timing ({num_runs} runs)...", flush=True)
    feed["kv_cache"] = np.zeros(
        (TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM), dtype=np.float32
    )

    start = time.perf_counter()
    for _ in range(num_runs):
        _, _, kv_cache = sess.run(None, feed)
        feed["kv_cache"] = kv_cache
    elapsed = time.perf_counter() - start

    return elapsed / num_runs * 1000.0


def _report(label: str, avg_ms: float) -> None:
    delay_frames = int(avg_ms / FRAME_MS) + 1
    print(f"\n  {label}")
    print(f"    Average inference : {avg_ms:.2f} ms")
    print(f"    Recommended delay : {delay_frames} frame(s)"
          f"  ({delay_frames * FRAME_MS:.0f} ms budget)")
    if delay_frames <= 1:
        print(f"    Experience        : Excellent — essentially instant")
    elif delay_frames <= 3:
        print(f"    Experience        : Good — near-instant reactions")
    elif delay_frames <= 8:
        print(f"    Experience        : Acceptable — slightly perceptible lag")
    else:
        print(f"    Experience        : High-delay — model decisions lag noticeably")


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark proposed Citrus transformer architecture inference time"
    )
    parser.add_argument("--num-entities", type=int, default=30,
                        help="Entity tokens per frame for benchmark (default 30; range ~20-40)")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs",   type=int, default=100)
    parser.add_argument("--skip-onnx",  action="store_true",
                        help="Skip ONNX export and benchmark")
    parser.add_argument("--onnx-path",  type=str,
                        default="citrus_transformer_bench.onnx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Citrus Transformer Architecture Benchmark")
    print(f"{'='*60}")
    print(f"Device           : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Entity tokens    : {args.num_entities}  (typical range: 20-40)")
    print(f"Entity dim       : {ENTITY_DIM}  ({ENTITY_LAYERS} layers, {ENTITY_HEADS} heads)")
    print(f"Temporal dim     : {TEMPORAL_DIM}  ({TEMPORAL_LAYERS} layers, {TEMPORAL_HEADS} heads)")
    print(f"Temporal window  : {SEQ_LEN} frames  ({SEQ_LEN / 60:.1f}s at 60fps)")
    print(f"KV cache shape   : [{TEMPORAL_LAYERS}, 2, 1, {SEQ_LEN-1}, {TEMPORAL_DIM}]")
    print(f"Warmup / runs    : {args.num_warmup} / {args.num_runs}")

    # Build model
    entity_encoder = EntityEncoder()
    temporal       = TemporalTransformer()
    model          = CitrusInferenceModel(entity_encoder, temporal).to(device)
    print(f"\nParameter counts:")
    print(f"  EntityEncoder        : {_count_params(entity_encoder):>10,}")
    print(f"  TemporalTransformer  : {_count_params(temporal):>10,}")
    print(f"  ControllerHead       : {_count_params(model.btn_head) + _count_params(model.stk_head):>10,}")
    print(f"  Total                : {_count_params(model):>10,}")

    # PyTorch benchmark
    print(f"\n[1/2] PyTorch benchmark (fp32, {'CUDA' if device.type == 'cuda' else 'CPU'})")
    pt_ms = _benchmark_pytorch(model, args.num_entities, device,
                                args.num_warmup, args.num_runs)
    _report("PyTorch (fp32)", pt_ms)

    if args.skip_onnx:
        print("\n[2/2] ONNX skipped (--skip-onnx)")
        return

    # ONNX export + benchmark
    print(f"\n[2/2] ONNX Runtime benchmark")
    cpu_model = model.cpu()

    print(f"  Exporting to {args.onnx_path} ...", flush=True)
    ok = _export_onnx(cpu_model, args.num_entities, torch.device("cpu"), args.onnx_path)

    if not ok:
        print("  Skipping ONNX benchmark due to export failure.")
        print("  PyTorch result above is still valid for latency estimation.")
        return

    size_mb = os.path.getsize(args.onnx_path) / (1024 * 1024)
    print(f"  Exported: {args.onnx_path}  ({size_mb:.1f} MB)")

    try:
        ort_ms = _benchmark_onnx(args.onnx_path, args.num_entities,
                                  args.num_warmup, args.num_runs)
        _report("ONNX Runtime (fp32)", ort_ms)
    except ImportError:
        print("  onnxruntime not installed — skipping ONNX benchmark.")
        print("  Install with: pip install onnxruntime  (or onnxruntime-gpu)")
    except Exception as e:
        print(f"  ONNX benchmark failed: {e}")

    print(f"\n{'='*60}")
    print("Note: ONNX Runtime is what Dolphin uses at inference time.")
    print("Train with the delay that matches your ONNX Runtime result.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
