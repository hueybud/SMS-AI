"""Behavior-cloning inference agent over IPC.

Python equivalent of the C++ ``LocalOnnxBackend`` worker loop.  Loads the
PyTorch ``CitrusTransformerBC`` checkpoint (not the ONNX), applies norm
stats from ``norm_stats.npz`` in Python before the forward pass (rather
than baked-into-graph), and produces the same ``(btn_flags, stick_vals)``
that ``LocalOnnxBackend`` would have produced.

Owns per-instance:
- ``kv_cache``: ``[3, 2, 1, 127, 512]`` — same shape the ONNX model carries.
- ``prev_labels``: 11 floats — last frame's 7 button flags + 4 stick floats.

Both are reset to zero when a STATE packet arrives with ``reset_context=True``
(phase boundary), matching ``LocalOnnxBackend::WorkerLoop``'s reset path.

Why PyTorch instead of ONNX over IPC?  RL training (Phase C) needs gradients,
so the policy lives in PyTorch.  Phase B exists to prove the IPC + Python
inference path produces the same behavior as the in-process ONNX path
before we add PPO on top.  Once a model is worth shipping for live play,
``export_onnx_transformer.py`` re-bakes it for ``LocalOnnxBackend``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Reuse the inference-clean model definition + entity layout from the
# export script.  Single source of truth for the architecture.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_onnx_transformer import (  # noqa: E402  (path manipulation above)
    BUTTON_DIM,
    CitrusTransformerBC,
    FEATURE_DIM,
    SEQ_LEN,
    STICK_DIM,
    TEMPORAL_DIM,
    TEMPORAL_LAYERS,
    _flat_to_entities_onnx,
)

from .protocol import CORE_FEATURE_DIM, StateFrame

# ``LocalOnnxBackend`` calls these PREV_ACTION_DIM (= BUTTON_DIM + STICK_DIM = 11).
PREV_ACTION_DIM = BUTTON_DIM + STICK_DIM
KV_CACHE_SHAPE = (TEMPORAL_LAYERS, 2, 1, SEQ_LEN - 1, TEMPORAL_DIM)

assert FEATURE_DIM == CORE_FEATURE_DIM + PREV_ACTION_DIM, (
    f"Feature layout mismatch: FEATURE_DIM={FEATURE_DIM} != "
    f"CORE_FEATURE_DIM({CORE_FEATURE_DIM}) + PREV_ACTION_DIM({PREV_ACTION_DIM})"
)


class BCAgent:
    """One agent = one Dolphin env.  Stateful between calls.

    Usage::

        agent = BCAgent("runs/transformer_v3/best_model.pt",
                        "runs/transformer_v3/norm_stats.npz")
        for state in env_loop:
            btn, stk = agent.act(state)        # handles reset_context internally
            env.send_action(state.frame_id, btn, stk)
    """

    def __init__(
        self,
        checkpoint_path: str,
        norm_stats_path: str,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.model = CitrusTransformerBC().to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            print(
                f"[BCAgent] loading from full checkpoint "
                f"(epoch {state.get('epoch', '?')})"
            )
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[BCAgent] weights loaded from {checkpoint_path}")

        # Norm stats: applied here in Python (vs baked-in for the ONNX path)
        # so RL training can update them later if needed.
        stats = np.load(norm_stats_path)
        self.norm_mean = (
            torch.from_numpy(stats["mean"].astype(np.float32))
            .view(1, FEATURE_DIM)
            .to(self.device)
        )
        self.norm_std = (
            torch.from_numpy(stats["std"].astype(np.float32))
            .view(1, FEATURE_DIM)
            .to(self.device)
        )
        print(f"[BCAgent] norm stats loaded from {norm_stats_path}")

        # Per-agent state — mirrors LocalOnnxBackend::m_kv_cache + m_prev_labels.
        self.kv_cache = torch.zeros(
            KV_CACHE_SHAPE, dtype=torch.float32, device=self.device
        )
        self.prev_labels = np.zeros(PREV_ACTION_DIM, dtype=np.float32)

    def reset(self) -> None:
        """Zero KV cache + prev_labels.  Called automatically when a STATE
        packet arrives with ``reset_context=True``; can also be called
        explicitly between matches."""
        self.kv_cache.zero_()
        self.prev_labels.fill(0.0)

    @torch.no_grad()
    def act(self, state: StateFrame) -> Tuple[np.ndarray, np.ndarray]:
        """One inference step.  Returns ``(btn_flags, stick_vals)`` ready to
        ship in an ACTION packet.

        - ``btn_flags``: float32 ``[7]`` — 0/1 from ACTION_VOCAB[Gumbel-argmax].
        - ``stick_vals``: float32 ``[4]`` — Gumbel-sampled bins mapped to [-1,1].
        """
        if state.reset_context:
            self.reset()

        # Build the 194-feature vector: 183 core (from C++) + 11 prev_labels
        # (owned by us, since RL training will need this state on the trainer
        # side).  Order MUST match AIController::ReadGameStateCore + the
        # trailing prev_action entity at offset 174 in _ENTITY_DEFS.
        feat_full = np.concatenate(
            [state.core_features, self.prev_labels], dtype=np.float32
        )
        feat_t = torch.from_numpy(feat_full).view(1, FEATURE_DIM).to(self.device)
        feat_t = (feat_t - self.norm_mean) / self.norm_std

        entities = _flat_to_entities_onnx(feat_t)
        btn_flags, stick_vals, kv_out = self.model.forward_infer(
            entities, self.kv_cache
        )

        self.kv_cache = kv_out

        btn_np = btn_flags.detach().cpu().numpy().reshape(-1).astype(np.float32)
        stk_np = stick_vals.detach().cpu().numpy().reshape(-1).astype(np.float32)

        # Update prev_labels for the next call.  Equivalent to the C++ update
        # in ``LocalOnnxBackend::WorkerLoop`` (which derives the same flags
        # from the decoded GCPadStatus bits — by construction those map back
        # to the ACTION_VOCAB row, so we can store the row directly).
        self.prev_labels[:BUTTON_DIM] = btn_np
        self.prev_labels[BUTTON_DIM : BUTTON_DIM + STICK_DIM] = stk_np

        return btn_np, stk_np
