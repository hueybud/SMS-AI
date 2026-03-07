#!/usr/bin/env python3
"""export_onnx.py — Export trained StrikersLSTM to stateful ONNX.

The exported model processes one game frame at a time, accepting the LSTM
hidden/cell state as explicit inputs and returning updated state as outputs.
AIController.cpp feeds h/c back on every OnFrameEnd() call, maintaining
temporal context across the entire match.

The AR controller head runs autoregressively at inference time:
  buttons A→B→X→Y→L→R predicted in sequence (each conditioned on previous),
  then stick_x→stick_y→cstick_x→cstick_y (discrete 21-bin categorical).
  Stick bins are converted back to float values before output.

ONNX I/O:
    Inputs:
        features  [1, 440]     — single game frame feature vector (430 core + 10 prev action)
        h_in      [2, 1, 512]  — LSTM hidden state  (layers, batch, hidden)
        c_in      [2, 1, 512]  — LSTM cell state
    Outputs:
        btn_probs  [1, 6]      — button probabilities (sigmoid of AR logits)
        stick_vals [1, 4]      — stick values in [-1,1] (discrete bins→float)
        h_out      [2, 1, 512] — updated hidden state
        c_out      [2, 1, 512] — updated cell state

Usage:
    python export_onnx.py best_model.pt best_model.onnx

Verify with:
    python -c "
    import onnxruntime as ort, numpy as np
    sess = ort.InferenceSession('best_model.onnx')
    feeds = {
        'features': np.zeros((1, 440), dtype=np.float32),
        'h_in':     np.zeros((2, 1, 512), dtype=np.float32),
        'c_in':     np.zeros((2, 1, 512), dtype=np.float32),
    }
    btn, stk, h, c = sess.run(None, feeds)
    print('btn:', btn.shape, 'stk:', stk.shape, 'h:', h.shape, 'c:', c.shape)
    "
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Mirror constants from train.py (must stay in sync)
FEATURE_DIM = 440
BUTTON_DIM  = 6
STICK_DIM   = 4
STICK_BINS  = 21
LSTM_HIDDEN = 512
LSTM_LAYERS = 2
LSTM_PROJ   = 256


# --- Architecture (replicated from train.py) ---------------------------------

class ARControllerHead(nn.Module):
    """Autoregressive controller head — see train.py for full documentation."""

    RESIDUAL_DIM = 128

    def __init__(self, input_dim: int, stick_bins: int = STICK_BINS):
        super().__init__()
        self.stick_bins = stick_bins
        R = self.RESIDUAL_DIM

        self.residual_proj = nn.Linear(input_dim, R)
        self.btn_heads  = nn.ModuleList([nn.Linear(R + 2, 1)           for _ in range(BUTTON_DIM)])
        self.btn_embeds = nn.ModuleList([nn.Linear(2, R)               for _ in range(BUTTON_DIM)])
        self.stk_heads  = nn.ModuleList([nn.Linear(R + stick_bins, stick_bins)
                                         for _ in range(STICK_DIM)])
        self.stk_embeds = nn.ModuleList([nn.Linear(stick_bins, R)      for _ in range(STICK_DIM)])

    @staticmethod
    def _btn_emb(x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return torch.stack([1.0 - xf, xf], dim=-1)

    def _stk_emb(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(x.long(), self.stick_bins).float()

    def forward_infer(self, policy_t: torch.Tensor):
        """Autoregressive inference — policy_t: [B, input_dim] (time dim squeezed)."""
        B = policy_t.shape[0]
        R = self.residual_proj(policy_t)

        btn_probs_list = []
        prev_emb = torch.zeros(B, 2, device=policy_t.device)
        for i in range(BUTTON_DIM):
            logit = self.btn_heads[i](torch.cat([R, prev_emb], dim=-1))
            prob  = torch.sigmoid(logit.squeeze(-1))
            btn_probs_list.append(prob)
            prev_emb = self._btn_emb(prob > 0.5)
            R = R + self.btn_embeds[i](prev_emb)

        btn_probs = torch.stack(btn_probs_list, dim=-1)   # [B, BUTTON_DIM]

        stk_bins_list = []
        prev_emb = torch.zeros(B, self.stick_bins, device=policy_t.device)
        for i in range(STICK_DIM):
            logits = self.stk_heads[i](torch.cat([R, prev_emb], dim=-1))
            bins_i = torch.argmax(logits, dim=-1)
            stk_bins_list.append(bins_i)
            prev_emb = self._stk_emb(bins_i)
            R = R + self.stk_embeds[i](prev_emb)

        stick_bins = torch.stack(stk_bins_list, dim=-1)   # [B, STICK_DIM]
        return btn_probs, stick_bins


class StrikersLSTM(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, stick_bins: int = STICK_BINS):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, LSTM_PROJ)
        self.lstm       = nn.LSTM(LSTM_PROJ, LSTM_HIDDEN,
                                  num_layers=LSTM_LAYERS, batch_first=True)
        self.policy_fc  = nn.Linear(LSTM_HIDDEN, LSTM_PROJ)
        self.ar_head    = ARControllerHead(LSTM_PROJ, stick_bins)
        self.value_head = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x, h=None, c=None):
        """Inference-only forward (no teacher forcing).  T must equal 1."""
        proj = torch.relu(self.input_proj(x))
        hc   = (h, c) if (h is not None and c is not None) else None
        out, (h_out, c_out) = self.lstm(proj, hc)
        policy = torch.relu(self.policy_fc(out))   # [B, 1, LSTM_PROJ]
        # AR decode: squeeze time dim → [B, LSTM_PROJ]
        btn_probs, stick_bins = self.ar_head.forward_infer(policy[:, 0, :])
        return btn_probs, stick_bins, h_out, c_out


# --- Single-frame ONNX wrapper -----------------------------------------------

class _LSTMSingleFrameWrapper(nn.Module):
    """Wraps StrikersLSTM for single-frame stateful ONNX export.

    Takes one frame at a time (no T dimension in input), passes h/c explicitly,
    and returns updated h/c alongside action outputs.

    If norm_mean/norm_std are provided they are registered as buffers and baked
    into the ONNX graph as constants (constant-folded), so the exported model
    accepts raw feature values — no external normalization needed at inference.
    """

    def __init__(self, model: StrikersLSTM,
                 norm_mean: torch.Tensor = None,
                 norm_std:  torch.Tensor = None):
        super().__init__()
        self.model = model
        if norm_mean is not None and norm_std is not None:
            self.register_buffer("norm_mean", norm_mean.view(1, -1))  # [1, F]
            self.register_buffer("norm_std",  norm_std.view(1, -1))   # [1, F]
        else:
            self.norm_mean = None
            self.norm_std  = None

    def forward(self, features: torch.Tensor,
                h_in: torch.Tensor,
                c_in: torch.Tensor):
        """
        features: [1, FEATURE_DIM]  — raw (unnormalized) feature vector
        h_in:     [LSTM_LAYERS, 1, LSTM_HIDDEN]
        c_in:     [LSTM_LAYERS, 1, LSTM_HIDDEN]

        Returns:
            btn_probs  [1, BUTTON_DIM]  — sigmoid probability of each button press
            stick_vals [1, STICK_DIM]   — stick values in [-1,1] (discretized via AR bins)
            h_out      [LSTM_LAYERS, 1, LSTM_HIDDEN]
            c_out      [LSTM_LAYERS, 1, LSTM_HIDDEN]
        """
        # Normalize with baked-in constants (no-op if not provided)
        if self.norm_mean is not None:
            features = (features - self.norm_mean) / self.norm_std

        # Add time dimension for LSTM: [1, 1, FEATURE_DIM]
        x = features.unsqueeze(1)
        btn_probs, stick_bins, h_out, c_out = self.model(x, h_in, c_in)
        # Convert discrete stick bin indices back to normalized float values [-1, 1]
        stick_vals = stick_bins.float() / (STICK_BINS - 1) * 2.0 - 1.0   # [1, STICK_DIM]
        return btn_probs, stick_vals, h_out, c_out


# --- Export ------------------------------------------------------------------

def export(weights_path: str, output_path: str,
           norm_stats_path: str = None) -> None:
    device = torch.device("cpu")

    # Load model
    model = StrikersLSTM(FEATURE_DIM, STICK_BINS).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from {weights_path}")

    # Load normalization stats (baked into ONNX graph as constants)
    norm_mean = norm_std = None
    if norm_stats_path:
        stats = np.load(norm_stats_path)
        norm_mean = torch.from_numpy(stats["mean"])
        norm_std  = torch.from_numpy(stats["std"])
        print(f"Loaded norm stats from {norm_stats_path} — baking into ONNX graph")
    else:
        print("WARNING: no --norm-stats provided — exporting without normalization")

    wrapper = _LSTMSingleFrameWrapper(model, norm_mean, norm_std).to(device)
    wrapper.eval()

    # Dummy inputs
    dummy_features = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
    dummy_h        = torch.zeros(LSTM_LAYERS, 1, LSTM_HIDDEN, dtype=torch.float32)
    dummy_c        = torch.zeros(LSTM_LAYERS, 1, LSTM_HIDDEN, dtype=torch.float32)

    # Sanity-check forward pass
    with torch.no_grad():
        btn, stk, h_out, c_out = wrapper(dummy_features, dummy_h, dummy_c)
    print(f"Forward pass OK: btn={tuple(btn.shape)}  stk={tuple(stk.shape)}  "
          f"h_out={tuple(h_out.shape)}  c_out={tuple(c_out.shape)}")
    assert btn.shape   == (1, BUTTON_DIM),               f"btn shape mismatch: {btn.shape}"
    assert stk.shape   == (1, STICK_DIM),                f"stk shape mismatch: {stk.shape}"
    assert h_out.shape == (LSTM_LAYERS, 1, LSTM_HIDDEN), f"h_out shape mismatch: {h_out.shape}"
    assert c_out.shape == (LSTM_LAYERS, 1, LSTM_HIDDEN), f"c_out shape mismatch: {c_out.shape}"
    # Verify stick values are in [-1, 1] (discrete bins mapped to float)
    assert stk.min() >= -1.0 and stk.max() <= 1.0, f"stick_vals out of range: [{stk.min():.3f}, {stk.max():.3f}]"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_h, dummy_c),
        output_path,
        export_params       = True,
        opset_version       = 17,
        do_constant_folding = True,
        input_names         = ["features", "h_in", "c_in"],
        output_names        = ["btn_probs", "stick_vals", "h_out", "c_out"],
        dynamic_axes        = {
            # batch dimension is always 1 at inference — no dynamic axes needed
        },
    )
    print(f"Exported ONNX model to {output_path}")

    # Validate with onnxruntime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        # Verify input/output names
        in_names  = [inp.name for inp in sess.get_inputs()]
        out_names = [out.name for out in sess.get_outputs()]
        print(f"ORT inputs:  {in_names}")
        print(f"ORT outputs: {out_names}")
        assert in_names  == ["features", "h_in", "c_in"],                     f"input names: {in_names}"
        assert out_names == ["btn_probs", "stick_vals", "h_out", "c_out"],    f"output names: {out_names}"

        feeds = {
            "features": np.zeros((1, FEATURE_DIM), dtype=np.float32),
            "h_in":     np.zeros((LSTM_LAYERS, 1, LSTM_HIDDEN), dtype=np.float32),
            "c_in":     np.zeros((LSTM_LAYERS, 1, LSTM_HIDDEN), dtype=np.float32),
        }
        btn_out, stk_out, h_out, c_out = sess.run(None, feeds)
        print(f"onnxruntime validation OK:")
        print(f"  btn_probs  {btn_out.shape}  range=[{btn_out.min():.4f}, {btn_out.max():.4f}]")
        print(f"  stick_vals {stk_out.shape}  range=[{stk_out.min():.4f}, {stk_out.max():.4f}]"
              f"  (discrete: {STICK_BINS} bins)")
        print(f"  h_out      {h_out.shape}")
        print(f"  c_out      {c_out.shape}")

        # Verify stateful behaviour: running two frames should change h/c
        feeds2 = {
            "features": np.random.randn(1, FEATURE_DIM).astype(np.float32),
            "h_in":     h_out,
            "c_in":     c_out,
        }
        btn2, stk2, h2, c2 = sess.run(None, feeds2)
        state_changed = not np.allclose(h2, h_out)
        print(f"  Stateful check (h changes after step 2): {'PASS' if state_changed else 'FAIL'}")

    except ImportError:
        print("WARNING: onnxruntime not installed — skipping validation.")
        print("  pip install onnxruntime  # then re-run to validate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export StrikersLSTM to stateful ONNX")
    parser.add_argument("weights",      help="Path to best_model.pt (PyTorch state dict)")
    parser.add_argument("output",       help="Output path for .onnx file")
    parser.add_argument("--norm-stats", default=None, metavar="NPZ",
                        help="Path to norm_stats.npz from train.py; bakes normalization "
                             "into the ONNX graph so Dolphin receives raw features")
    args = parser.parse_args()

    export(args.weights, args.output, args.norm_stats)
