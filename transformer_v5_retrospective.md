# Transformer v5 Training — Retrospective & v6 Notes

## What v5 Fixed (vs. v4 AR Head)

v5 replaced `ARControllerHead` with `IndependentControllerHead`, eliminating the
teacher forcing gap that caused all-neutral sticks at inference. From epoch 1,
sx=80.7% non-neutral — sticks moving immediately. In-game the model plays at
roughly LSTM level: sticks moving, sprint (R) used well, occasional chip shots
and charged shots.

---

## Problems Encountered This Session

### 1. NaN Losses Starting at Epoch 7 (bf16 AMP Overflow)

**Symptom:** Starting at epoch 7, large clusters of `WARN: non-finite loss
(bce=nan stk_ce=nan)` — ~150 skipped batches in the first occurrence, growing
worse each time the run was restarted.

**Root cause:** bf16 (AMP) training. As model weights grow during training,
intermediate activations in the transformer attention layers overflow bf16's
max value (~65504). Specifically:
- `EntityTransformerLayer` (nn.MultiheadAttention): Q*K^T overflows bf16 →
  softmax(overflow) = NaN
- `TemporalLayer` (F.scaled_dot_product_attention): same overflow path
- FFN/LayerNorm: Inf activations from FFN → LayerNorm computes Inf-Inf = NaN

**What we tried:**
| Fix | Result |
|-----|--------|
| Move NaN check before backward() | Prevented gradient poisoning, didn't fix NaN source |
| fp32 cast in EntityTransformerLayer | OOM (stale GPU process eating 42.99GB), reverted |
| fp32 cast in TemporalLayer | Reduced NaN but didn't eliminate it |
| fp32 logit cast before loss | Didn't help — NaN was in activations, not loss computation |
| Input NaN check | Confirmed training data is clean (no corrupt X.npy sequences) |
| fp32 cast in EntityTransformerLayer (retry after stale process killed) | Reduced cluster onset from batch 439 → batch 1464, still not eliminated |

**Final fix:** Remove `--amp` entirely. fp32 training on A6000 (47GB VRAM)
eliminates all bf16 overflow. Cost: ~2x slower per epoch (~36 min vs ~18 min).
Epochs 7+ ran completely clean with no NaN skips.

**For v6:** Either train without AMP from epoch 1, or add explicit fp32 casts
to ALL attention and FFN layers before enabling AMP. Do not rely on only
partial fp32 casts — the overflow can come from any layer once weights grow.

---

### 2. Resume Logic Missing from Training Script

**Problem:** No `--resume` / `--start-epoch` args existed. When epoch 7 NaN
was diagnosed, we had only `best_model.pt` (model weights only, no optimizer
state).

**Fixed:** Added `--resume path/to/model.pt` and `--start-epoch N` to
`train_transformer.py`. Scheduler fast-forwards via `last_epoch` parameter;
`initial_lr` must be manually set in optimizer param_groups when resuming
without optimizer state.

**Also fixed:** CSV/plot history was opened in `"w"` mode, wiping prior epochs
on resume. Now opens in append mode when resuming, and seeds `plot_history`
and `best_val_loss` from the existing CSV rows before the resume epoch.

**For v6:** Save full checkpoint (model + optimizer + scheduler state) every N
epochs. This avoids the LR schedule restart issue and enables true resume.
```python
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_val_loss": best_val_loss,
}, out / f"checkpoint_epoch{epoch}.pt")
```

---

### 3. Dolphin FPS Impact (Background Thread TODO)

**Problem:** ONNX inference runs on the emulation thread. Timing breakdown:
- ReadGameState: ~0.08–0.11ms (trivial)
- ONNX inference: ~4.2–6.0ms
- Total per frame: ~4.5ms = ~27% of the 16.7ms frame budget

Result: CPU saturation → FPS drops below 60 even though inference alone looks
fast enough. Emulation + inference + rendering all compete for the same thread.

**Proposed fix:** Move ONNX inference to a background thread.
- `TickAIController()` dispatches inference async
- `PlayController()` delivers the most recently completed result (at most 1
  frame stale — imperceptible)
- Emulation thread is freed for its ~12ms of actual emulation work

**Status:** Not yet implemented. v5 model plays at LSTM level but FPS is not
consistently 60. This should be addressed before extended in-game use.

---

## v5 Final Metrics (Best Checkpoint — Epoch ~10)

| Metric | Value |
|--------|-------|
| val_loss | ~1.349 |
| stick_err | ~24.3–24.5° |
| R (sprint) F1 | 0.839 |
| B (shoot) F1 | ~0.705 |
| A (pass) F1 | ~0.660 |
| idle_acc | ~0.590 |
| sx non-neutral | ~81–83% |
| sy non-neutral | ~76–78% |
| cx non-neutral | ~2–3% (collapsed) |
| cy non-neutral | ~0.7% (collapsed) |

---

## Known Pitfalls (Not Fixed by More Epochs)

1. **C-stick collapse** — cy=0.7% non-neutral throughout all of v5. 93% neutral
   in training data overwhelms even per-axis CE weights at max_weight=10.
   Needs higher max_weight specifically for cy, or neutral undersampling for
   c-stick axes.

2. **Kickoff inaction** — eGameState=1 features differ from active play; KV
   cache cold start at kickoff. Model stands around during kickoff phase.
   Needs kickoff-specific oversampling or a separate kickoff policy.

3. **L button weak (F1 ~0.4)** — L is a modifier (L+A=lob pass, L+B=chip
   shot) but is trained as an independent binary output. Model has to learn
   the conjunction implicitly.

4. **idle_acc ~0.59** — Model presses buttons spuriously ~40% of neutral
   frames. Threshold tuning in AIController.cpp (raise sigmoid threshold
   above 0.5 per-button) can help without retraining.

5. **Netplay delay mismatch** — All training data recorded with 5–15 frame
   input buffer delay. Model may act slightly early relative to optimal timing.

---

## Suggested v6 Improvements

### Data / Labels
- **Composite lob labels** — Replace standalone L with `lob_pass` (L∧A) and
  `lob_shot` (L∧B) as dedicated output heads. Direct signal eliminates the
  conjunction learning problem. Requires y.npy rebuild.
- **Re-enable prev_labels as input** — Currently zeroed to fix AR head feedback
  loop, but IndependentControllerHead has no feedback loop. Prev_labels as a
  read-only feature gives the model continuity context (was I holding R/B last
  frame?). One-line training script change — stop zeroing
  `X_b[:, :, -PREV_ACTION_DIM:]`.
- **Explicit geometric features** — dist_to_goal, dist_to_ball, angle_to_goal
  as explicit floats rather than derivable-from-positions. Helps shot-timing
  decisions.
- **Per-phase oversampling** — Kickoff frames (phase 1) currently mixed with
  active play. Oversample or treat kickoff as separate segment type.

### Training
- **Full checkpointing** — Save optimizer + scheduler state every N epochs.
- **Train without AMP** — Or add fp32 casts to every attention + FFN block
  before attempting bf16 again.
- **Higher max_weight for cy** — Per-axis max_weight: give cstick_y its own
  ceiling (e.g. 25–30) since 93% neutral is extreme and 10.0 is insufficient.

### Inference (Dolphin)
- **Background thread for ONNX** — Move inference off the emulation thread.
  Deliver most recent completed output to PlayController() (1 frame stale max).
  Required for consistent 60fps during AI play.
- **Per-button sigmoid thresholds** — Tune above 0.5 for lob_pass/lob_shot
  (high-commitment moves); possibly lower for B (shoot) if model is too passive.
