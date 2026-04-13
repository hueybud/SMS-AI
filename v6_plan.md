# Transformer v6 — Change Plan

Summary of all agreed-upon changes from v5 retrospective + architecture review.

---

## 1. Embed Action States (replace one-hot)

**Problem:** 294 of 442 features (66.5%) are sparse one-hot action state vectors.
Striker state = 30 dims, goalie state = 27 dims, repeated across 10 characters.

**Change:** Replace one-hot encoding with learned embeddings.
- Striker action state: integer index → `nn.Embedding(30, 8)` → 8-dim dense vector
- Goalie action state: integer index → `nn.Embedding(27, 8)` → 8-dim dense vector
- Saves ~214 dims across all entity tokens
- Lets the model learn that similar states (running/sprinting) are close in embedding space

**Files:** `build_dataset.py` (output integer index instead of OH), `train_transformer.py`
(add embeddings in EntityEncoder), `AIController.cpp` (pass integer index instead of OH),
`export_onnx_transformer.py`

**Requires dataset rebuild:** Yes

---

## 2. Add Active Field Items to Features

**Problem:** CITF captures up to 25 active field items (shells, bob-ombs, chain chomps)
with position, velocity, type — but none of this appears in the feature vector. The model
cannot see items on the field.

**Change:** Add field item tokens to the entity set. Options:
- **Option A (simple):** Top-K nearest items as additional entity tokens (e.g., 4 tokens,
  each with type_embed + pos_delta + vel + strength). Increases N_ENTITIES from 16 to 20.
- **Option B (summary):** Aggregate item features into a fixed-size summary per type
  (nearest shell distance, nearest bomb distance, etc.). Fewer dims but loses spatial detail.

Option A preferred — it fits naturally into the entity-attention framework.

**Files:** `build_dataset.py`, `train_transformer.py` (N_ENTITIES, entity defs),
`AIController.cpp` (read active items from memory), `export_onnx_transformer.py`

**Requires dataset rebuild:** Yes

---

## 3. Conditional Stick Heads (Buttons → Sticks)

**Problem:** Independent action heads can't model correlated outputs. R modifies stick
meaning (sprint vs walk), and the stick head needs to know button context to predict
appropriate direction/commitment.

**Change:** Two-stage head:
1. Predict button logits independently (as now)
2. Concatenate sigmoid(btn_logits) with policy embedding, predict stick logits

```
policy_emb → btn_head → btn_logits (independent)
concat(policy_emb, sigmoid(btn_logits)) → stk_heads → stk_logits
```

Always feed `sigmoid(btn_logits)` in both training and inference. The gradient flows
through sigmoid so button heads still learn. Sticks see soft button probabilities in both
modes — zero train/inference gap.

Key benefit: stick head sees R=1 and learns sprint directions (committed, held longer)
vs walk directions (exploratory, frequent changes). Also sees lob_pass/chip_shot buttons
and learns appropriate aim for those actions.

**Files:** `train_transformer.py` (IndependentControllerHead → ConditionalControllerHead),
`export_onnx_transformer.py`

**Requires dataset rebuild:** No

---

## 4. Composite Lob Labels (replace standalone L)

**Problem:** L is not a standalone action — it is purely a modifier. L+A = lob pass,
L+B = chip shot. L alone does nothing in the game. Training L as an independent binary
output forces the model to independently predict "press L" and "press A" and hope they
coincide, with no training signal rewarding the combination.

**Change:** Remove standalone L from button outputs. Add two composite actions:
- `lob_pass` = (L AND A) in the training data
- `chip_shot` = (L AND B) in the training data

New button set: **A, B, X, Y, lob_pass, chip_shot, R** (7 buttons, up from 6).
At decode time: if `lob_pass` fires → set L+A on pad. If `chip_shot` fires → set L+B.

Note: A and B remain in the output alongside lob_pass/chip_shot. When lob_pass fires,
A should also fire (it IS a pass, just lobbed). The decoder sets both L and A flags.
Similarly chip_shot implies B. This means:
- Regular pass: A=1, lob_pass=0
- Lob pass: A=1, lob_pass=1 (decoder sets L+A)
- Regular shot: B=1, chip_shot=0
- Chip shot: B=1, chip_shot=1 (decoder sets L+B)

**Files:** `build_dataset.py` (label extraction), `train_transformer.py` (BUTTON_DIM=7,
button names, loss), `AIController.cpp` (DecodeOutput), `export_onnx_transformer.py`

**Requires dataset rebuild:** Yes

---

## 5. Re-enable prev_labels

**Problem:** prev_labels (last 10 features) are zeroed during training and inference as a
legacy fix for the v4 AR head feedback loop. With the new ConditionalControllerHead,
there is no feedback loop — prev_labels are read-only input.

**Change:** Stop zeroing `X_b[:, :, -PREV_ACTION_DIM:]` in training. In AIController.cpp,
populate m_prev_labels from the model's actual output after each frame.

Note: PREV_ACTION_DIM will change from 10 to 11 (7 buttons + 4 sticks) with the composite
lob labels. The prev_labels in the dataset need to match the new label format.

**Files:** `train_transformer.py` (remove zeroing line), `AIController.cpp` (update
m_prev_labels after DecodeOutput)

**Requires dataset rebuild:** Yes (prev_labels format changes with new button set)

---

## 6. Add Ball-to-Goal Features

**Problem:** The model has ball position and velocity but no explicit shot geometry. The
goal position (goal_x) is never passed to the model. The self-to-goal features only work
as a proxy when the controlled character is the ball carrier.

**Change:** Add 3 floats to the ball token:
- `ball_to_goal_dist / 40`
- `ball_to_goal_dx / dist` (normalized X component)
- `ball_to_goal_dy / dist` (normalized Y component)

Ball token grows from 19 to 22 dims (well within ENTITY_RAW_DIM=64 padding).

**Files:** `build_dataset.py`, `AIController.cpp`

**Requires dataset rebuild:** Yes

---

## 7. Remove Score Diff and Time Fraction

**Problem:** Score diff and time elapsed fraction (2 floats in context token) are noise.
Experts play to win regardless of score or clock. Stalling is prohibited in the community.
These features add nothing and may introduce spurious correlations.

**Change:** Remove `score_diff / 5` and `game_time / match_time_allotted` from the context
token and from AIController.cpp feature extraction.

**Files:** `build_dataset.py`, `AIController.cpp`

**Requires dataset rebuild:** Yes

---

## 8. Add Kickoff and Goalie-Has-Ball Features + Kickoff Oversampling

**Problem:** The model does nothing during kickoff — stands still until the game forces an
auto-pass after ~1-2 seconds. Kickoff frames are rare, get filtered by neutral_keep, and
the KV cache cold-starts. Similarly, goalie-has-ball transitions are underrepresented.

**Changes:**
- Add `is_kickoff` boolean (gamePhase == 1) to context token: 1 float
- Add `goalie_has_ball` boolean (friendly goalie is ball carrier) to context token: 1 float
- Force-include all kickoff-containing sequences in `oversample_seqs` (bypass neutral_keep
  filtering for any sequence that overlaps a kickoff phase)

**Files:** `build_dataset.py` (features + gamePhase in per-frame data for oversampling),
`train_transformer.py` (oversampling logic), `AIController.cpp` (new features)

**Requires dataset rebuild:** Yes

---

## 9. Focal Loss + Rare-Action Oversampling

**Problem:** Rare but important actions (item use, lob pass, chip shot) are severely
underrepresented. Current pos_weight caps at 20, but true inverse-frequency weights for
X=125, Y=51, L=75. The cap prevents gradient explosion but means the model still favors
"never press X."

**Changes:**

### A. Focal loss (replaces pos_weight BCE)
Replace `BCEWithLogitsLoss(pos_weight=...)` with focal loss:
```
FL(p, y) = -alpha * (1 - p_t)^gamma * log(p_t)
```
where `p_t = p` if `y=1`, else `p_t = 1-p`. Recommended: `gamma=2.0`, `alpha` per-button
set to inverse frequency (no cap needed — focal loss self-regulates via the `(1-p_t)^gamma`
term). Eliminates the need for manual pos_weight caps.

### B. Rare-action sequence oversampling
Duplicate sequences containing rare actions to bring them within 2-3x of A/B's
representation (~8% frame rate baseline). Rates derived from v5 training data:

| Action | Frame rate | vs A/B (~8%) | Duplication |
|--------|-----------|-------------|-------------|
| A, B (pass/shoot) | ~8% | 1x baseline | 1x (no dup) |
| c-stick (deke) | ~2-3% | 3-4x rarer | 3-4x |
| Y | 1.9% | 4x rarer | 3-4x |
| L→lob_pass/chip_shot | 1.3% | 6x rarer | 4-5x |
| X (item use) | 0.8% | 10x rarer | 4-5x |

Deke is a fundamental offensive tool — the counter to defensive hits. Rare in frame count
but critical for getting past defenders. A successful deke followed by R gives a turbo
boost, making the deke→sprint sequence a core skill pattern.

### C. neutral_keep stays at 1.0 (no neutral undersampling)
v5 data shows only ~15% of frames are truly idle (no buttons + neutral stick). The other
~25% of "no button" frames have active stick movement (positioning, spacing, defense).
These are valuable training data. The real imbalance is common vs rare actions within
action frames, not action vs neutral. Rare-action duplication + focal loss address this
directly.

**Files:** `train_transformer.py` (focal loss implementation, oversampling logic)

**Requires dataset rebuild:** No (oversampling is done at training time from existing labels)

---

## 10. Increase SEQ_LEN to 128

**Problem:** SEQ_LEN=64 (~1 second) limits the model's temporal context for positional
play. Experts think 2-3 seconds ahead for positioning and through balls.

**Change:** SEQ_LEN=128 (~2.1 seconds). KV cache at inference grows from [3,2,1,63,512]
to [3,2,1,127,512]. Training memory increases but should fit on A6000 47GB with fp32.

**Files:** `train_transformer.py` (constant), `export_onnx_transformer.py`,
`AIController.h` (KV_CACHE_SEQ constant)

**Requires dataset rebuild:** No (sequences are windowed at training time from the flat arrays)

---

## 11. Full Checkpointing

**Problem:** v5 only saved model weights (best_model.pt). No optimizer or scheduler state.
Resuming required manual LR schedule fast-forwarding and lost optimizer momentum.

**Change:** Save full checkpoint every N epochs:
```python
torch.save({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_val_loss": best_val_loss,
}, out / f"checkpoint_epoch{epoch}.pt")
```

**Files:** `train_transformer.py`

**Requires dataset rebuild:** No

---

## 12. Train Without AMP

**Problem:** bf16 AMP causes NaN losses from attention overflow starting around epoch 7.
Partial fp32 casts reduce but don't eliminate the issue.

**Change:** Default to fp32 training. Remove `--amp` from launch command. Keep the flag
available but document that it is known to cause NaN on this architecture.

**Files:** `train_transformer.py` (no code change, just launch config)

**Requires dataset rebuild:** No

---

## Summary

**Requires dataset rebuild (do together):** #1, #2, #4, #5, #6, #7, #8
**Model/training only:** #3, #9, #10, #11, #12

### Action space (v5 → v6)
- v5: A, B, X, Y, L, R (6 buttons, independent) + 4 stick axes (independent)
- v6: A, B, X, Y, lob_pass, chip_shot, R (7 buttons, independent) + 4 stick axes
  (conditioned on buttons)

### Key architectural changes
- One-hot action states → learned embeddings (in model, not features)
- Independent heads → conditional heads (buttons → sticks)
- BCE + pos_weight → focal loss (self-regulating, no cap)
- Standalone L → composite lob_pass / chip_shot
- Active field items as entity tokens
- Kickoff-aware features + oversampling

### Implementation order (suggested)
1. Dataset rebuild: #1, #2, #4, #5, #6, #7, #8 (one pass of build_dataset.py)
2. Model changes: #1 embeddings, #3 conditional heads, #4 composite decode, #9 focal loss
3. Training config: #10 SEQ_LEN, #11 checkpointing, #12 no AMP
4. Inference: update AIController.cpp + export_onnx_transformer.py to match
