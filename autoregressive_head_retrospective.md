# Autoregressive Controller Head — Retrospective

## What It Was

The `ARControllerHead` was a neural network module that predicted controller outputs
sequentially rather than all at once. The idea: each output is predicted conditioned
on what was predicted before it, mimicking how a human might think about inputs
(knowing you're pressing A might influence whether you also press B).

The prediction order was:
1. Button A
2. Button B
3. Button X
4. Button Y
5. Button L
6. Button R
7. Stick X
8. Stick Y
9. C-Stick X
10. C-Stick Y

A shared "residual" vector `R` was projected from the transformer policy output and
updated after each prediction by adding an embedding of what was just predicted. Each
subsequent output was predicted from the current state of `R`.

This design was inspired by Slippi-AI's autoregressive controller head, which worked
well for their Melee imitation model.

---

## Problems Encountered

### 1. Shared R Contamination (stick mode collapse)

The original architecture used a single shared `R` vector for both buttons and sticks.
After 6 button predictions — nearly all "not pressed" — the R vector accumulated 6
"nothing is happening" embeddings before any stick prediction was made. This
systematically biased `R` toward a neutral representation, causing the stick heads to
always predict bin 10 (neutral, stickX=128, stickY=128).

**Diagnosis:** The training data showed stick_x/y are ~85% non-neutral (players are
almost always moving). The collapse was architectural, not a data problem.

**Fix attempted:** Added a separate `stk_residual_proj` linear layer giving sticks their
own independent pathway from the policy that bypassed the button loop entirely.

### 2. Class Imbalance on C-Stick (cstick mode collapse)

C-stick (deke) is rarely used — 78% neutral for cstick_x, 93% neutral for cstick_y.
Without weighting, the CE loss minimized by always predicting neutral for cstick.

**Fix attempted:** Added inverse-frequency class weights per stick axis
(`compute_stick_bin_weights`). Initially used a single shared weight vector averaged
across all 4 axes, which diluted the per-axis imbalance. Later refactored to per-axis
weights giving each axis its own weight vector.

### 3. Teacher Forcing Gap (the fatal problem)

During training, the AR head received the **true** previous outputs as context
(teacher forcing). During inference, it received its **own** previous predictions.

When the model made one wrong prediction, that wrong value was fed back as context
for the next prediction, compounding the error. The model was never trained to handle
its own mistakes, so it locked into degenerate fixed points at inference time.

**Observed in-game:** All buttons pressed simultaneously (A=B=X=Y=1.0 every frame),
all sticks neutral (128/128). The model had never seen "zero context → first prediction"
during training, so it produced garbage from a cold start that fed back and reinforced
itself.

**Mitigation attempted:** Zeroed out `prev_labels` in AIController.cpp so the model's
own outputs were not fed back as context. This broke the button feedback loop
(occasional A presses appeared), but sticks remained neutral.

**Root cause of persistence:** The 85% non-neutral metric in validation was measured
using teacher-forced predictions. The actual ONNX `forward_infer` path — which is what
runs in-game — produced completely different (collapsed) results. After 24 epochs, the
non-neutral validation percentages were flat (sx=85%, sy=87%, cx=26%, cy=0.7%) and
in-game behavior was unchanged: no stick movement, no meaningful button presses.

---

## What We Tried to Fix It

| Attempt | Result |
|---------|--------|
| Separate `stk_residual_proj` for sticks | Sticks showed ~1° improvement but still collapsed at inference |
| Per-axis CE class weights | Helped cstick_x slightly (cx: 22→26%), cstick_y still 0.7% |
| Reduced max_weight (20→10) to stabilize NaN losses | NaN rate improved but reappeared in clusters |
| Zeroed `prev_labels` in AIController.cpp | Fixed button cascade, sticks unchanged |
| 24 epochs of training | Metrics completely flat from epoch 4 onward |

---

## How We Resolved It

Replaced `ARControllerHead` with `IndependentControllerHead`. All outputs are predicted
directly and independently from the transformer policy vector with no conditioning
between them:

```python
class IndependentControllerHead(nn.Module):
    def __init__(self, input_dim, stick_bins=STICK_BINS):
        self.btn_head  = nn.Linear(input_dim, BUTTON_DIM)
        self.stk_heads = nn.ModuleList([nn.Linear(input_dim, stick_bins)
                                        for _ in range(STICK_DIM)])

    def forward_train(self, policy, btn_targets, stk_targets):
        btn_logits = self.btn_head(policy)
        stk_logits = [self.stk_heads[i](policy) for i in range(STICK_DIM)]
        return btn_logits, stk_logits

    def forward_infer(self, policy_t):
        btn_probs = torch.sigmoid(self.btn_head(policy_t))
        stk_bins  = torch.stack([self.stk_heads[i](policy_t).argmax(dim=-1)
                                  for i in range(STICK_DIM)], dim=-1)
        return btn_probs, stk_bins
```

`forward_train` and `forward_infer` are now mathematically identical — there is no
teacher forcing gap. Whatever the model learns during training is exactly what runs
at inference.

Additionally, `prev_labels` (the last 10 features of the 442-feature input vector)
are now zeroed in both the training loop and at inference, eliminating the remaining
source of train/inference mismatch.

**Result at epoch 1 of v5:** sx=80.7%, sy=78.3% non-neutral — sticks actively moving
from the very first epoch, which had never been achieved before with the AR head.

---

## Why Slippi-AI Didn't Have This Problem

Slippi-AI uses the same AR head design with the same shared-R contamination flaw.
They avoided the collapse because:

1. Melee players move the stick almost constantly — their training data had enough
   non-neutral signal to overcome the architectural bias through sheer volume.
2. They trained on ~10,000 hours of gameplay vs our smaller dataset.
3. Data volume bulldozed through problems that surface at smaller scale.

Our dataset, while large (173.9M frames, 6,418 games), had enough neutral frames and
a smaller absolute size that the AR head's biases won.
