# RL Gym Design (Strikers)

Status: **design + foundation laid** (updated 2026-04-29).
- BC transformer **v3 final** (composite=0.844) — this is the teacher for RL.
- C++ AIInferenceBackend abstraction landed (commit `88bb363ade` on `ai-controller`). Inference moved off the emu thread; `IpcBackend` slots in as a sibling of `LocalOnnxBackend` with no `AIController` changes needed.
- Remaining work: implement `IpcBackend` + Python trainer + reward function + PPO loop.

This doc captures the conversation on what an RL setup for Strikers should look like, using vladfi1's `slippi-ai` as the primary reference architecture.

Pick this up by reading the "MVP Path" and "Open Questions" sections at the bottom.

---

## 1. High-Level Architecture

```
          ┌───────────────────────────────────────────────┐
          │  Python trainer (long-lived, GPU)             │
          │   • owns PyTorch policy + PPO optimizer       │
          │   • owns per-env KV cache (batched)           │
          │   • computes rewards from state trajectories  │
          │   • orchestrates N dolphin subprocesses       │
          └───────▲─────────────▲─────────────▲───────────┘
                  │ IPC         │             │
             ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
             │ dolphin │   │ dolphin │  …│ dolphin │    (N headless instances)
             │  env 0  │   │  env 1  │   │  env N  │
             └─────────┘   └─────────┘   └─────────┘
```

- **Python = supervisor + training + inference.** Spawns dolphin processes, assigns ports, handles crashes/restarts, runs forward passes on GPU, runs PPO.
- **C++ Dolphin = pure env.** Runs one match at a time, reads GC memory, injects controller inputs, loads savestates on command. Does **not** run the model.
- **Process tree is 1-to-N**: one long-lived Python trainer over N short-lived Dolphin subprocesses. Lifecycle/error recovery lives in Python.

This is an inversion of today's setup, where C++ owns inference in-process via ONNX Runtime.

---

## 2. Three Timescales

Critical to keep these separate in your head — they often get conflated.

| Timescale | Frequency | What happens |
|---|---|---|
| **Frame**   | ~60 Hz (or faster, unthrottled) | C++ reads GC memory, builds 194-float state packet + flags, sends over IPC. Python runs one batched PyTorch forward pass across all N envs, returns N actions. C++ caches the action, `PlayController()` delivers it. |
| **Rollout** | Every ~240 frames (~4s) | Python finalizes a `Trajectory(states[241], actions[241], rewards[240], is_resetting[241])`. **Rewards are computed here**, in numpy, from the stored state buffer. No gradient update yet. |
| **Batch**   | Every N rollouts (e.g. 16 × 240 = ~64s game-time) | PPO runs multiple gradient epochs over the accumulated batch. New weights take effect; from the envs' POV the policy changes in one big jump. |

Rewards are **per-frame** (PPO's advantage estimator needs that), but **computed post-hoc** every rollout from stored state. They are *not* sent inline from C++.

Why post-hoc, not inline:
- Iteration speed — tuning reward weights is a Python edit, not a Dolphin rebuild.
- Unit testability — replay old rollouts against a new reward function.
- Cleaner protocol — C++ just ships state; it doesn't need to know what a reward is.

---

## 3. IPC Layer

The new C++/Python split requires an IPC layer that didn't exist before.

### Messages

1. **State → Python** (every frame, C++ → Python): the 194-float feature vector AIController already builds, plus raw values the reward needs (score, ball xyz, possession flag, `eGameState` at `cGame+0x24`), plus an `is_resetting` bit.
2. **Action → C++** (every frame, Python → C++): 6 button probs + 4 stick values, or a pre-decoded `GCPadStatus`. Decoding in Python is slightly simpler because hysteresis state naturally lives with the policy.
3. **Control → C++** (occasional, Python → C++): `reset(savestate_id)`, `shutdown`. Reset tells Dolphin to `State::LoadFromBuffer` the chosen savestate and set `is_resetting=True` on the next state packet.

### Direction: C++ listens, Python connects

Matches slippi-ai's pattern. Startup dance per env:

1. Python picks a free port via `portpicker.pick_unused_port()`.
2. Python launches the dolphin subprocess with the port as a CLI / INI arg (e.g. `[Movie] AIIpcPort=NNNNN` — follows existing `AIModelPath`, `AIControlledPort`, `AIMirrorX` convention).
3. AIController constructs an **IpcBackend** instead of LocalOnnxBackend (selected by INI: AIIpcPort set → IPC, else local). IpcBackend binds + listens on that port during init, on its own worker thread (the AIInferenceBackend abstraction is already in place from commit `88bb363ade` — no off-thread plumbing needed beyond the socket).
4. Python retries `connect()` in a loop with a ~30s timeout; dolphin takes a few seconds to boot.
5. Connection established; `send` / `recv` both ways for the rest of the session.

### Concurrency model

- N dolphin processes, each with its own port, each with its own socket.
- Python holds a list of N sockets. Either one thread per socket into a shared queue, or a `select`/`poll` loop over all of them. slippi-ai's `AsyncEnvMP` uses forkserver subprocess pipes — we can use TCP instead and get basically the same pipelining.
- Don't multiplex N envs over one port. The "which env is this from" bookkeeping isn't worth the saved port.

### Wire format

TCP loopback, fixed-size binary packets. Low-latency requirement: sub-millisecond round trips, so no JSON. A simple length-prefix + float array is fine. Exact schema is TBD — see Open Questions.

### Crash recovery

slippi-ai's `SafeEnvironment` pattern: if a dolphin crashes or a socket breaks, Python kills that subprocess, spawns a replacement on a new port, reloads the savestate, resumes. The trainer can lose one env without taking down the whole batch.

---

## 4. Savestates Replace Menu Scripting

libmelee ships a `menu_helper` that navigates character select → stage select → match start because Melee's match loop is `menu → match → results → menu → match → ...` and they want continuous play.

We don't need any of that. Dolphin has `State::LoadFromBuffer` (in-memory savestates). Plan:

- Record a `kickoff.sav` once from a captain/sidekick/arena/items setup we like, paused at the frame right before kickoff.
- Episode = `LoadFromBuffer(kickoff.sav)` → play until terminal condition → `LoadFromBuffer(kickoff.sav)` → repeat.
- Terminal condition is whatever we pick: first goal, N seconds elapsed, ball-dead, `eGameState` leaves the "active play" family (4/5). We never have to watch a goal cam, celebration, replay, or results screen.
- For variety later, build a pool of savestates (different captains, sidekicks, arenas, items) and sample one per reset. Zero menu code.

In-memory savestate load should be ~millisecond-fast, much faster than menu navigation. slippi-ai's reset cost is hundreds of frames of menu-time; ours is basically a memcpy.

This makes the Citrus gym **simpler** than slippi-ai's, not just equal to it.

---

## 5. Reward Function (to-be-designed)

Computed in Python post-rollout from the state trajectory buffer. All sparse+dense inputs are readable from addresses we already have.

Candidates (prioritize while iterating):
- **Sparse**: goal diff per frame (∆ score). The true objective.
- **Dense shaping**:
  - Ball x-coordinate toward opponent goal (approaching factor equivalent).
  - Possession flag delta (gained/lost possession).
  - Shots on goal, saves.
  - Deke landed (from eFielderActionState transitions).
  - Item pickup.
  - Body-check / red-card penalty.
- **Regularization**:
  - KL to teacher (BC model) penalty — stays human-like, prevents degenerate strategies. slippi-ai's `kl_teacher_weight ≈ 3e-3`.

Zero-sum between teams (Team A reward = -Team B reward) prevents collusion in self-play.

Exact weights TBD. Start with goal-diff-only + KL-to-teacher, add dense shaping only if sparse reward doesn't learn.

---

## 6. PPO + Teacher KL (port of slippi-ai)

Key hyperparams from `scripts/rl_example.sh` as starting point:
- `reward_halflife = 4s`
- `kl_teacher_weight = 3e-3`
- `policy_gradient_weight = 5`
- `ppo.num_epochs = 2`
- `ppo.num_batches = 16`
- `ppo.epsilon = 1e-2`
- `ppo.beta = 3e-1`
- `num_envs = 96` (aspirational; start much smaller)
- `rollout_length = 240`
- Optimizer burnin before RL, value-function burnin if training against CPUs.

Teacher = frozen BC checkpoint. Policy starts as a copy of teacher. KL(policy || teacher) penalty keeps the policy from drifting too far from human-like play during RL.

---

## 7. Inference Path During Training

**Training**: policy runs in PyTorch inside the Python trainer. Batched forward pass across all N envs per frame: `state[N, 194]` + `kv_cache[N, 3, 2, 127, 512]` → `btn_probs[N, 7]` + `stick_vals[N, 4]` + `kv_cache_out[N, ...]`. GPU-resident.

**No ONNX during training.** Serializing PyTorch → ONNX every batch (~64s of game time) would be seconds of overhead per minute of training. Dead on arrival.

**Deploy**: when a training run finishes (or at a checkpoint we like), run the existing `SMS AI/scripts/export_onnx_transformer.py` once to produce a new `best_model.onnx`. That ships to Dolphin's AIController for live play via the same code path we have today. Same export pipeline, run at the end instead of every batch.

### Normalization

Today norm stats are baked into the ONNX graph. During training, we apply norm stats in Python (from `norm_stats.npz`) before the forward pass. Zero functional difference; just a different location for the multiply-add.

### KV cache ownership flips

Today: `LocalOnnxBackend` owns `m_kv_cache` on its worker thread (AIController itself holds nothing model-specific anymore).
RL: Python owns per-env caches, stacked into a batched tensor for the forward pass. Reset on episode boundary. The `IpcBackend` C++ side doesn't hold the cache — it just relays state out and actions in.

---

## 8. Self-Play / Opponent

Strikers is 4-player (2v2). The decision tree is bigger than Melee's ditto.

Options (easy → hard):
1. **1 agent + 3 game CPUs.** Simplest. Validates the RL loop before scaling.
2. **2 policies vs 2 game CPUs.** Cooperative learning — two agents on same team share a policy (or two separate policies).
3. **Full 4-policy self-play / ditto.** Mirror match where all 4 slots are the current policy. Matches slippi-ai's `opponent.type=self`.
4. **Policy vs older-snapshot opponent** (league / PBT territory). Later.

Start at #1. `opponent.train=True` (slippi-ai equivalent) means train on both teams' trajectories — doubles effective batch size and symmetrizes learning.

---

## 9. What to Steal from slippi-ai

Layered env stack from `slippi_ai/envs.py`:
- `Environment` — single dolphin, sync.
- `SafeEnvironment` — auto-retry on disconnect/timeout/wrong-state.
- `BatchedEnvironment` — N sync envs, batched input/output.
- `AsyncEnvMP` — single env in its own subprocess, queue-pipelined push/pop.
- `AsyncBatchedEnvironmentMP` — N envs across M subprocesses. This is what unlocks the 96-env throughput in their setup.

`RolloutWorker` from `slippi_ai/evaluators.py` — drives batched env, handles `online_delay` + agent `batch_steps` with the "env runahead" invariant, emits `Trajectory(states, actions, rewards, is_resetting, initial_state, delayed_actions)` shape `[T+1, B]`.

`LearnerManager` + `Learner` from `slippi_ai/rl/{run_lib,learner}.py` — PPO loop with teacher KL. Our version rewrites the policy unroll around our transformer + per-env KV cache instead of their RNN state, but the outer loop is portable almost as-is.

---

## 10. What slippi-ai Gets Free, We Have to Build

| slippi-ai layer | Citrus equivalent | Status |
|---|---|---|
| AIInferenceBackend abstraction (worker-thread plumbing) | already there | **DONE** (commit `88bb363ade`) |
| Slippstream IPC in Slippi Dolphin | `IpcBackend::Submit()` writes packet over TCP | **to build** (~150 lines C++) |
| Dolphin pipe input backend | `IpcBackend` receive thread reads action, publishes to output slot | **to build** (output-slot delivery already wired through PlayController) |
| libmelee.Console (launch + parse) | Python `Dolphin` class | **to build** (subprocess launch + socket parse) |
| libmelee.Controller | folded into IPC | to build |
| libmelee.menu_helper | **not needed** (savestates) | skipped |
| libmelee.Parser (state → features) | **not needed** (`ReadGameStateCore` already builds 183 floats) | skipped |
| FFW Gecko codes + EXI_AI build | may not be needed | headless already works on our fork |
| Reward (numpy over trajectory) | Strikers-specific reward | to design |
| PPO learner + teacher KL | port of `slippi_ai/rl/` | to build (adapt to transformer + KV cache) |

We skip the two biggest libmelee components — menu scripting and state parsing — because savestates + AIController already cover those.

---

## 11. What We Already Have

- **Headless Dolphin** validated on Windows (AIController build runs Strikers without video/audio). This was expected to be a gap; it isn't.
- **AIController.cpp** already reads live GC memory and builds the 183-float CORE feature vector via `ReadGameStateCore()` (worker appends prev_labels to make 194). On the emu thread, costs ~0.01ms.
- **AIInferenceBackend abstraction** (commit `88bb363ade`) — pluggable inference site. `LocalOnnxBackend` runs ORT on a dedicated worker thread today; `IpcBackend` slots in as a sibling for RL. The whole emu↔backend handoff (single-slot latest-wins input, mutex-protected output slot, reset_context flag for KV/prev_labels) is already wired and tested.
- **BC transformer v3 final** — 7.55M param model, composite=0.844 on the v3 dataset. ONNX export + baked normalization. This is the teacher for RL.
- **Game state RE** — `cGame` singleton at `0x80373708`, `eGameState` at +0x24, field geometry, character array, action states (eFielderActionState / eGoalieActionState). All needed for reward + terminal detection.
- **Export pipeline** — `export_onnx_transformer.py` with baked-in normalization. Reuse at deploy time post-RL.
- **INI config plumbing** — `[Movie] AIModelPath`, `AIControlledPort`, `AIMirrorX` already live in Dolphin.ini. New keys (`AIIpcPort`, `AISavestatePath`) slot in naturally.
- **Per-frame timing instrumentation** — `WindowedStats` dumps every ~10s for `gs_ms`, `interval_ms` (emu-thread cadence), `infer_ms` (worker-thread inference). Same pattern can measure IPC roundtrip latency for free once `IpcBackend` lands.

---

## 12. MVP Path

Prioritize correctness on one env over throughput on many.

1. **C++ `IpcBackend` (sibling of `LocalOnnxBackend`)**: implements `AIInferenceBackend`. On `Submit(AIInputFrame)`: serializes the 183-float core features + reward-relevant raw values + `is_resetting` flag, writes a length-prefix binary packet over TCP loopback. Owns its own worker thread (not the same one as LocalOnnxBackend; different backend, different lifecycle).
2. **C++ `IpcBackend` receive thread**: reads action packets from socket, decodes into `GCPadStatus`, publishes to the existing output slot — `PlayController` already reads from there, no controller changes.
3. **C++ INI selection**: `AIIpcPort` set → `AIController::Load` constructs `IpcBackend` instead of `LocalOnnxBackend`. ~5 lines.
4. **C++ reset**: handle a `reset(savestate_id)` control message → `State::LoadFromBuffer`. Lives in IpcBackend or a sibling shim that has access to Dolphin's State API.
5. **Python `Dolphin` subprocess wrapper**: portpicker → launch subprocess with `--config Movie.AIIpcPort=N` → connect with retry.
6. **Python single-env Environment class**: thin wrapper around the socket (send action, recv state, reset).
7. **Python rollout worker**: PyTorch policy + per-env KV cache, collect 240-frame trajectories.
8. **Python reward function**: start with goal-diff only, zero-sum.
9. **Python PPO + teacher KL loop**: port of `slippi_ai/rl/learner.py`, adapted to our transformer.
10. **Single env, slow, prove the loop trains.** Confirm loss curves make sense, policy improves on goal-diff-per-minute.
11. **Then scale**: `BatchedEnvironment` (N sync envs), then `AsyncBatchedEnvironmentMP` if GIL / subprocess overhead is the bottleneck.
12. **Then optimize**: FFW Gecko codes for Strikers if throughput isn't enough.

Steps 1-3 are the only C++ changes needed and they all fit inside the existing AIInferenceBackend abstraction. No `AIController` changes; no new threading primitives to design (the input slot, output slot, condvar wait pattern are already established by `LocalOnnxBackend`).

Don't pre-optimize. Throughput is a known-solution problem (more envs, async pipelining, FFW); a broken reward function or a policy that won't learn from its own trajectories is a design bug, and you want to find those on one env first.

---

## 13. Open Questions

Things we didn't decide in this pass; pick up here next session:

- **Wire format specifics.** Exact field layout of the state packet (order, types, padding). Exact action packet shape. How control messages are distinguished from action messages (different socket? tag byte?).
- **Reward shaping specifics.** Weights for goal diff vs ball-x vs possession vs deke landings. Ideally inform with some small experiments before committing.
- **Episode termination policy.** First goal? Fixed N seconds? Time-limited match with multiple goals? Different choices give very different episode-length statistics.
- **Savestate pool.** Start with one, or curate a diverse set from day 1?
- **How/where opponents get chosen.** Fix game CPU level? Rotate among multiple CPU levels for robustness?
- **Observation filter.** slippi-ai has per-port `ObservationConfig` (e.g. to limit what opponent info the agent sees). Probably not needed at MVP but worth remembering exists.
- **Action delay.** slippi-ai trains with `online_delay=18` to match their netplay conditions. We probably want 0 at first; revisit if we want to deploy over netplay.
- **Savestate location on disk vs buffer.** `LoadFromBuffer` is faster but we need to SaveToBuffer once at startup. Worth measuring.

---

## 14. Reference

- slippi-ai source: `C:\Users\Brian\Documents\Slippi AI\`
- Key files:
  - `slippi_ai/dolphin.py` — subprocess launch, headless, EXI_AI build flags
  - `slippi_ai/envs.py` — layered env stack (Single → Safe → Batched → Async+MP)
  - `slippi_ai/evaluators.py` — `RolloutWorker`, `Trajectory`
  - `slippi_ai/reward.py` — numpy reward functions over trajectory state
  - `slippi_ai/rl/run_lib.py` — entry point, opponent types, training loop
  - `slippi_ai/rl/learner.py` — PPO + teacher KL + value function
  - `scripts/rl_example.sh` — canonical hyperparams (i7-11700K + 3080Ti + 64GB)

- Our code:
  - `Source/Core/Core/AIController.{h,cpp}` — inference today, will split for RL
  - `SMS AI/scripts/train_transformer.py` — BC training (teacher source)
  - `SMS AI/scripts/export_onnx_transformer.py` — post-RL deploy path
  - Game RE addresses in `memory/MEMORY.md` → "Strikers RE (Ghidra)" section
