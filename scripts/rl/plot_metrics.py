"""Plot RL training metrics from metrics.csv.

Usage::

    py -3.9 -m rl.plot_metrics                                # uses default run_dir
    py -3.9 -m rl.plot_metrics --run-dir path/to/rl_run       # specific run
    py -3.9 -m rl.plot_metrics --csv path/to/metrics.csv      # explicit csv

Reads the CSV that ``rl.train`` writes (one row per rollout) and saves
``metrics.png`` next to it.  Mirrors the BC training plot style: a grid
of small panels covering the things we actually look at to judge
progress (reward / score / losses / inference lag / drain).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

DEFAULT_RUN_DIR = (
    r"C:\Users\Brian\Documents\SMS AI\runs\rl_phase_c"
)


def _read_csv(path: Path) -> dict:
    cols: dict = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def _to_float(xs: List[str]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for x in xs:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _to_int(xs: List[str]) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for x in xs:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _running_avg(xs: List[float], window: int) -> List[float]:
    """Simple trailing average; pads early entries with the partial mean."""
    out: List[float] = []
    acc = 0.0
    n = 0
    from collections import deque
    q: deque = deque(maxlen=window)
    for x in xs:
        q.append(x)
        out.append(sum(q) / len(q))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--csv", default=None,
                   help="explicit metrics.csv path (overrides --run-dir)")
    p.add_argument("--out", default=None,
                   help="explicit output PNG path (default: metrics.png in same dir)")
    p.add_argument("--smooth", type=int, default=20,
                   help="trailing-average window for noisy series (default 20)")
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else Path(args.run_dir) / "metrics.csv"
    if not csv_path.exists():
        print(f"[plot] no csv at {csv_path}")
        return 1

    cols = _read_csv(csv_path)
    if not cols.get("rollout"):
        print(f"[plot] csv has no rows: {csv_path}")
        return 1

    rollout = _to_int(cols["rollout"])
    update_tag = cols["update_tag"]
    reward_sum = _to_float(cols["reward_sum"])
    reward_events = _to_int(cols["reward_events"])
    score_left = _to_int(cols["score_left"])
    score_right = _to_int(cols["score_right"])
    loss_total = _to_float(cols["loss_total"])
    loss_ppo = _to_float(cols["loss_ppo"])
    loss_value = _to_float(cols["loss_value"])
    loss_kl = _to_float(cols["loss_kl_teacher"])
    loss_entropy = _to_float(cols["loss_entropy"])
    grad_norm = _to_float(cols["grad_norm"])
    act_avg = _to_float(cols["act_avg_ms"])
    act_max = _to_float(cols["act_max_ms"])
    drained = _to_int(cols["drained_states"])
    unique_actions = _to_int(cols["unique_actions"])
    actions_total = _to_int(cols["actions_total"])

    # Filter UPD rows for loss panels — col/skp rows have null metrics.
    upd_mask = [t == "UPD" for t in update_tag]
    upd_rollout = [r for r, m in zip(rollout, upd_mask) if m]
    upd_loss_total = [v for v, m in zip(loss_total, upd_mask) if m]
    upd_loss_ppo = [v for v, m in zip(loss_ppo, upd_mask) if m]
    upd_loss_value = [v for v, m in zip(loss_value, upd_mask) if m]
    upd_loss_kl = [v for v, m in zip(loss_kl, upd_mask) if m]
    upd_loss_entropy = [v for v, m in zip(loss_entropy, upd_mask) if m]
    upd_grad_norm = [v for v, m in zip(grad_norm, upd_mask) if m]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[plot] matplotlib not available: {exc}")
        return 1

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(f"RL Training Metrics — {csv_path}", fontsize=11)

    # ── Score differential ────────────────────────────────────────────
    # Differential = left (AI) − right (CPU).  Positive = AI ahead.
    # Per-side absolute lines are drawn lightly underneath for context.
    ax = axes[0, 0]
    score_diff = [
        (l - r) if (l is not None and r is not None) else None
        for l, r in zip(score_left, score_right)
    ]
    ax.plot(rollout, score_left, color="steelblue", linewidth=0.6,
            alpha=0.4, label="left (AI)")
    ax.plot(rollout, score_right, color="tomato", linewidth=0.6,
            alpha=0.4, label="right (CPU)")
    ax.plot(rollout, score_diff, color="black", linewidth=1.4,
            label="differential (AI − CPU)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Score (live game)")
    ax.set_xlabel("rollout")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Reward sum + events ────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(rollout, reward_sum, color="purple", linewidth=0.6,
            label="reward_sum (per rollout)")
    if len(reward_sum) >= args.smooth:
        ax.plot(
            rollout,
            _running_avg([float(x) for x in reward_sum], args.smooth),
            color="black", linewidth=1.2,
            label=f"avg-{args.smooth}",
        )
    ax.set_title("Reward sum")
    ax.set_xlabel("rollout")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(rollout, reward_events, color="darkgreen", linewidth=0.6,
            label="events/rollout")
    if len(reward_events) >= args.smooth:
        ax.plot(
            rollout,
            _running_avg([float(x) for x in reward_events], args.smooth),
            color="black", linewidth=1.2,
            label=f"avg-{args.smooth}",
        )
    ax.set_title("Reward events / rollout")
    ax.set_xlabel("rollout")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Loss panels (UPD only) ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(upd_rollout, upd_loss_ppo, color="steelblue", label="ppo_obj")
    ax.plot(upd_rollout, upd_loss_value, color="orange", label="value")
    ax.plot(upd_rollout, upd_loss_kl, color="tomato", label="kl_teacher")
    ax.set_title("Loss components (UPD)")
    ax.set_xlabel("rollout")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(upd_rollout, upd_loss_total, color="black", label="loss/total")
    ax.set_title("Total loss (UPD)")
    ax.set_xlabel("rollout")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(upd_rollout, upd_loss_entropy, color="mediumseagreen",
            label="entropy")
    ax.set_title("Policy entropy (UPD)")
    ax.set_xlabel("rollout")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Grad norm + KL detail ─────────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(upd_rollout, upd_grad_norm, color="brown", linewidth=0.6,
            label="|grad|")
    if len(upd_grad_norm) >= args.smooth:
        ax.plot(
            upd_rollout,
            _running_avg(upd_grad_norm, args.smooth),
            color="black", linewidth=1.2, label=f"avg-{args.smooth}",
        )
    ax.set_title("Grad norm (UPD)")
    ax.set_xlabel("rollout")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Inference lag (act_avg, act_max) ───────────────────────────────
    ax = axes[2, 1]
    ax.plot(rollout, act_avg, color="steelblue", linewidth=0.6,
            label="act_avg ms")
    ax.plot(rollout, act_max, color="tomato", linewidth=0.4, alpha=0.6,
            label="act_max ms")
    ax.axhline(16.7, color="gray", linestyle="--", linewidth=0.8,
               label="60fps budget (16.7ms)")
    ax.set_title("Python inference time")
    ax.set_xlabel("rollout")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Drain + action diversity ───────────────────────────────────────
    ax = axes[2, 2]
    ax2 = ax.twinx()
    ax.plot(rollout, drained, color="orange", linewidth=0.6,
            label="drained STATE/rollout")
    if unique_actions and actions_total:
        diversity = [
            (u / a) if (a and u is not None) else None
            for u, a in zip(unique_actions, actions_total)
        ]
        ax2.plot(rollout, diversity, color="purple", linewidth=0.6,
                 label="unique/total actions")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("action diversity", color="purple")
    ax.set_title("Drain + action diversity")
    ax.set_xlabel("rollout")
    ax.set_ylabel("drained STATE", color="orange")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(args.out) if args.out else csv_path.with_name("metrics.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
