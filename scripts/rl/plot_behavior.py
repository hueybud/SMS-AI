"""Plot behavioral telemetry from a train_batched.py run.

Companion to ``rl.plot_metrics`` focused on the columns added 2026-05-27
(``shots_for``, ``shots_against``, ``gains``, ``losses``, ``stag_events``,
plus per-cycle ``goals_for`` / ``goals_against``).  Use this to see what
the policy is DOING differently across cycles -- shots taken, ball
gains/losses, stagnation -- rather than just whether reward is rising.

Run::

    python3 -m rl.plot_behavior --run-dir /home/ubuntu/SMS-AI/runs/rl_batched_v3_rookie_kl3e2

Writes ``behavior.png`` to the run dir.  Each panel shows the raw
per-cycle count (faint) plus a rolling mean (heavy line) so trends are
readable through the per-cycle noise.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_RUN_DIR = "/home/ubuntu/SMS-AI/runs/rl_batched"


def _rolling_mean(xs: list, window: int) -> list:
    """Simple centered moving average; pads ends with the boundary value."""
    n = len(xs)
    if n == 0 or window <= 1:
        return list(xs)
    out = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def _col(rows, key, cast=float):
    """Pull a column from the CSV rows, tolerant of missing fields."""
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None or v == "":
            vals.append(0)
        else:
            try:
                vals.append(cast(v))
            except ValueError:
                vals.append(0)
    return vals


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                   help="run directory containing metrics.csv "
                        "(default: %(default)s)")
    p.add_argument("--csv", default=None,
                   help="explicit metrics.csv path (overrides --run-dir)")
    p.add_argument("--out", default=None,
                   help="output PNG path (default: behavior.png in run dir)")
    p.add_argument("--window", type=int, default=20,
                   help="rolling-mean window in cycles (default 20 ~ 4 min)")
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else (Path(args.run_dir) / "metrics.csv")
    if not csv_path.exists():
        print(f"[plot-behavior] no csv at {csv_path}", file=sys.stderr)
        return 1

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"[plot-behavior] csv empty: {csv_path}", file=sys.stderr)
        return 1
    if "shots_for" not in rows[0]:
        print(f"[plot-behavior] csv has no behavior columns (run pre-dates "
              f"2026-05-27 telemetry); skipping.", file=sys.stderr)
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot-behavior] matplotlib not available: {exc}", file=sys.stderr)
        return 1

    cycle = _col(rows, "cycle", int)
    shots_for     = _col(rows, "shots_for", int)
    shots_against = _col(rows, "shots_against", int)
    gains         = _col(rows, "gains", int)
    losses        = _col(rows, "losses", int)
    stag          = _col(rows, "stag_events", int)
    goals_for     = _col(rows, "goals_for", int)
    goals_against = _col(rows, "goals_against", int)

    sf_s  = _rolling_mean(shots_for,     args.window)
    sa_s  = _rolling_mean(shots_against, args.window)
    g_s   = _rolling_mean(gains,         args.window)
    l_s   = _rolling_mean(losses,        args.window)
    st_s  = _rolling_mean(stag,          args.window)
    gf_s  = _rolling_mean(goals_for,     args.window)
    ga_s  = _rolling_mean(goals_against, args.window)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Behavior: {csv_path.parent.name} "
                 f"(window={args.window}-cycle rolling mean)", fontsize=12)

    # Top-left: shots
    ax = axes[0, 0]
    ax.plot(cycle, shots_for,     color="steelblue", linewidth=0.5, alpha=0.3)
    ax.plot(cycle, shots_against, color="tomato",    linewidth=0.5, alpha=0.3)
    ax.plot(cycle, sf_s, color="steelblue", linewidth=2.0, label="shots_for")
    ax.plot(cycle, sa_s, color="tomato",    linewidth=2.0, label="shots_against")
    ax.set_title("Shots / cycle (offensive aggression)")
    ax.set_xlabel("cycle")
    ax.set_ylabel("count / cycle (16 envs)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Top-right: gains vs losses
    ax = axes[0, 1]
    ax.plot(cycle, gains,  color="darkgreen", linewidth=0.5, alpha=0.3)
    ax.plot(cycle, losses, color="darkred",   linewidth=0.5, alpha=0.3)
    ax.plot(cycle, g_s, color="darkgreen", linewidth=2.0, label="gains")
    ax.plot(cycle, l_s, color="darkred",   linewidth=2.0, label="losses")
    ax.set_title("Possession gains vs losses / cycle")
    ax.set_xlabel("cycle")
    ax.set_ylabel("count / cycle (16 envs)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Bottom-left: stagnation
    ax = axes[1, 0]
    ax.plot(cycle, stag, color="goldenrod", linewidth=0.5, alpha=0.3)
    ax.plot(cycle, st_s, color="goldenrod", linewidth=2.0, label="stag_events")
    ax.set_title("Stagnation events / cycle "
                 "(STAGNATION + BACKWARD_FREE + WALL_CAMP)")
    ax.set_xlabel("cycle")
    ax.set_ylabel("count / cycle (16 envs)")
    ax.grid(alpha=0.3)

    # Bottom-right: goals for context
    ax = axes[1, 1]
    ax.plot(cycle, goals_for,     color="steelblue", linewidth=0.5, alpha=0.3)
    ax.plot(cycle, goals_against, color="tomato",    linewidth=0.5, alpha=0.3)
    ax.plot(cycle, gf_s, color="steelblue", linewidth=2.0, label="goals_for")
    ax.plot(cycle, ga_s, color="tomato",    linewidth=2.0, label="goals_against")
    ax.set_title("Goals / cycle (outcome)")
    ax.set_xlabel("cycle")
    ax.set_ylabel("goals / cycle (16 envs)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(args.out) if args.out else (csv_path.parent / "behavior.png")
    fig.savefig(out_path, dpi=110)
    print(f"[plot-behavior] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
