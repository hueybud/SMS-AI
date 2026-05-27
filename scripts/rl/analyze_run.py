"""Standard analyzer for batched PPO training runs.

The canonical way to share results during trial-and-error tuning: run
this against a run dir, paste the output back, get a consistent
analysis.  Beats sharing raw CSV/log fragments because the report
always covers the same fields in the same order -- easy to compare
across runs and to spot what changed.

Usage::

    python3 -m rl.analyze_run /home/ubuntu/SMS-AI/runs/rl_batched_v2_rookie
    python3 -m rl.analyze_run <dir> --last-n 50      # widen the recent window

Reads:
  * ``metrics.csv``  -- required.  All numerical curves come from this.
  * ``training.log`` -- optional.  Scanned for warning/error counts.

Output sections:
  Header   -- run dir, cycles, wall-clock, throughput
  Health   -- skip rate, drained traffic, rho band, log warnings
  Curves   -- decile-binned trajectory (reward, goals, kl, |gn|, etc.)
  Recent   -- last-N-cycle aggregates with vs-early deltas
  Verdict  -- ✓ / ⚠ bullet list keyed off explicit thresholds

Designed to be self-contained: only stdlib + the same csv our training
loop writes.  No matplotlib (use rl.plot_metrics for that).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# ── Thresholds (single source of truth for what "healthy" means) ──────────
# Tuned to flag real problems without flooding noise.  Bump these as we
# learn more about the typical training profile across runs.
SKP_HIGH_FRAC      = 0.30     # >30% skip → likely policy collapse
SKP_WATCH_FRAC     = 0.10     # >10% skip → keep an eye on it
KL_RECENT_HIGH     = 3.0      # recent-window mean KL above this → tighten leash
KL_PEAK_WARN       = 5.0      # any cycle hit this peak → excursion happened
GN_RECENT_HIGH     = 15.0     # large steady-state gradient steps
RHO_BAND_LO        = 0.8      # IS ratio band
RHO_BAND_HI        = 1.3
RHO_IN_BAND_MIN    = 0.80     # <80% in band → trust-region issues
DRAINED_PER_CYCLE  = 500      # sync pacing should keep this near zero
GOAL_DIFF_DELTA    = 0.5      # per-cycle change to call "trending"


def _num(row: dict, key: str, *, as_int: bool = False):
    v = row.get(key, "")
    if v == "" or v is None:
        return None
    return int(v) if as_int else float(v)


def _mean(seq):
    seq = [v for v in seq if v is not None]
    return sum(seq) / len(seq) if seq else 0.0


def _mean_abs(seq):
    return _mean([abs(v) for v in seq if v is not None])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir",
                   help="run directory containing metrics.csv and training.log")
    p.add_argument("--last-n", type=int, default=20,
                   help="cycles included in the 'Recent' section (default 20)")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    csv_path = run_dir / "metrics.csv"
    log_path = run_dir / "training.log"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        return 1

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    if n == 0:
        print("metrics.csv has no data rows (only header?)")
        return 1

    has_goals = "goals_for" in rows[0]

    # ── Header ───────────────────────────────────────────────────────────
    sep = "=" * 78
    print(sep)
    print(f"Run analysis: {run_dir}")
    print(sep)
    wallclock_s = _num(rows[-1], "wallclock") or 0.0
    total_frames = _num(rows[-1], "total_frames", as_int=True) or 0
    num_envs = _num(rows[-1], "num_envs", as_int=True) or "?"
    cycles_per_hour = (n * 3600 / wallclock_s) if wallclock_s > 0 else 0.0
    print(f"  Cycles:        {n}")
    print(f"  Wall-clock:    {wallclock_s/60:.1f} min ({wallclock_s/3600:.2f} h)")
    print(f"  Total frames:  {total_frames:,}")
    print(f"  Cycles/hour:   {cycles_per_hour:.1f}")
    print(f"  N (envs):      {num_envs}")
    print(f"  Goal tracking: {'yes' if has_goals else 'no (legacy run)'}")
    print()

    # ── Health ───────────────────────────────────────────────────────────
    print("-- Health --")
    tags = [r.get("update_tag", "?") for r in rows]
    upd = tags.count("UPD")
    skp = tags.count("skp")
    col = tags.count("col")
    skp_frac = skp / n if n else 0.0
    print(f"  Updates:       UPD={upd}  skp={skp}  col={col}  "
          f"(skp_rate={100*skp_frac:.1f}%)")

    drained = [_num(r, "drained_states", as_int=True) or 0 for r in rows]
    print(f"  Drained:       total={sum(drained)}  "
          f"mean={_mean(drained):.0f}/cycle  max={max(drained)}")

    timeouts = fails = errors = 0
    if log_path.exists():
        text = log_path.read_text(errors="replace")
        timeouts = text.count("TIMEOUT")
        fails    = text.count("FAIL")
        errors   = text.count("ERROR")
    print(f"  Log warnings:  TIMEOUT={timeouts}  FAIL={fails}  ERROR={errors}"
          f"{'  (training.log missing)' if not log_path.exists() else ''}")

    rho_means = [_num(r, "rho_mean") for r in rows]
    rho_in_band = (
        sum(1 for v in rho_means if v is not None and RHO_BAND_LO < v < RHO_BAND_HI)
        / n
    )
    print(f"  rho_mean:      range [{min(v for v in rho_means if v is not None):.3f}, "
          f"{max(v for v in rho_means if v is not None):.3f}]  "
          f"in [{RHO_BAND_LO},{RHO_BAND_HI}] for {100*rho_in_band:.0f}% of cycles")

    kl_peak = max((_num(r, "loss_kl_teacher") or 0) for r in rows)
    gn_peak = max((_num(r, "grad_norm") or 0) for r in rows)
    print(f"  Peaks:         kl={kl_peak:.2f}   |gn|={gn_peak:.2f}")
    print()

    # ── Decile curves ────────────────────────────────────────────────────
    print(f"-- Learning curves (by decile of {n} cycles) --")
    if has_goals:
        hdr = (f"  {'cyc':>10s} | {'reward':>7s} {'gf':>4s} {'ga':>4s} "
               f"{'diff':>5s} {'rst':>4s} | {'evt':>4s} {'kl':>5s} {'|gn|':>5s} "
               f"{'ppo':>6s} {'rho':>5s}")
    else:
        hdr = (f"  {'cyc':>10s} | {'reward':>7s} | {'evt':>4s} {'kl':>5s} "
               f"{'|gn|':>5s} {'ppo':>6s} {'rho':>5s}")
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")

    n_bins = min(10, n)
    for b in range(n_bins):
        s = b * n // n_bins
        e = (b + 1) * n // n_bins
        sub = rows[s:e]
        if not sub:
            continue
        rs   = _mean([_num(r, "reward_sum") for r in sub])
        evts = _mean([_num(r, "reward_events") for r in sub])
        kl   = _mean([_num(r, "loss_kl_teacher") for r in sub])
        gn   = _mean([_num(r, "grad_norm") for r in sub])
        ppo  = _mean_abs([_num(r, "loss_ppo") for r in sub])
        rho  = _mean([_num(r, "rho_mean") for r in sub])
        if has_goals:
            gf = sum(_num(r, "goals_for", as_int=True) or 0 for r in sub)
            ga = sum(_num(r, "goals_against", as_int=True) or 0 for r in sub)
            rst = sum(_num(r, "matches_reset", as_int=True) or 0 for r in sub)
            print(f"  {s:>4d}-{e-1:<4d}  | {rs:+7.2f} {gf:>4d} {ga:>4d} "
                  f"{gf-ga:+5d} {rst:>4d} | {evts:>4.0f} {kl:>5.2f} {gn:>5.2f} "
                  f"{ppo:>6.3f} {rho:>5.3f}")
        else:
            print(f"  {s:>4d}-{e-1:<4d}  | {rs:+7.2f} | {evts:>4.0f} {kl:>5.2f} "
                  f"{gn:>5.2f} {ppo:>6.3f} {rho:>5.3f}")
    print()

    # ── Recent window vs early window ────────────────────────────────────
    L = min(args.last_n, n)
    recent = rows[-L:]
    early = rows[:L] if n >= 2 * L else []   # only show delta if windows don't overlap
    print(f"-- Recent {L} cycles" +
          (f"  (vs first {L})" if early else "  (run too short for vs-early)") + " --")

    def m(window, key, abs_=False):
        vals = [_num(r, key) for r in window]
        return _mean_abs(vals) if abs_ else _mean(vals)

    def sum_int(window, key):
        return sum(_num(r, key, as_int=True) or 0 for r in window)

    def show(label, recent_val, early_val=None, fmt="{:+.2f}"):
        line = f"  {label:<20s} {fmt.format(recent_val)}"
        if early_val is not None:
            delta = recent_val - early_val
            line += f"   (early {fmt.format(early_val)}, delta {fmt.format(delta)})"
        print(line)

    show("reward_sum mean:", m(recent, "reward_sum"),
         m(early, "reward_sum") if early else None)

    if has_goals:
        gf_r = sum_int(recent, "goals_for")
        ga_r = sum_int(recent, "goals_against")
        rst_r = sum_int(recent, "matches_reset")
        gd_r = (gf_r - ga_r) / L
        print(f"  goals (recent):      {gf_r} for / {ga_r} against / "
              f"diff {gf_r-ga_r:+d}  ({gd_r:+.2f}/cycle)  matches_reset={rst_r}")
        if early:
            gf_e = sum_int(early, "goals_for")
            ga_e = sum_int(early, "goals_against")
            gd_e = (gf_e - ga_e) / L
            print(f"  goals (early):       {gf_e} for / {ga_e} against / "
                  f"diff {gf_e-ga_e:+d}  ({gd_e:+.2f}/cycle)")
            print(f"  goal_diff/cycle delta: {gd_r - gd_e:+.2f}")

    show("kl_teacher mean:",   m(recent, "loss_kl_teacher"),
         m(early, "loss_kl_teacher") if early else None, fmt="{:.2f}")
    show("grad_norm mean:",    m(recent, "grad_norm"),
         m(early, "grad_norm") if early else None, fmt="{:.2f}")
    show("|ppo loss| mean:",   m(recent, "loss_ppo", abs_=True),
         m(early, "loss_ppo", abs_=True) if early else None, fmt="{:.3f}")
    show("value loss mean:",   m(recent, "loss_value"),
         m(early, "loss_value") if early else None, fmt="{:.4f}")
    show("collect_s mean:",    m(recent, "collect_s"),
         m(early, "collect_s") if early else None, fmt="{:.1f}s")
    show("update_s mean:",     m(recent, "update_s"),
         m(early, "update_s") if early else None, fmt="{:.1f}s")
    show("act_avg_ms mean:",   m(recent, "act_avg_ms"),
         m(early, "act_avg_ms") if early else None, fmt="{:.1f}ms")
    print()

    # ── Verdict ──────────────────────────────────────────────────────────
    print("-- Verdict --")
    good, warn = [], []

    if skp_frac > SKP_HIGH_FRAC:
        warn.append(f"HIGH skip rate {100*skp_frac:.0f}% -- policy may have collapsed "
                    f"(zero reward events most cycles)")
    elif skp_frac > SKP_WATCH_FRAC:
        warn.append(f"Elevated skip rate {100*skp_frac:.0f}% -- usually transient, watch it")
    else:
        good.append(f"Skip rate fine ({100*skp_frac:.0f}%)")

    recent_kl = m(recent, "loss_kl_teacher")
    if recent_kl > KL_RECENT_HIGH:
        warn.append(f"KL teacher elevated in recent window: {recent_kl:.2f} "
                    f"(consider raising --kl-teacher next run)")
    elif kl_peak > KL_PEAK_WARN and recent_kl == 0.0:
        # Frozen, not settled.  STUCK warning (if present) covers root cause.
        warn.append(f"KL teacher peaked at {kl_peak:.2f} then dropped to 0.00 -- "
                    f"policy frozen (no updates firing), NOT converged")
    elif kl_peak > KL_PEAK_WARN:
        warn.append(f"KL teacher had a peak of {kl_peak:.2f} -- excursion happened; "
                    f"OK if recent has settled (it's at {recent_kl:.2f})")
    else:
        good.append(f"KL teacher in healthy band (recent {recent_kl:.2f}, peak {kl_peak:.2f})")

    recent_gn = m(recent, "grad_norm")
    if recent_gn > GN_RECENT_HIGH:
        warn.append(f"grad_norm elevated recently: {recent_gn:.2f} -- large updates")
    elif recent_gn == 0.0 and stuck:
        # Don't report 0.0 as "in band" when the policy is frozen.
        pass
    else:
        good.append(f"grad_norm in band (recent {recent_gn:.2f})")

    if rho_in_band < RHO_IN_BAND_MIN:
        warn.append(f"rho out of trust region: only {100*rho_in_band:.0f}% in "
                    f"[{RHO_BAND_LO},{RHO_BAND_HI}]")
    else:
        good.append(f"rho in trust region {100*rho_in_band:.0f}% of cycles")

    drained_mean = _mean(drained)
    if drained_mean > DRAINED_PER_CYCLE:
        warn.append(f"drained mean {drained_mean:.0f}/cycle is high -- sync pacing slipping?")
    else:
        good.append(f"drained low ({drained_mean:.0f}/cycle)")

    if timeouts > 0:
        warn.append(f"{timeouts} sync-wait TIMEOUT(s) in training.log -- Python "
                    f"missed the 200ms watchdog this many times")
    if fails > 0 or errors > 0:
        warn.append(f"FAIL/ERROR markers in training.log "
                    f"(FAIL={fails} ERROR={errors})")

    # Detect "stuck emulator": recent window has ~zero events AND zero
    # matches completing.  This is the FPS=0 wedged-Dolphin pattern --
    # socket alive, IPC responding, but the game CPU thread is frozen
    # post-savestate-load.  Override the goal_diff / KL "trending up"
    # verdicts in this case: their math is meaningless when recent is
    # all degenerate zeros.
    recent_events_mean = m(recent, "reward_events")
    recent_resets_total = sum_int(recent, "matches_reset") if has_goals else None
    stuck = (
        len(recent) >= 20
        and recent_events_mean < 1.0
        and (recent_resets_total == 0 if has_goals else True)
    )
    if stuck:
        warn.append(
            f"STUCK: recent {L} cycles produced ~0 reward events"
            + (f" and 0 matches completed" if has_goals else "")
            + ". Emulator likely wedged (FPS=0 with live socket). batch_env "
            "now has frame_id-stagnation auto-restart -- check that you're "
            "on a build with it, and inspect *.crashed*.log archives."
        )
    if has_goals and early:
        gf_r = sum_int(recent, "goals_for")
        ga_r = sum_int(recent, "goals_against")
        gf_e = sum_int(early, "goals_for")
        ga_e = sum_int(early, "goals_against")
        gd_r = (gf_r - ga_r) / L
        gd_e = (gf_e - ga_e) / L
        if stuck:
            pass   # don't celebrate "trending up" when recent is degenerate
        elif gd_r - gd_e > GOAL_DIFF_DELTA:
            good.append(f"goal_diff trending UP: {gd_e:+.2f} -> {gd_r:+.2f}/cycle")
        elif gd_r - gd_e < -GOAL_DIFF_DELTA:
            warn.append(f"goal_diff trending DOWN: {gd_e:+.2f} -> {gd_r:+.2f}/cycle "
                        f"(policy may be regressing)")
        else:
            good.append(f"goal_diff roughly stable: {gd_e:+.2f} -> {gd_r:+.2f}/cycle")
    elif has_goals and not stuck:
        gf_r = sum_int(recent, "goals_for")
        ga_r = sum_int(recent, "goals_against")
        good.append(f"goal_diff (recent only): {(gf_r-ga_r)/L:+.2f}/cycle")

    for g in good:
        print(f"  [ok] {g}")
    for w in warn:
        print(f"  [!]  {w}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
