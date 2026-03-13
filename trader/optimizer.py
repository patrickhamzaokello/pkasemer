#!/usr/bin/env python3
"""
Pknwitq Signal Optimizer

Reads resolved trade outcomes from SQLite, runs multi-dimensional analysis
across score, entry price, hour, and regime dimensions, then makes small,
evidence-based adjustments to config.json.

Design principles:
  - Payout structure is protected first: wins must structurally cover losses.
  - Minimum trade count required before any dimension change fires.
  - Max step size per run (converges gradually, never overfits on 5 trades).
  - All changes bounded — no parameter can leave a safe operating range.
  - Every applied change is logged with the evidence that drove it.

Modes:
    python optimizer.py              # analyse only, print report
    python optimizer.py --apply      # analyse + write config.json
    python optimizer.py --watch 30   # re-run every 30 min (implies --apply)
    python optimizer.py --report     # detailed breakdown, no changes

From scheduler:
    from optimizer import run_optimizer
    run_optimizer(apply=True)
"""

import os
import json
import time
import sqlite3
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR     = Path("/data") if Path("/data").exists() else Path(__file__).parent.parent / "data"
_IMAGE_CONFIG = Path(__file__).parent / "config.json"
_DATA_CONFIG  = _DATA_DIR / "config.json"
_OPT_LOG      = Path(__file__).parent / "optimizer.log"

# ---------------------------------------------------------------------------
# Section routing — maps optimizer-tunable parameters to their config section.
# Used when writing changes back to the nested config structure.
# ---------------------------------------------------------------------------

_PARAM_SECTION = {
    # signal section
    "composite_threshold":        "signal",
    "slow_composite_threshold":   "signal",
    "active_composite_threshold": "signal",
    "min_momentum_pct":           "signal",
    "max_volatility_5m":          "signal",
    "blocked_hours":              "signal",
    "boosted_hours":              "signal",
    # trading section
    "max_entry_yes":              "trading",
    "min_entry_yes":              "trading",
    "min_entry_no":               "trading",
    "max_entry_no":               "trading",
    "slow_max_entry_yes":         "trading",
    "slow_min_entry_no":          "trading",
    "min_payout_ratio":           "trading",
    "slow_min_payout_ratio":      "trading",
    "max_position":               "trading",
    "slow_position_pct_cap":      "signal",
    "normal_position_pct_cap":    "signal",
    "active_position_pct_cap":    "signal",
    "rsi_overbought":             "signal",
    "rsi_oversold":               "signal",
    "max_no_score":               "trading",
}

# ---------------------------------------------------------------------------
# Safety bounds — optimizer can never push a parameter outside these limits.
# The payout-related bounds are deliberately tight to protect profitability.
# "wins must cover losses" requires min_payout_ratio >= 1.05 always.
# ---------------------------------------------------------------------------

BOUNDS = {
    # Signal quality thresholds
    "composite_threshold":          (0.65, 0.90),
    "slow_composite_threshold":     (0.68, 0.92),
    "active_composite_threshold":   (0.60, 0.82),

    # Entry price — payout ratio = (1-p)/p.
    # At 0.476: payout=1.10x. At 0.42: payout=1.38x.
    # Lower bound 0.40 ensures we're never buying with < 1.50x payout floor.
    # Upper bound 0.490 ensures every YES entry has ≥ 1.04x payout.
    "max_entry_yes":                (0.40, 0.490),
    "min_entry_yes":                (0.25, 0.50),
    "min_entry_no":                 (0.20, 0.50),
    "max_entry_no":                 (0.50, 0.75),
    "slow_max_entry_yes":           (0.38, 0.485),
    "slow_min_entry_no":            (0.515, 0.62),

    # Payout ratio — NEVER below 1.05 (wins must cover losses structurally).
    "min_payout_ratio":             (1.05, 1.60),
    "slow_min_payout_ratio":        (1.08, 1.80),

    # Momentum and volatility filters
    "min_momentum_pct":             (0.03, 0.40),
    "max_volatility_5m":            (0.50, 5.00),

    # Position sizing caps
    "max_position":                 (1.00, 25.00),
    "slow_position_pct_cap":        (0.30, 1.00),
    "normal_position_pct_cap":      (0.50, 1.00),
    "active_position_pct_cap":      (0.50, 1.00),

    # RSI extremes
    "rsi_overbought":               (75,   98),
    "rsi_oversold":                 (2,    25),

    # Score cap for NO side
    "max_no_score":                 (0.22, 0.40),
}

# Minimum resolved trades before making any change
MIN_TRADES_OVERALL  = 15   # global changes (composite_threshold)
MIN_TRADES_BUCKET   = 5    # per-bucket changes (score/price/hour analysis)
MIN_TRADES_HOUR     = 4    # per-hour block/boost decisions

# Max step size per optimizer run — keeps convergence gradual
MAX_STEP = {
    "composite_threshold":          0.02,
    "slow_composite_threshold":     0.02,
    "active_composite_threshold":   0.02,
    "max_entry_yes":                0.01,
    "min_payout_ratio":             0.05,
    "slow_min_payout_ratio":        0.05,
    "min_momentum_pct":             0.02,
    "max_no_score":                 0.02,
}

TARGET_WIN_RATE      = 0.70
MIN_WIN_RATE         = 0.60   # below this = tighten thresholds
MAX_WIN_RATE         = 0.82   # above this = can relax slightly
MIN_PNL_RATIO        = 1.00   # avg_win / avg_loss must stay above this
TARGET_PNL_RATIO     = 1.10   # aim for wins covering losses by 10%

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _find_db():
    candidates = [
        str(_DATA_DIR / "signal_research.db"),
        "/data/signal_research.db",
        str(Path(__file__).parent.parent / "data" / "signal_research.db"),
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def _load_trades(db_path):
    """
    Load all resolved trades. Returns list of dicts with all relevant columns.
    Columns: side, entry_price, position_size, pnl, outcome, score,
             hour_utc, momentum_5m, vs_ref, poly_yes_price
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT side, entry_price, position_size, pnl,
                   trade_outcome AS outcome, score,
                   hour_utc, momentum_5m, vs_ref, poly_yes_price
            FROM trades
            WHERE resolved = 1
              AND trade_outcome IN ('win', 'loss')
            ORDER BY id ASC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return []


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _win_rate(group):
    if not group:
        return None
    return sum(1 for t in group if t["outcome"] == "win") / len(group)


def _avg(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def analyze_pnl_structure(trades):
    """
    Core profitability check.
    Returns dict: win_rate, avg_win_pnl, avg_loss_pnl, pnl_ratio, total_pnl
    pnl_ratio = avg_win_pnl / avg_loss_pnl  (must stay > 1.0)
    """
    wins   = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    if not trades:
        return None

    win_rate    = len(wins) / len(trades)
    avg_win     = _avg([t["pnl"] for t in wins])   or 0.0
    avg_loss    = _avg([abs(t["pnl"]) for t in losses]) or 0.0
    pnl_ratio   = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    total_pnl   = sum(t["pnl"] for t in trades if t["pnl"] is not None)
    expected_ev = win_rate * avg_win - (1 - win_rate) * avg_loss

    return {
        "n":           len(trades),
        "wins":        len(wins),
        "losses":      len(losses),
        "win_rate":    win_rate,
        "avg_win_pnl": avg_win,
        "avg_loss_pnl": avg_loss,
        "pnl_ratio":   pnl_ratio,
        "total_pnl":   total_pnl,
        "expected_ev": expected_ev,
    }


def analyze_by_score_bucket(trades, bucket_size=0.05):
    """
    Group trades by |score - 0.5| * 2 (signal strength in [0,1]).
    Returns list of (bucket_label, n, win_rate) sorted by strength ascending.
    Used to find the lowest-conviction bucket that still wins.
    """
    buckets = defaultdict(list)
    for t in trades:
        s = t.get("score")
        if s is None:
            continue
        strength = abs(s - 0.5) * 2   # maps 0.5→0, 1.0→1
        bucket   = round(int(strength / bucket_size) * bucket_size, 3)
        buckets[bucket].append(t)

    result = []
    for b in sorted(buckets.keys()):
        group = buckets[b]
        wr    = _win_rate(group)
        result.append({
            "bucket":    f"{b:.2f}-{b+bucket_size:.2f}",
            "min_score": 0.5 + b / 2,   # YES score equivalent
            "n":         len(group),
            "win_rate":  wr,
        })
    return result


def analyze_by_entry_price(trades):
    """
    Group trades by entry_price bucket (0.05 wide).
    Returns list of (bucket, n, win_rate, avg_pnl_ratio).
    """
    buckets = defaultdict(list)
    for t in trades:
        p = t.get("entry_price")
        if p is None:
            continue
        bucket = round(int(p / 0.05) * 0.05, 3)
        buckets[bucket].append(t)

    result = []
    for b in sorted(buckets.keys()):
        group    = buckets[b]
        wr       = _win_rate(group)
        pnl_vals = [t["pnl"] for t in group if t["pnl"] is not None]
        avg_pnl  = _avg(pnl_vals)
        result.append({
            "bucket":   f"{b:.2f}-{b+0.05:.2f}",
            "n":        len(group),
            "win_rate": wr,
            "avg_pnl":  avg_pnl,
        })
    return result


def analyze_by_hour(trades):
    """
    Win rate per UTC hour (0-23).
    Returns dict: {hour_int: {"n": int, "win_rate": float}}
    """
    by_hour = defaultdict(list)
    for t in trades:
        h = t.get("hour_utc")
        if h is not None:
            by_hour[int(h)].append(t)

    result = {}
    for h in range(24):
        group = by_hour[h]
        if group:
            result[h] = {"n": len(group), "win_rate": _win_rate(group)}
    return result


def analyze_by_side(trades):
    """Win rate and P&L breakdown for YES vs NO trades."""
    yes_trades = [t for t in trades if t.get("side") == "yes"]
    no_trades  = [t for t in trades if t.get("side") == "no"]
    return {
        "yes": analyze_pnl_structure(yes_trades) if yes_trades else None,
        "no":  analyze_pnl_structure(no_trades)  if no_trades  else None,
    }


# ---------------------------------------------------------------------------
# Adjustment engine
# ---------------------------------------------------------------------------

def _clamp(value, param_name):
    lo, hi = BOUNDS.get(param_name, (float("-inf"), float("inf")))
    return max(lo, min(hi, value))


def _step(current, delta, param_name):
    """Apply delta, respect max step size, then clamp to bounds."""
    max_s  = MAX_STEP.get(param_name, 0.05)
    delta  = max(-max_s, min(max_s, delta))
    return _clamp(current + delta, param_name)


def compute_adjustments(overall, score_buckets, price_buckets, hour_analysis,
                        side_analysis, config):
    """
    Produce a list of (param, new_value, reason) tuples.
    All proposed values are already bounded.
    """
    changes  = []
    n        = overall["n"]
    wr       = overall["win_rate"]
    pr       = overall["pnl_ratio"]
    # Read from domain sections where available, fall back to flat config
    _sig     = config.get("signal",  config)
    _trd     = config.get("trading", config)
    cur_thr  = _sig.get("composite_threshold", 0.70)
    cur_mpr  = _trd.get("min_payout_ratio", 1.10)
    cur_smpr = _trd.get("slow_min_payout_ratio", 1.15)
    cur_mey  = _trd.get("max_entry_yes", 0.476)

    # ── 1. Payout structure (runs regardless of trade count if we have wins/losses)
    # This is the most critical check: avg win must cover avg loss.
    if overall["wins"] >= 3 and overall["losses"] >= 3:
        if pr < MIN_PNL_RATIO:
            # Wins not covering losses → tighten entry price (lower max_entry_yes)
            # At lower entry price, payout ratio improves structurally.
            new_mey = _step(cur_mey, -0.01, "max_entry_yes")
            if new_mey < cur_mey:
                changes.append((
                    "max_entry_yes", new_mey,
                    f"pnl_ratio={pr:.3f} < {MIN_PNL_RATIO} — tighten entry price "
                    f"to improve payout (wins not covering losses)"
                ))
            # Also raise min_payout_ratio target
            new_mpr = _step(cur_mpr, 0.05, "min_payout_ratio")
            if new_mpr > cur_mpr:
                changes.append((
                    "min_payout_ratio", new_mpr,
                    f"pnl_ratio={pr:.3f} below target — raise payout floor"
                ))

        elif pr > TARGET_PNL_RATIO * 1.5 and wr > TARGET_WIN_RATE:
            # Very high payout + high win rate → slightly relax entry price cap
            new_mey = _step(cur_mey, +0.005, "max_entry_yes")
            if new_mey > cur_mey:
                changes.append((
                    "max_entry_yes", new_mey,
                    f"pnl_ratio={pr:.3f} healthy, wr={wr:.1%} — minor relaxation"
                ))

    # ── 2. Win rate calibration (needs MIN_TRADES_OVERALL)
    if n >= MIN_TRADES_OVERALL:
        if wr < MIN_WIN_RATE:
            new_thr = _step(cur_thr, +0.02, "composite_threshold")
            if new_thr > cur_thr:
                changes.append((
                    "composite_threshold", new_thr,
                    f"win_rate={wr:.1%} < {MIN_WIN_RATE:.0%} target "
                    f"— raising threshold to filter weak signals"
                ))

        elif wr > MAX_WIN_RATE and pr >= TARGET_PNL_RATIO:
            # High win rate + good payout = can accept slightly weaker signals
            new_thr = _step(cur_thr, -0.01, "composite_threshold")
            if new_thr < cur_thr:
                changes.append((
                    "composite_threshold", new_thr,
                    f"win_rate={wr:.1%} > {MAX_WIN_RATE:.0%}, pnl_ratio={pr:.2f} "
                    f"— minor threshold relaxation"
                ))

    # ── 3. Score bucket analysis — raise threshold if low-conviction bucket is losing
    if n >= MIN_TRADES_OVERALL * 2:
        # Find lowest score bucket (nearest to threshold) with enough trades
        for bucket_info in score_buckets:
            if bucket_info["n"] < MIN_TRADES_BUCKET:
                continue
            bwr = bucket_info["win_rate"]
            b_score = bucket_info["min_score"]  # equivalent YES score
            # This bucket is near our entry threshold and losing — raise threshold
            if bwr is not None and bwr < 0.58 and b_score < cur_thr + 0.10:
                # The weakest-conviction bucket is a drag — raise threshold past it
                new_thr = _step(cur_thr, +0.02, "composite_threshold")
                if new_thr > cur_thr:
                    changes.append((
                        "composite_threshold", new_thr,
                        f"score bucket {bucket_info['bucket']} has wr={bwr:.1%} "
                        f"(N={bucket_info['n']}) — raising threshold above this bucket"
                    ))
                break  # only act on the weakest bucket once per run

    # ── 4. Entry price analysis — check if high entry prices are losing more
    if n >= MIN_TRADES_OVERALL:
        # Find the highest entry price bucket with enough trades and bad win rate
        for pb in reversed(price_buckets):
            if pb["n"] < MIN_TRADES_BUCKET:
                continue
            pwr = pb["win_rate"]
            # Parse bucket upper bound
            try:
                p_upper = float(pb["bucket"].split("-")[1])
            except Exception:
                continue
            # High entry price with bad win rate → tighten max_entry_yes
            if pwr is not None and pwr < 0.55 and p_upper > cur_mey - 0.03:
                new_mey = _step(cur_mey, -0.01, "max_entry_yes")
                if new_mey < cur_mey:
                    changes.append((
                        "max_entry_yes", new_mey,
                        f"entry price bucket {pb['bucket']} has wr={pwr:.1%} "
                        f"(N={pb['n']}) — tighten entry price cap"
                    ))
                break  # one change per run

    # ── 5. Hour analysis — block bad hours, boost good ones
    if n >= MIN_TRADES_OVERALL:
        blocked = list(_sig.get("blocked_hours", [6, 8, 16]))
        boosted = {int(k): v for k, v in
                   _sig.get("boosted_hours", {}).items()}
        hour_changes = False

        for h, hdata in hour_analysis.items():
            hn  = hdata["n"]
            hwr = hdata["win_rate"]
            if hn < MIN_TRADES_HOUR:
                continue

            if hwr < 0.45 and h not in blocked:
                blocked.append(h)
                # Remove from boosted if it was there
                boosted.pop(h, None)
                hour_changes = True
                changes.append((
                    "_blocked_hours_add", h,
                    f"hour {h:02d}h has wr={hwr:.1%} (N={hn}) — adding to blocked_hours"
                ))

            elif hwr > 0.77 and hn >= MIN_TRADES_HOUR * 2 and h not in blocked:
                # Good hour — ensure it has a boost
                current_delta = boosted.get(h, 0.0)
                if current_delta > -0.03:
                    boosted[h] = round(current_delta - 0.01, 3)
                    hour_changes = True
                    changes.append((
                        "_boosted_hours_add", h,
                        f"hour {h:02d}h has wr={hwr:.1%} (N={hn}) — adding/improving boost"
                    ))

            elif hwr > 0.50 and h in blocked:
                # Hour was blocked but is now performing okay — unblock
                blocked.remove(h)
                hour_changes = True
                changes.append((
                    "_blocked_hours_remove", h,
                    f"hour {h:02d}h now has wr={hwr:.1%} (N={hn}) — removing block"
                ))

        if hour_changes:
            changes.append(("blocked_hours", sorted(set(blocked)), "hour analysis update"))
            changes.append(("boosted_hours",
                            {str(k): v for k, v in sorted(boosted.items())},
                            "hour analysis update"))

    # ── 6. NO side underperformance — tighten max_no_score
    no_stats = side_analysis.get("no")
    if no_stats and no_stats["n"] >= MIN_TRADES_BUCKET:
        no_wr = no_stats["win_rate"]
        cur_mns = _trd.get("max_no_score", 0.30)
        if no_wr < 0.50:
            new_mns = _step(cur_mns, -0.02, "max_no_score")
            if new_mns < cur_mns:
                changes.append((
                    "max_no_score", new_mns,
                    f"NO-side win_rate={no_wr:.1%} (N={no_stats['n']}) "
                    f"— tighten max_no_score to filter weaker NO signals"
                ))
        elif no_wr > 0.78 and no_stats["n"] >= MIN_TRADES_BUCKET * 2:
            new_mns = _step(cur_mns, +0.01, "max_no_score")
            if new_mns > cur_mns:
                changes.append((
                    "max_no_score", new_mns,
                    f"NO-side win_rate={no_wr:.1%} strong — minor relaxation"
                ))

    # ── Deduplicate: keep first occurrence of each param key
    seen = set()
    deduped = []
    for item in changes:
        key = item[0]
        if key.startswith("_"):
            continue   # internal markers, already merged into blocked/boosted
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def _load_config():
    path = _DATA_CONFIG if _DATA_CONFIG.exists() else _IMAGE_CONFIG
    try:
        with open(path) as f:
            return json.load(f), path
    except Exception as e:
        return {}, None


def _set_config_key(config, key, value):
    """Set a config key in the correct section (nested) or at top-level (flat)."""
    section = _PARAM_SECTION.get(key)
    if section and section in config and isinstance(config[section], dict):
        config[section][key] = value
    else:
        config[key] = value


def _save_config(config):
    """Write to /data/config.json (persistent volume). Never touches /app/config.json."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DATA_CONFIG, "w") as f:
        json.dump(config, f, indent=2)


def _log_changes(changes, overall):
    try:
        with open(_OPT_LOG, "a") as f:
            ts  = datetime.now(timezone.utc).isoformat()
            wr  = overall["win_rate"]
            pr  = overall["pnl_ratio"]
            pnl = overall["total_pnl"]
            f.write(
                f"\n[{ts}] n={overall['n']} win_rate={wr:.1%} "
                f"pnl_ratio={pr:.3f} total_pnl={pnl:+.2f}\n"
            )
            for param, value, reason in changes:
                f.write(f"  {param} = {value!r}  ({reason})\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_optimizer(apply=False, verbose=True):
    """
    Main entry point — called by scheduler.py every 30 minutes.

    apply=True  → writes config.json and logs changes.
    apply=False → analysis-only (dry run), prints proposed changes.
    """
    db_path = _find_db()
    if not db_path:
        if verbose:
            print("[optimizer] No signal_research.db found — skipping")
        return

    trades = _load_trades(db_path)
    if not trades:
        if verbose:
            print("[optimizer] No resolved trades in DB — skipping")
        return

    config, cfg_path = _load_config()
    if not config:
        if verbose:
            print("[optimizer] Could not load config — skipping")
        return

    # ── Run all analyses ──────────────────────────────────────────────────────
    overall       = analyze_pnl_structure(trades)
    score_buckets = analyze_by_score_bucket(trades)
    price_buckets = analyze_by_entry_price(trades)
    hour_analysis = analyze_by_hour(trades)
    side_analysis = analyze_by_side(trades)

    if verbose:
        wr  = overall["win_rate"]
        pr  = overall["pnl_ratio"]
        ev  = overall["expected_ev"]
        pnl = overall["total_pnl"]
        print(f"\n[optimizer] ── Resolved trades: {overall['n']} "
              f"({overall['wins']}W / {overall['losses']}L)")
        print(f"  Win rate:    {wr:.1%}   (target {TARGET_WIN_RATE:.0%})")
        print(f"  P&L ratio:   {pr:.3f}   (avg win / avg loss — must be ≥1.0)")
        print(f"  Expected EV: {ev:+.3f}  per trade")
        print(f"  Total P&L:   {pnl:+.2f} USDC")

        if score_buckets:
            print(f"\n  Score bucket win rates (YES-equivalent):")
            for sb in score_buckets:
                if sb["n"] >= MIN_TRADES_BUCKET:
                    print(f"    {sb['bucket']}  wr={sb['win_rate']:.1%}  N={sb['n']}")

        if price_buckets:
            print(f"\n  Entry price win rates:")
            for pb in price_buckets:
                if pb["n"] >= MIN_TRADES_BUCKET:
                    avg_s = f"{pb['avg_pnl']:+.3f}" if pb["avg_pnl"] is not None else "n/a"
                    print(f"    {pb['bucket']}  wr={pb['win_rate']:.1%}  N={pb['n']}  avg_pnl={avg_s}")

        if hour_analysis:
            print(f"\n  Hour win rates (UTC, min {MIN_TRADES_HOUR} trades):")
            for h in sorted(hour_analysis.keys()):
                hd = hour_analysis[h]
                if hd["n"] >= MIN_TRADES_HOUR:
                    blocked = h in _sig.get("blocked_hours", config.get("blocked_hours", []))
                    tag = " [BLOCKED]" if blocked else ""
                    print(f"    {h:02d}h  wr={hd['win_rate']:.1%}  N={hd['n']}{tag}")

        for label, stats in [("YES", side_analysis.get("yes")),
                              ("NO",  side_analysis.get("no"))]:
            if stats:
                print(f"\n  {label} trades: {stats['n']}  "
                      f"wr={stats['win_rate']:.1%}  "
                      f"pnl_ratio={stats['pnl_ratio']:.3f}  "
                      f"total={stats['total_pnl']:+.2f}")

    # ── Generate adjustments ──────────────────────────────────────────────────
    changes = compute_adjustments(
        overall, score_buckets, price_buckets, hour_analysis, side_analysis, config
    )

    if not changes:
        if verbose:
            print(f"\n[optimizer] No adjustments needed.")
        return

    if verbose:
        print(f"\n[optimizer] Proposed changes ({len(changes)}):")
        _sig_disp = config.get("signal",  config)
        _trd_disp = config.get("trading", config)
        for param, value, reason in changes:
            section = _PARAM_SECTION.get(param)
            if section == "signal":
                cur = _sig_disp.get(param, "—")
            elif section == "trading":
                cur = _trd_disp.get(param, "—")
            else:
                cur = config.get(param, "—")
            print(f"  {param}: {cur!r} → {value!r}")
            print(f"    reason: {reason}")

    if apply:
        for param, value, _ in changes:
            _set_config_key(config, param, value)
        _save_config(config)
        _log_changes(changes, overall)
        if verbose:
            print(f"\n[optimizer] ✓ Config updated ({len(changes)} change(s))")
    else:
        if verbose:
            print(f"\n[optimizer] Dry run — pass --apply to write changes")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pknwitq Signal Optimizer")
    parser.add_argument("--apply",  action="store_true",
                        help="Apply changes to config.json")
    parser.add_argument("--watch",  type=int, metavar="MINUTES",
                        help="Re-run every N minutes (implies --apply)")
    parser.add_argument("--report", action="store_true",
                        help="Detailed report, no changes")
    parser.add_argument("--quiet",  action="store_true",
                        help="Minimal output")
    args = parser.parse_args()

    _apply   = args.apply or (args.watch is not None)
    _verbose = not args.quiet

    if args.watch:
        print(f"[optimizer] Watching — running every {args.watch} minutes (Ctrl+C to stop)")
        while True:
            run_optimizer(apply=_apply, verbose=_verbose)
            time.sleep(args.watch * 60)
    else:
        run_optimizer(apply=_apply, verbose=_verbose)
