#!/usr/bin/env python3
"""
Pknwitq Signal Optimizer

Reads resolved trade outcomes from trade_log.json + SQLite, analyses winning vs
losing patterns across every tunable dimension, then incrementally updates
config.json to push win rate toward the target (default 70%).

Safety model
------------
- All changes are bounded (MAX/MIN guards on every parameter).
- Changes per run are small steps — the system converges gradually.
- Dry-run by default. Pass --apply to actually write config.json.
- Every applied change is appended to optimizer.log with explanation.

Modes
-----
    python optimizer.py              # analyse only, print report
    python optimizer.py --apply      # analyse + update config.json
    python optimizer.py --watch 30   # re-run every 30 minutes, --apply implied
    python optimizer.py --report     # detailed breakdown, no changes

Usage from scheduler
--------------------
    from optimizer import run_optimizer
    run_optimizer(apply=True)
"""

import os
import sys
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

_DATA_DIR   = Path("/data") if Path("/data").exists() else Path("data")
_CONFIG     = Path(__file__).parent / "config.json"
_TRADE_LOG  = _DATA_DIR / "trade_log.json"
_DB_PATH    = os.environ.get("DB_PATH", str(_DATA_DIR / "signal_research.db"))
_OPT_LOG    = _DATA_DIR / "optimizer.log"

# ---------------------------------------------------------------------------
# Safety bounds — optimizer will never go outside these
# ---------------------------------------------------------------------------

TARGET_WIN_RATE  = float(os.environ.get("OPT_TARGET_WIN_RATE", "0.70"))
MIN_SAMPLE       = 5      # minimum resolved trades before touching a parameter
STEP_THRESH      = 0.01   # max composite_threshold movement per run
MAX_THRESH       = 0.82
MIN_THRESH       = 0.62
MAX_MOM          = 0.18
MIN_MOM          = 0.06
STEP_MOM         = 0.01
MAX_BLOCK_HOURS  = 8      # never block more than 8 hours of the day
MAX_WEIGHT_NUDGE = 0.025  # max weight change per signal per run


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def _load_config():
    if _CONFIG.exists():
        try:
            return json.loads(_CONFIG.read_text())
        except Exception:
            return {}
    return {}


def _save_config(cfg):
    _CONFIG.write_text(json.dumps(cfg, indent=2))


def _append_log(lines: list[str]):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(_OPT_LOG, "a") as f:
        for line in lines:
            f.write(f"{ts}  {line}\n")


def _send_telegram_alert(cfg, changes_applied, overall_wr):
    """Send a Telegram message summarising optimizer changes. Non-fatal."""
    try:
        token = cfg.get("telegram_bot_token", "")
        chat_id = cfg.get("telegram_chat_id", "")
        if not token or not chat_id:
            return

        wr_pct = f"{overall_wr * 100:.1f}%" if overall_wr is not None else "n/a"
        target_pct = f"{TARGET_WIN_RATE * 100:.0f}%"
        n_changes = len(changes_applied)

        lines_html = ""
        for line in changes_applied[:8]:
            lines_html += f"\n• {line.strip()}"

        text = (
            f"⚙️ <b>OPTIMIZER UPDATE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Win rate: {wr_pct}  →  target {target_pct}\n"
            f"Changes: {n_changes}"
            + (f"\n{lines_html}" if lines_html else "")
        )

        import urllib.request as _urllib_req
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode()
        req = _urllib_req.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})
        with _urllib_req.urlopen(req, timeout=10):
            pass
    except Exception:
        pass


def _write_history(win_rate, n_resolved, changes_applied):
    """Append an optimizer run entry to optimizer_history.json. Non-fatal."""
    try:
        history_path = _DATA_DIR / "optimizer_history.json"
        if history_path.exists():
            try:
                entries = json.loads(history_path.read_text())
            except Exception:
                entries = []
        else:
            entries = []

        parsed_changes = []
        for raw in changes_applied:
            raw_stripped = raw.strip()
            category = ""
            cat_match = raw_stripped.find("[")
            cat_end = raw_stripped.find("]")
            if cat_match != -1 and cat_end != -1:
                category = raw_stripped[cat_match + 1:cat_end]
            parsed_changes.append({"category": category, "raw": raw_stripped})

        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
            "n": n_resolved,
            "changes": parsed_changes,
        }
        entries.append(entry)
        # Keep last 500 entries
        entries = entries[-500:]
        history_path.write_text(json.dumps(entries, indent=2))
    except Exception:
        pass


def _load_trades_json():
    if not _TRADE_LOG.exists():
        return []
    try:
        return json.loads(_TRADE_LOG.read_text())
    except Exception:
        return []


def _load_trades_db():
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM trades
            WHERE resolved = 1
              AND trade_outcome IN ('win', 'loss')
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def load_resolved_trades():
    """
    Merge trade_log.json and SQLite trades table.
    SQLite 'trade_outcome' maps to JSON 'outcome' field.
    Deduplicate by trade_id, preferring DB records (more up-to-date).
    """
    db_trades = _load_trades_db()
    for t in db_trades:
        if "outcome" not in t and "trade_outcome" in t:
            t["outcome"] = t["trade_outcome"]

    json_trades = _load_trades_json()

    # Index by trade_id; DB wins on conflict
    merged = {}
    for t in json_trades:
        tid = t.get("trade_id")
        if tid and t.get("outcome") in ("win", "loss"):
            merged[tid] = t
    for t in db_trades:
        tid = t.get("trade_id")
        if tid:
            merged[tid] = t   # DB overrides

    return list(merged.values())


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _win_rate(trades):
    resolved = [t for t in trades if t.get("outcome") in ("win", "loss")]
    if not resolved:
        return None, 0
    wins = sum(1 for t in resolved if t["outcome"] == "win")
    return wins / len(resolved), len(resolved)


def score_threshold_sweep(resolved):
    """
    Sweep score thresholds from high to low.
    Return the lowest threshold where win_rate >= TARGET and n >= MIN_SAMPLE.
    Returns None if no threshold achieves the target.
    """
    by_score = sorted(resolved, key=lambda t: t.get("score", 0.5), reverse=True)
    best_threshold = None
    for i in range(MIN_SAMPLE, len(by_score) + 1):
        subset = by_score[:i]
        wr, n = _win_rate(subset)
        if wr is not None and wr >= TARGET_WIN_RATE:
            best_threshold = subset[-1]["score"]
    return best_threshold


def hour_win_rates(resolved):
    buckets = defaultdict(list)
    for t in resolved:
        h = t.get("hour_utc")
        if h is not None:
            buckets[int(h)].append(t)
    return {h: _win_rate(trades) for h, trades in buckets.items()}


def momentum_threshold_sweep(resolved):
    """Return (best_threshold, win_rate_at_threshold)."""
    thresholds = [round(x * 0.01, 2) for x in range(6, 21)]  # 0.06 → 0.20
    best = None
    best_wr = 0.0
    for thresh in thresholds:
        subset = [t for t in resolved
                  if abs(t.get("momentum_5m", 0) or 0) >= thresh]
        wr, n = _win_rate(subset)
        if n >= MIN_SAMPLE and wr is not None and wr > best_wr:
            best_wr = wr
            best = thresh
    return best, best_wr


def entry_price_win_rates(resolved):
    """Win rate split: entry <= 0.50 vs entry > 0.50."""
    low  = [t for t in resolved if (t.get("entry_price") or 0.5) <= 0.50]
    high = [t for t in resolved if (t.get("entry_price") or 0.5) >  0.50]
    return {
        "<=0.50": _win_rate(low),
        ">0.50":  _win_rate(high),
    }


def signal_win_deltas(resolved):
    """
    For each raw signal stored at trade time, compute:
        mean(value | win) - mean(value | loss)
    Positive = higher values associated with wins.
    """
    signal_fields = [
        "score", "confidence", "momentum_5m", "momentum_1m", "momentum_15m",
        "vs_ref", "cex_poly_lag", "price_acceleration", "vol_adjusted_momentum",
        "volume_ratio", "rsi_14",
    ]
    wins   = [t for t in resolved if t.get("outcome") == "win"]
    losses = [t for t in resolved if t.get("outcome") == "loss"]
    deltas = {}
    for field in signal_fields:
        wv = [t.get(field) for t in wins   if t.get(field) is not None]
        lv = [t.get(field) for t in losses if t.get(field) is not None]
        if len(wv) >= 3 and len(lv) >= 3:
            deltas[field] = round(sum(wv) / len(wv) - sum(lv) / len(lv), 5)
    return deltas


def slow_market_win_rate(resolved):
    """Win rate for slow-regime trades (|m5| < 0.12)."""
    slow = [t for t in resolved if abs(t.get("momentum_5m", 0) or 0) < 0.12]
    return _win_rate(slow)


# ---------------------------------------------------------------------------
# Optimization rules
# ---------------------------------------------------------------------------

def optimize(cfg: dict, resolved: list, apply: bool, verbose: bool = True) -> list[str]:
    """
    Analyse resolved trades and produce incremental config changes.

    Returns list of human-readable change messages.
    Modifies cfg in-place if apply=True.
    """
    changes = []
    overall_wr, n_total = _win_rate(resolved)

    def log(msg):
        changes.append(msg)
        if verbose:
            print(msg)

    if overall_wr is None:
        log("  [optimizer] No resolved trades — skipping")
        return changes

    log(f"  [optimizer] win_rate={overall_wr:.1%}  n={n_total}  target={TARGET_WIN_RATE:.0%}")

    # ── Rule 1: Composite threshold ──────────────────────────────────────────
    current_thresh = cfg.get("composite_threshold", 0.70)
    best_thresh = score_threshold_sweep(resolved)

    if best_thresh is not None:
        # Clip and step-limit
        target_t = round(min(max(best_thresh, MIN_THRESH), MAX_THRESH), 2)
        # Step: move at most STEP_THRESH per run, allow small drops too
        if target_t > current_thresh:
            new_t = round(min(current_thresh + STEP_THRESH, target_t), 2)
        else:
            new_t = round(max(current_thresh - STEP_THRESH, target_t), 2)

        if new_t != current_thresh:
            log(f"  [threshold] {current_thresh:.2f} → {new_t:.2f}  "
                f"(sweep found {best_thresh:.3f} achieves {TARGET_WIN_RATE:.0%}+)")
            if apply:
                cfg["composite_threshold"] = new_t
    elif overall_wr < TARGET_WIN_RATE and n_total >= MIN_SAMPLE:
        # No single threshold found — nudge upward to be more selective
        new_t = round(min(current_thresh + STEP_THRESH, MAX_THRESH), 2)
        if new_t != current_thresh:
            log(f"  [threshold] nudge up {current_thresh:.2f} → {new_t:.2f}  "
                f"(overall {overall_wr:.1%} below target, no clear cutoff found)")
            if apply:
                cfg["composite_threshold"] = new_t

    # ── Rule 2: Slow-market composite threshold ──────────────────────────────
    slow_wr, slow_n = slow_market_win_rate(resolved)
    if slow_n >= MIN_SAMPLE and slow_wr is not None:
        cur_slow = cfg.get("slow_composite_threshold", 0.73)
        if slow_wr < TARGET_WIN_RATE - 0.05:
            new_slow = round(min(cur_slow + STEP_THRESH, MAX_THRESH), 2)
            if new_slow != cur_slow:
                log(f"  [slow_threshold] {cur_slow:.2f} → {new_slow:.2f}  "
                    f"(slow win_rate={slow_wr:.1%} n={slow_n})")
                if apply:
                    cfg["slow_composite_threshold"] = new_slow
        elif slow_wr >= TARGET_WIN_RATE and cur_slow > current_thresh + 0.01:
            # Slow-market strategy is working — relax slightly to capture more
            new_slow = round(max(cur_slow - STEP_THRESH, current_thresh + 0.01), 2)
            if new_slow != cur_slow:
                log(f"  [slow_threshold] relax {cur_slow:.2f} → {new_slow:.2f}  "
                    f"(slow win_rate={slow_wr:.1%} already at target)")
                if apply:
                    cfg["slow_composite_threshold"] = new_slow

    # ── Rule 3: Hour blocking / boosting ─────────────────────────────────────
    h_rates = hour_win_rates(resolved)
    blocked  = set(cfg.get("blocked_hours", [6, 8, 16]))
    boosted  = {int(k): v for k, v in cfg.get("boosted_hours", {}).items()}

    for h, (wr_h, n_h) in h_rates.items():
        if n_h < MIN_SAMPLE or wr_h is None:
            continue
        if wr_h < 0.50 and h not in blocked:
            if len(blocked) < MAX_BLOCK_HOURS:
                log(f"  [hours] BLOCK {h:02d}h  win_rate={wr_h:.1%}  n={n_h}")
                if apply:
                    blocked.add(h)
        elif wr_h >= 0.55 and h in blocked:
            # Recovered — unblock
            log(f"  [hours] UNBLOCK {h:02d}h  win_rate improved to {wr_h:.1%}  n={n_h}")
            if apply:
                blocked.discard(h)
        if wr_h >= 0.72 and n_h >= MIN_SAMPLE and h not in blocked:
            delta = round(-min(0.04, (wr_h - 0.65) * 0.25), 2)
            if boosted.get(h) != delta:
                log(f"  [hours] BOOST {h:02d}h  delta={delta:+.2f}  win_rate={wr_h:.1%}")
                if apply:
                    boosted[h] = delta
        elif wr_h < 0.65 and h in boosted:
            log(f"  [hours] REMOVE boost {h:02d}h  win_rate={wr_h:.1%} no longer earns boost")
            if apply:
                del boosted[h]

    if apply:
        cfg["blocked_hours"] = sorted(blocked)
        cfg["boosted_hours"]  = {str(k): v for k, v in sorted(boosted.items())}

    # ── Rule 4: Minimum momentum threshold ──────────────────────────────────
    best_mom, best_mom_wr = momentum_threshold_sweep(resolved)
    if best_mom is not None and n_total >= MIN_SAMPLE:
        cur_mom = cfg.get("min_momentum_pct", 0.08)
        if best_mom != cur_mom:
            # Step-limit
            if best_mom > cur_mom:
                new_mom = round(min(cur_mom + STEP_MOM, best_mom, MAX_MOM), 2)
            else:
                new_mom = round(max(cur_mom - STEP_MOM, best_mom, MIN_MOM), 2)
            if new_mom != cur_mom:
                log(f"  [momentum] min_momentum_pct {cur_mom:.2f} → {new_mom:.2f}  "
                    f"(best win_rate={best_mom_wr:.1%} at |m5|>={best_mom:.2f}%)")
                if apply:
                    cfg["min_momentum_pct"] = new_mom

    # ── Rule 5: Entry price gate ─────────────────────────────────────────────
    ep = entry_price_win_rates(resolved)
    low_wr, low_n   = ep["<=0.50"]
    high_wr, high_n = ep[">0.50"]
    cur_payout = cfg.get("min_payout_ratio", 0.90)
    if low_n >= MIN_SAMPLE and high_n >= MIN_SAMPLE:
        if high_wr is not None and high_wr < 0.55 and cur_payout < 1.0:
            new_payout = round(min(cur_payout + 0.05, 1.20), 2)
            log(f"  [payout] min_payout_ratio {cur_payout:.2f} → {new_payout:.2f}  "
                f"(high-entry trades win_rate={high_wr:.1%} n={high_n})")
            if apply:
                cfg["min_payout_ratio"] = new_payout
        elif low_wr is not None and low_wr >= TARGET_WIN_RATE and cur_payout > 0.85:
            new_payout = round(max(cur_payout - 0.05, 0.85), 2)
            log(f"  [payout] relax min_payout_ratio {cur_payout:.2f} → {new_payout:.2f}  "
                f"(low-entry trades win_rate={low_wr:.1%} n={low_n})")
            if apply:
                cfg["min_payout_ratio"] = new_payout

    # ── Rule 6: Signal weight nudges (needs >= 20 resolved) ─────────────────
    if n_total >= 20:
        deltas = signal_win_deltas(resolved)
        weights = dict(cfg.get("signal_weights", {}))
        # Map trade-log field name → config weight key
        field_to_weight = {
            "vol_adjusted_momentum": "vol_adjusted_momentum",
            "cex_poly_lag":          "cex_poly_lag",
            "vs_ref":                "btc_vs_reference",
            "momentum_5m":          "momentum_5m",
            "momentum_1m":          "momentum_1m",
            "price_acceleration":   "price_acceleration",
            "momentum_15m":         "momentum_15m",
        }
        weight_changes = {}
        for field, wkey in field_to_weight.items():
            d = deltas.get(field)
            if d is None or wkey not in weights:
                continue
            cur_w = weights[wkey]
            if d > 0.02:     # wins score higher — boost this signal
                nudge = min(MAX_WEIGHT_NUDGE, abs(d) * 0.05)
                weight_changes[wkey] = round(cur_w + nudge, 4)
            elif d < -0.02:  # losses score higher — reduce this signal
                nudge = min(MAX_WEIGHT_NUDGE, abs(d) * 0.05)
                weight_changes[wkey] = round(max(0.0, cur_w - nudge), 4)

        if weight_changes:
            # Merge and re-normalise to sum = 1.0
            new_weights = {**weights, **weight_changes}
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}
            diff_parts = [
                f"{k}: {weights[k]:.3f}→{v:.3f}"
                for k, v in weight_changes.items()
                if abs(v - weights.get(k, 0)) > 0.001
            ]
            if diff_parts:
                log(f"  [weights] nudge: {', '.join(diff_parts)}")
                if apply:
                    cfg["signal_weights"] = new_weights

    # ── Rule 7: Active-market composite threshold ────────────────────────────
    active = [t for t in resolved if abs(t.get("momentum_5m", 0) or 0) >= 0.25]
    active_wr, active_n = _win_rate(active)
    if active_n >= MIN_SAMPLE and active_wr is not None:
        cur_active = cfg.get("active_composite_threshold", 0.68)
        if active_wr < TARGET_WIN_RATE - 0.05:
            new_active = round(min(cur_active + STEP_THRESH, MAX_THRESH), 2)
            if new_active != cur_active:
                log(f"  [active_threshold] {cur_active:.2f} → {new_active:.2f}  "
                    f"(active win_rate={active_wr:.1%} n={active_n})")
                if apply:
                    cfg["active_composite_threshold"] = new_active
        elif active_wr >= TARGET_WIN_RATE and cur_active > MIN_THRESH:
            new_active = round(max(cur_active - STEP_THRESH, MIN_THRESH), 2)
            if new_active != cur_active:
                log(f"  [active_threshold] relax {cur_active:.2f} → {new_active:.2f}  "
                    f"(active win_rate={active_wr:.1%} already at target)")
                if apply:
                    cfg["active_composite_threshold"] = new_active

    # ── Rule 8: max_entry_yes / max_entry_no (poly price gate) ───────────────
    # If trades entered at higher poly prices lose more, tighten the bounds.
    if n_total >= MIN_SAMPLE:
        wins   = [t for t in resolved if t.get("outcome") == "win"]
        losses = [t for t in resolved if t.get("outcome") == "loss"]

        yes_trades = [t for t in resolved if t.get("side") == "yes"]
        no_trades  = [t for t in resolved if t.get("side") == "no"]

        # YES side: tighten max_entry_yes if high-price YES trades lose
        if len(yes_trades) >= MIN_SAMPLE:
            cur_max_yes = cfg.get("max_entry_yes", 0.60)
            high_yes = [t for t in yes_trades
                        if (t.get("poly_yes_price") or 0.5) > cur_max_yes - 0.05]
            low_yes  = [t for t in yes_trades
                        if (t.get("poly_yes_price") or 0.5) <= cur_max_yes - 0.05]
            high_wr_y, high_n_y = _win_rate(high_yes)
            low_wr_y,  low_n_y  = _win_rate(low_yes)
            if high_n_y >= MIN_SAMPLE and high_wr_y is not None and high_wr_y < 0.50:
                new_max_yes = round(max(cur_max_yes - 0.02, 0.52), 2)
                if new_max_yes != cur_max_yes:
                    log(f"  [poly_gate] max_entry_yes {cur_max_yes:.2f} → {new_max_yes:.2f}  "
                        f"(high-price YES win_rate={high_wr_y:.1%} n={high_n_y})")
                    if apply:
                        cfg["max_entry_yes"] = new_max_yes
            elif high_n_y >= MIN_SAMPLE and high_wr_y is not None and high_wr_y >= TARGET_WIN_RATE:
                new_max_yes = round(min(cur_max_yes + 0.02, 0.70), 2)
                if new_max_yes != cur_max_yes:
                    log(f"  [poly_gate] relax max_entry_yes {cur_max_yes:.2f} → {new_max_yes:.2f}  "
                        f"(high-price YES win_rate={high_wr_y:.1%})")
                    if apply:
                        cfg["max_entry_yes"] = new_max_yes

        # NO side: tighten min_entry_no if low-price NO trades lose
        if len(no_trades) >= MIN_SAMPLE:
            cur_min_no = cfg.get("min_entry_no", 0.28)
            low_no  = [t for t in no_trades
                       if (t.get("poly_yes_price") or 0.5) < cur_min_no + 0.05]
            high_no = [t for t in no_trades
                       if (t.get("poly_yes_price") or 0.5) >= cur_min_no + 0.05]
            low_wr_n, low_n_n = _win_rate(low_no)
            if low_n_n >= MIN_SAMPLE and low_wr_n is not None and low_wr_n < 0.50:
                new_min_no = round(min(cur_min_no + 0.02, 0.40), 2)
                if new_min_no != cur_min_no:
                    log(f"  [poly_gate] min_entry_no {cur_min_no:.2f} → {new_min_no:.2f}  "
                        f"(low-price NO win_rate={low_wr_n:.1%} n={low_n_n})")
                    if apply:
                        cfg["min_entry_no"] = new_min_no

    # ── Rule 9: max_no_score (cap on NO-side score) ──────────────────────────
    no_trades = [t for t in resolved if t.get("side") == "no"]
    no_wr, no_n = _win_rate(no_trades)
    if no_n >= MIN_SAMPLE and no_wr is not None:
        cur_max_no = cfg.get("max_no_score", 0.25)
        if no_wr < TARGET_WIN_RATE - 0.08:
            # NO trades losing badly — cap the score lower (require stronger signal)
            new_max_no = round(max(cur_max_no - 0.01, 0.20), 3)
            if new_max_no != cur_max_no:
                log(f"  [no_score] max_no_score {cur_max_no:.3f} → {new_max_no:.3f}  "
                    f"(NO win_rate={no_wr:.1%} n={no_n})")
                if apply:
                    cfg["max_no_score"] = new_max_no
        elif no_wr >= TARGET_WIN_RATE and cur_max_no < 0.30:
            new_max_no = round(min(cur_max_no + 0.01, 0.30), 3)
            if new_max_no != cur_max_no:
                log(f"  [no_score] relax max_no_score {cur_max_no:.3f} → {new_max_no:.3f}  "
                    f"(NO win_rate={no_wr:.1%})")
                if apply:
                    cfg["max_no_score"] = new_max_no

    # ── Rule 10: min_time_remaining ──────────────────────────────────────────
    # Trades entered with little time left may win less (market already priced in).
    if n_total >= MIN_SAMPLE:
        time_buckets = {
            "low":  [t for t in resolved if (t.get("time_remaining") or 300) < 220],
            "high": [t for t in resolved if (t.get("time_remaining") or 300) >= 220],
        }
        low_wr_t,  low_n_t  = _win_rate(time_buckets["low"])
        high_wr_t, high_n_t = _win_rate(time_buckets["high"])
        cur_min_time = cfg.get("min_time_remaining", 200)
        if low_n_t >= MIN_SAMPLE and low_wr_t is not None and low_wr_t < 0.50:
            new_min_time = min(cur_min_time + 10, 280)
            if new_min_time != cur_min_time:
                log(f"  [timing] min_time_remaining {cur_min_time} → {new_min_time}s  "
                    f"(low-time win_rate={low_wr_t:.1%} n={low_n_t})")
                if apply:
                    cfg["min_time_remaining"] = new_min_time
        elif high_n_t >= MIN_SAMPLE and high_wr_t is not None and high_wr_t >= TARGET_WIN_RATE \
                and cur_min_time > 160:
            new_min_time = max(cur_min_time - 10, 160)
            if new_min_time != cur_min_time:
                log(f"  [timing] relax min_time_remaining {cur_min_time} → {new_min_time}s  "
                    f"(high-time win_rate={high_wr_t:.1%})")
                if apply:
                    cfg["min_time_remaining"] = new_min_time

    # ── Rule 11: RSI thresholds ───────────────────────────────────────────────
    # If loss trades have high RSI on YES or low RSI on NO, tighten filters.
    if n_total >= MIN_SAMPLE:
        yes_losses = [t for t in resolved
                      if t.get("side") == "yes" and t.get("outcome") == "loss"
                      and t.get("rsi_14") is not None]
        no_losses  = [t for t in resolved
                      if t.get("side") == "no" and t.get("outcome") == "loss"
                      and t.get("rsi_14") is not None]
        cur_ob = cfg.get("rsi_overbought", 85)
        cur_os = cfg.get("rsi_oversold", 15)
        if len(yes_losses) >= MIN_SAMPLE:
            avg_rsi_yes_loss = sum(t["rsi_14"] for t in yes_losses) / len(yes_losses)
            if avg_rsi_yes_loss > 70 and cur_ob > 72:
                new_ob = max(cur_ob - 2, 72)
                log(f"  [rsi] rsi_overbought {cur_ob} → {new_ob}  "
                    f"(avg RSI in YES losses={avg_rsi_yes_loss:.0f})")
                if apply:
                    cfg["rsi_overbought"] = new_ob
            elif avg_rsi_yes_loss < 60 and cur_ob < 88:
                new_ob = min(cur_ob + 2, 88)
                log(f"  [rsi] relax rsi_overbought {cur_ob} → {new_ob}  "
                    f"(avg RSI in YES losses={avg_rsi_yes_loss:.0f}, not extreme)")
                if apply:
                    cfg["rsi_overbought"] = new_ob
        if len(no_losses) >= MIN_SAMPLE:
            avg_rsi_no_loss = sum(t["rsi_14"] for t in no_losses) / len(no_losses)
            if avg_rsi_no_loss < 30 and cur_os < 28:
                new_os = min(cur_os + 2, 28)
                log(f"  [rsi] rsi_oversold {cur_os} → {new_os}  "
                    f"(avg RSI in NO losses={avg_rsi_no_loss:.0f})")
                if apply:
                    cfg["rsi_oversold"] = new_os

    # ── Rule 12: slow_min_payout_ratio ───────────────────────────────────────
    slow_trades = [t for t in resolved if abs(t.get("momentum_5m", 0) or 0) < 0.12]
    slow_wr2, slow_n2 = _win_rate(slow_trades)
    if slow_n2 >= MIN_SAMPLE and slow_wr2 is not None:
        cur_slow_pay = cfg.get("slow_min_payout_ratio", 1.10)
        if slow_wr2 < TARGET_WIN_RATE - 0.05 and cur_slow_pay < 1.30:
            new_slow_pay = round(min(cur_slow_pay + 0.05, 1.30), 2)
            log(f"  [slow_payout] slow_min_payout_ratio {cur_slow_pay:.2f} → {new_slow_pay:.2f}  "
                f"(slow win_rate={slow_wr2:.1%} n={slow_n2})")
            if apply:
                cfg["slow_min_payout_ratio"] = new_slow_pay
        elif slow_wr2 >= TARGET_WIN_RATE and cur_slow_pay > 1.00:
            new_slow_pay = round(max(cur_slow_pay - 0.05, 1.00), 2)
            log(f"  [slow_payout] relax slow_min_payout_ratio {cur_slow_pay:.2f} → {new_slow_pay:.2f}  "
                f"(slow win_rate={slow_wr2:.1%})")
            if apply:
                cfg["slow_min_payout_ratio"] = new_slow_pay

    # ── Rule 13: slow_position_pct_cap ───────────────────────────────────────
    if slow_n2 >= MIN_SAMPLE and slow_wr2 is not None:
        cur_slow_cap = cfg.get("slow_position_pct_cap", 0.70)
        if slow_wr2 < TARGET_WIN_RATE - 0.10:
            new_slow_cap = round(max(cur_slow_cap - 0.05, 0.40), 2)
            if new_slow_cap != cur_slow_cap:
                log(f"  [slow_sizing] slow_position_pct_cap {cur_slow_cap:.2f} → {new_slow_cap:.2f}  "
                    f"(slow win_rate={slow_wr2:.1%}, reducing exposure)")
                if apply:
                    cfg["slow_position_pct_cap"] = new_slow_cap
        elif slow_wr2 >= TARGET_WIN_RATE and cur_slow_cap < 1.0:
            new_slow_cap = round(min(cur_slow_cap + 0.05, 1.0), 2)
            if new_slow_cap != cur_slow_cap:
                log(f"  [slow_sizing] slow_position_pct_cap {cur_slow_cap:.2f} → {new_slow_cap:.2f}  "
                    f"(slow win_rate={slow_wr2:.1%}, adding exposure)")
                if apply:
                    cfg["slow_position_pct_cap"] = new_slow_cap

    # ── Rule 14: max_entry_no (poly price gate for NO side) ──────────────────
    no_trades_ep = [t for t in resolved if t.get("side") == "no"]
    if len(no_trades_ep) >= MIN_SAMPLE:
        cur_max_no_ep = cfg.get("max_entry_no", 0.65)
        # For NO trades, poly_yes_price is the opponent's price.
        # If trades at high poly_yes_price (market is bullish) are losing NO bets → tighten
        high_no = [t for t in no_trades_ep
                   if (t.get("poly_yes_price") or 0.5) > cur_max_no_ep - 0.05]
        wr_high_no, n_high_no = _win_rate(high_no)
        if n_high_no >= MIN_SAMPLE and wr_high_no is not None and wr_high_no < 0.50:
            new_max_no_ep = round(max(cur_max_no_ep - 0.02, 0.52), 2)
            if new_max_no_ep != cur_max_no_ep:
                log(f"  [poly_gate] max_entry_no {cur_max_no_ep:.2f} → {new_max_no_ep:.2f}  "
                    f"(high poly_yes NO trades win_rate={wr_high_no:.1%} n={n_high_no})")
                if apply:
                    cfg["max_entry_no"] = new_max_no_ep
        elif n_high_no >= MIN_SAMPLE and wr_high_no is not None and wr_high_no >= TARGET_WIN_RATE:
            new_max_no_ep = round(min(cur_max_no_ep + 0.02, 0.70), 2)
            if new_max_no_ep != cur_max_no_ep:
                log(f"  [poly_gate] relax max_entry_no {cur_max_no_ep:.2f} → {new_max_no_ep:.2f}  "
                    f"(high poly_yes NO win_rate={wr_high_no:.1%})")
                if apply:
                    cfg["max_entry_no"] = new_max_no_ep

    # ── Rule 15: max_no_size (hard cap on NO trade size) ─────────────────────
    no_wr2, no_n2 = _win_rate([t for t in resolved if t.get("side") == "no"])
    if no_n2 >= MIN_SAMPLE and no_wr2 is not None:
        cur_max_no_sz = cfg.get("max_no_size", 2.50)
        if no_wr2 < TARGET_WIN_RATE - 0.10:
            # NO trades losing badly — shrink the max size to limit damage
            new_max_no_sz = round(max(cur_max_no_sz - 0.25, 0.50), 2)
            if new_max_no_sz != cur_max_no_sz:
                log(f"  [no_size] max_no_size ${cur_max_no_sz:.2f} → ${new_max_no_sz:.2f}  "
                    f"(NO win_rate={no_wr2:.1%} n={no_n2})")
                if apply:
                    cfg["max_no_size"] = new_max_no_sz
        elif no_wr2 >= TARGET_WIN_RATE and cur_max_no_sz < cfg.get("max_position", 3.50):
            new_max_no_sz = round(min(cur_max_no_sz + 0.25, cfg.get("max_position", 3.50)), 2)
            if new_max_no_sz != cur_max_no_sz:
                log(f"  [no_size] relax max_no_size ${cur_max_no_sz:.2f} → ${new_max_no_sz:.2f}  "
                    f"(NO win_rate={no_wr2:.1%})")
                if apply:
                    cfg["max_no_size"] = new_max_no_sz

    # ── Rule 16: min_yes_conf (minimum confidence for YES trades) ────────────
    yes_trades = [t for t in resolved if t.get("side") == "yes"]
    yes_wr, yes_n = _win_rate(yes_trades)
    if yes_n >= MIN_SAMPLE and yes_wr is not None:
        cur_min_conf = cfg.get("min_yes_conf", 0.0)
        if yes_wr < TARGET_WIN_RATE - 0.05:
            # Sweep confidence levels to find floor that hits target
            best_conf_thresh = None
            for conf_t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
                subset = [t for t in yes_trades
                          if (t.get("confidence") or 0) >= conf_t]
                wr_c, n_c = _win_rate(subset)
                if n_c >= MIN_SAMPLE and wr_c is not None and wr_c >= TARGET_WIN_RATE:
                    best_conf_thresh = conf_t
                    break
            if best_conf_thresh is not None:
                # Step toward it
                if best_conf_thresh > cur_min_conf:
                    new_conf = round(min(cur_min_conf + 0.05, best_conf_thresh), 2)
                else:
                    new_conf = round(max(cur_min_conf - 0.05, best_conf_thresh), 2)
                if new_conf != cur_min_conf:
                    log(f"  [yes_conf] min_yes_conf {cur_min_conf:.2f} → {new_conf:.2f}  "
                        f"(YES win_rate={yes_wr:.1%} n={yes_n})")
                    if apply:
                        cfg["min_yes_conf"] = new_conf
        elif yes_wr >= TARGET_WIN_RATE and cur_min_conf > 0:
            # Already at target — relax confidence gate to allow more trades
            new_conf = round(max(cur_min_conf - 0.05, 0.0), 2)
            if new_conf != cur_min_conf:
                log(f"  [yes_conf] relax min_yes_conf {cur_min_conf:.2f} → {new_conf:.2f}  "
                    f"(YES win_rate={yes_wr:.1%} already at target)")
                if apply:
                    cfg["min_yes_conf"] = new_conf

    # ── Rule 17: min_entry_price ──────────────────────────────────────────────
    if n_total >= MIN_SAMPLE:
        cur_min_ep = cfg.get("min_entry_price", 0.35)
        low_ep = [t for t in resolved if (t.get("entry_price") or 0.5) < cur_min_ep + 0.05]
        wr_low_ep, n_low_ep = _win_rate(low_ep)
        if n_low_ep >= MIN_SAMPLE and wr_low_ep is not None and wr_low_ep < 0.45:
            new_min_ep = round(min(cur_min_ep + 0.02, 0.42), 2)
            if new_min_ep != cur_min_ep:
                log(f"  [entry_price] min_entry_price {cur_min_ep:.2f} → {new_min_ep:.2f}  "
                    f"(low-entry trades win_rate={wr_low_ep:.1%} n={n_low_ep})")
                if apply:
                    cfg["min_entry_price"] = new_min_ep
        elif n_low_ep >= MIN_SAMPLE and wr_low_ep is not None and wr_low_ep >= TARGET_WIN_RATE \
                and cur_min_ep > 0.30:
            new_min_ep = round(max(cur_min_ep - 0.02, 0.30), 2)
            if new_min_ep != cur_min_ep:
                log(f"  [entry_price] relax min_entry_price {cur_min_ep:.2f} → {new_min_ep:.2f}  "
                    f"(low-entry win_rate={wr_low_ep:.1%})")
                if apply:
                    cfg["min_entry_price"] = new_min_ep

    # ── Rule 18: min_entry_yes / slow_max_entry_yes / slow_min_entry_no ──────
    # YES trades that entered too cheap (market hadn't moved) might signal under-conviction.
    yes_trades_e = [t for t in resolved if t.get("side") == "yes"]
    if len(yes_trades_e) >= MIN_SAMPLE:
        cur_min_yes = cfg.get("min_entry_yes", 0.35)
        too_cheap = [t for t in yes_trades_e
                     if (t.get("poly_yes_price") or 0.5) < cur_min_yes + 0.04]
        wr_cheap, n_cheap = _win_rate(too_cheap)
        if n_cheap >= MIN_SAMPLE and wr_cheap is not None and wr_cheap < 0.45:
            new_min_yes = round(min(cur_min_yes + 0.02, 0.44), 2)
            if new_min_yes != cur_min_yes:
                log(f"  [poly_gate] min_entry_yes {cur_min_yes:.2f} → {new_min_yes:.2f}  "
                    f"(cheap YES win_rate={wr_cheap:.1%} n={n_cheap})")
                if apply:
                    cfg["min_entry_yes"] = new_min_yes

    # Slow-market specific entry gates (tuned tighter than normal)
    slow_yes = [t for t in resolved
                if t.get("side") == "yes" and abs(t.get("momentum_5m", 0) or 0) < 0.12]
    if len(slow_yes) >= MIN_SAMPLE:
        cur_slow_max_yes = cfg.get("slow_max_entry_yes", 0.54)
        high_slow_yes = [t for t in slow_yes
                         if (t.get("poly_yes_price") or 0.5) > cur_slow_max_yes - 0.03]
        wr_hsly, n_hsly = _win_rate(high_slow_yes)
        if n_hsly >= MIN_SAMPLE and wr_hsly is not None and wr_hsly < 0.45:
            new_val = round(max(cur_slow_max_yes - 0.02, 0.48), 2)
            if new_val != cur_slow_max_yes:
                log(f"  [slow_gate] slow_max_entry_yes {cur_slow_max_yes:.2f} → {new_val:.2f}  "
                    f"(slow YES win_rate={wr_hsly:.1%} n={n_hsly})")
                if apply:
                    cfg["slow_max_entry_yes"] = new_val
        elif n_hsly >= MIN_SAMPLE and wr_hsly is not None and wr_hsly >= TARGET_WIN_RATE \
                and cur_slow_max_yes < 0.58:
            new_val = round(min(cur_slow_max_yes + 0.02, 0.58), 2)
            if new_val != cur_slow_max_yes:
                log(f"  [slow_gate] relax slow_max_entry_yes {cur_slow_max_yes:.2f} → {new_val:.2f}  "
                    f"(slow YES win_rate={wr_hsly:.1%})")
                if apply:
                    cfg["slow_max_entry_yes"] = new_val

    # ── Rule 19: max_time_remaining ───────────────────────────────────────────
    # Trades entered very early (lots of time left) may be premature.
    if n_total >= MIN_SAMPLE:
        cur_max_time = cfg.get("max_time_remaining", 350)
        early = [t for t in resolved if (t.get("time_remaining") or 0) > cur_max_time - 30]
        wr_early, n_early = _win_rate(early)
        if n_early >= MIN_SAMPLE and wr_early is not None and wr_early < 0.45:
            new_max_time = max(cur_max_time - 15, 260)
            if new_max_time != cur_max_time:
                log(f"  [timing] max_time_remaining {cur_max_time} → {new_max_time}s  "
                    f"(early-entry win_rate={wr_early:.1%} n={n_early})")
                if apply:
                    cfg["max_time_remaining"] = new_max_time
        elif n_early >= MIN_SAMPLE and wr_early is not None and wr_early >= TARGET_WIN_RATE \
                and cur_max_time < 420:
            new_max_time = min(cur_max_time + 15, 420)
            if new_max_time != cur_max_time:
                log(f"  [timing] relax max_time_remaining {cur_max_time} → {new_max_time}s  "
                    f"(early-entry win_rate={wr_early:.1%})")
                if apply:
                    cfg["max_time_remaining"] = new_max_time

    # ── Rule 20a: min_lag_override ────────────────────────────────────────────
    # min_lag_override: the minimum CEX/poly lag needed to override the poly
    # directional gate (i.e. allow a trade even when poly disagrees with CEX).
    # If trades that passed through the lag override are losing, raise the bar.
    if n_total >= MIN_SAMPLE:
        cur_lag = cfg.get("min_lag_override", 0.12)
        # We don't store lag in the trade log directly, so proxy: trades where
        # cex_poly_lag > cur_lag and outcome is win/loss
        lag_trades = [t for t in resolved if abs(t.get("cex_poly_lag") or 0) >= cur_lag]
        wr_lag, n_lag = _win_rate(lag_trades)
        if n_lag >= MIN_SAMPLE and wr_lag is not None and wr_lag < TARGET_WIN_RATE - 0.08:
            new_lag = round(min(cur_lag + 0.02, 0.25), 2)
            if new_lag != cur_lag:
                log(f"  [lag] min_lag_override {cur_lag:.2f} → {new_lag:.2f}  "
                    f"(lag-override trades win_rate={wr_lag:.1%} n={n_lag})")
                if apply:
                    cfg["min_lag_override"] = new_lag
        elif n_lag >= MIN_SAMPLE and wr_lag is not None and wr_lag >= TARGET_WIN_RATE \
                and cur_lag > 0.08:
            new_lag = round(max(cur_lag - 0.02, 0.08), 2)
            if new_lag != cur_lag:
                log(f"  [lag] relax min_lag_override {cur_lag:.2f} → {new_lag:.2f}  "
                    f"(lag trades win_rate={wr_lag:.1%})")
                if apply:
                    cfg["min_lag_override"] = new_lag

    # ── Rule 20: max_position_per_window ─────────────────────────────────────
    # If we're deploying too much into a single window and losing, reduce the cap.
    if n_total >= MIN_SAMPLE:
        cur_max_win = cfg.get("max_position_per_window", 4.0)
        max_pos = cfg.get("max_position", 3.5)
        if overall_wr < TARGET_WIN_RATE - 0.10 and cur_max_win > max_pos:
            new_max_win = round(max(cur_max_win - 0.25, max_pos), 2)
            if new_max_win != cur_max_win:
                log(f"  [window_cap] max_position_per_window ${cur_max_win:.2f} → ${new_max_win:.2f}  "
                    f"(overall win_rate={overall_wr:.1%}, reducing per-window exposure)")
                if apply:
                    cfg["max_position_per_window"] = new_max_win
        elif overall_wr >= TARGET_WIN_RATE and cur_max_win < max_pos * 1.5:
            new_max_win = round(min(cur_max_win + 0.25, max_pos * 1.5), 2)
            if new_max_win != cur_max_win:
                log(f"  [window_cap] relax max_position_per_window ${cur_max_win:.2f} → ${new_max_win:.2f}  "
                    f"(overall win_rate={overall_wr:.1%})")
                if apply:
                    cfg["max_position_per_window"] = new_max_win

    return changes


# ---------------------------------------------------------------------------
# Report (human-readable summary, no changes)
# ---------------------------------------------------------------------------

def print_report(resolved):
    overall_wr, n = _win_rate(resolved)
    if overall_wr is None:
        print("  No resolved trades to report.")
        return

    w = sum(1 for t in resolved if t.get("outcome") == "win")
    l = sum(1 for t in resolved if t.get("outcome") == "loss")
    avg_win  = sum(t["pnl"] for t in resolved if t.get("outcome") == "win" and t.get("pnl")) / max(w, 1)
    avg_loss = sum(t["pnl"] for t in resolved if t.get("outcome") == "loss" and t.get("pnl")) / max(l, 1)
    total_pnl = sum(t.get("pnl") or 0 for t in resolved)

    print(f"\n{'━'*58}")
    print(f"  OPTIMIZER REPORT  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'━'*58}")
    print(f"  Resolved trades : {n}  (wins={w}  losses={l})")
    print(f"  Win rate        : {overall_wr:.1%}  (target={TARGET_WIN_RATE:.0%})")
    print(f"  Total PnL       : ${total_pnl:+.2f}")
    print(f"  Avg win PnL     : ${avg_win:+.3f}")
    print(f"  Avg loss PnL    : ${avg_loss:+.3f}")
    if w > 0 and l > 0:
        print(f"  Win/loss ratio  : {abs(avg_win/avg_loss):.2f}x")

    print(f"\n  Win rate by hour (UTC):")
    h_rates = hour_win_rates(resolved)
    for h in sorted(h_rates):
        wr_h, n_h = h_rates[h]
        if n_h < 2:
            continue
        bar = "█" * int((wr_h or 0) * 20)
        tag = " ← BLOCK" if (wr_h or 1) < 0.50 else (" ← BOOST" if (wr_h or 0) >= 0.72 else "")
        print(f"    {h:02d}h  {bar:<20} {wr_h*100:5.1f}%  n={n_h}{tag}")

    print(f"\n  Win rate by score bucket:")
    buckets = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.00)]
    for lo, hi in buckets:
        sub = [t for t in resolved if lo <= (t.get("score") or 0) < hi]
        wr_s, n_s = _win_rate(sub)
        if n_s > 0:
            print(f"    score {lo:.2f}-{hi:.2f}  win_rate={wr_s*100:.1f}%  n={n_s}")

    print(f"\n  Signal win-delta (positive = higher in wins):")
    deltas = signal_win_deltas(resolved)
    for field, d in sorted(deltas.items(), key=lambda x: -abs(x[1])):
        bar = ("+" if d > 0 else "-") * min(20, int(abs(d) * 200))
        print(f"    {field:<28} {d:+.4f}  {bar}")

    print(f"\n  Win rate by momentum bucket:")
    for thresh, label in [(0.0, "all"), (0.08, ">=0.08%"), (0.12, ">=0.12%"),
                          (0.15, ">=0.15%"), (0.20, ">=0.20%")]:
        sub = [t for t in resolved if abs(t.get("momentum_5m", 0) or 0) >= thresh]
        wr_m, n_m = _win_rate(sub)
        if n_m > 0:
            print(f"    {label:<12}  win_rate={wr_m*100:.1f}%  n={n_m}")

    slow_wr, slow_n = slow_market_win_rate(resolved)
    if slow_n > 0:
        print(f"\n  Slow-regime trades: win_rate={slow_wr*100:.1f}%  n={slow_n}")

    print(f"{'━'*58}\n")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_optimizer(apply: bool = False, verbose: bool = True):
    """Called from scheduler or directly."""
    resolved = load_resolved_trades()
    cfg = _load_config()

    if verbose:
        print_report(resolved)

    overall_wr, n_total = _win_rate(resolved)
    changes = optimize(cfg, resolved, apply=apply, verbose=verbose)

    if apply and changes:
        _save_config(cfg)
        _append_log(changes)
        if verbose:
            print(f"\n  Config updated ({len(changes)} changes logged to optimizer.log)")
        # Filter to only the actual change lines (skip start/end lines)
        changes_applied = [c for c in changes if not c.strip().startswith("[optimizer]")]
        _send_telegram_alert(cfg, changes_applied, overall_wr)
        _write_history(overall_wr, n_total, changes_applied)
    elif not apply and verbose:
        print(f"\n  [dry-run] {len(changes)} potential changes — pass --apply to commit")

    return cfg


def main():
    global TARGET_WIN_RATE  # must be declared before any use in this scope
    parser = argparse.ArgumentParser(description="Pknwitq Signal Optimizer")
    parser.add_argument("--apply",   action="store_true", help="Write changes to config.json")
    parser.add_argument("--report",  action="store_true", help="Print report only, no changes")
    parser.add_argument("--watch",   type=int, metavar="MINUTES",
                        help="Run continuously every N minutes (implies --apply)")
    parser.add_argument("--target",  type=float, default=TARGET_WIN_RATE,
                        help=f"Win rate target (default: {TARGET_WIN_RATE:.0%})")
    args = parser.parse_args()

    TARGET_WIN_RATE = args.target

    if args.report:
        resolved = load_resolved_trades()
        print_report(resolved)
        return

    if args.watch:
        interval = args.watch * 60
        print(f"[watch] Running every {args.watch}min — apply=True")
        while True:
            try:
                run_optimizer(apply=True, verbose=True)
            except Exception as e:
                print(f"[watch] ERROR: {e}")
            print(f"[watch] sleeping {args.watch}min...")
            time.sleep(interval)
    else:
        run_optimizer(apply=args.apply, verbose=True)


if __name__ == "__main__":
    main()
