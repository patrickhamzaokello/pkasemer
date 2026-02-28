#!/usr/bin/env python3
"""
FastLoop Monitor API

Serves signal data and status to the monitoring dashboard.
Runs on port 5000 inside the container (mapped to host port).

Added endpoints:
  /api/logs          — tail scheduler.log for live activity feed
  /api/cycle-feed    — last N cycles with signal + decision summary
  /api/import-status — cache file state + daily quota usage
"""

import os
import json
import sqlite3
import math
import re
from datetime import datetime, timezone
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

app = Flask(__name__, static_folder="/app/dashboard")
CORS(app)

DB_PATH     = os.environ.get("DB_PATH", "/data/signal_research.db")
STATUS_PATH = "/data/status.json"
LOG_PATH    = "/app/logs/collector.log"   # unified stdout+stderr via tee (scheduler + trader)
CACHE_PATH  = "/data/market_id_cache.json"
SPEND_PATH  = "/data/daily_spend.json"    # on shared volume, written by fast_trader.py
CONFIG_PATH = "/app/config.json"          # trader/config.json, volume-mounted read-only
TRADE_PATH  = ("/data/trade_log.json")

def _load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

CONFIG = _load_config()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Existing endpoints (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    try:
        with open(STATUS_PATH) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "No status yet — scheduler may not be running"})


@app.route("/api/observations")
def observations():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT id, ts, market_slug, seconds_remaining, outcome, resolved,
                   momentum_1m, momentum_5m, momentum_15m, rsi_14, volume_ratio,
                   order_imbalance, trade_flow_ratio, volatility_5m,
                   price_acceleration, poly_yes_price, poly_divergence,
                   cex_poly_lag, momentum_consistency, vol_adjusted_momentum,
                   price_now, price_to_beat, btc_vs_reference,
                   signal_score, signal_side, signal_confidence, would_trade, filter_reason
            FROM signal_observations
            ORDER BY id DESC LIMIT 200
        """).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/correlations")
def correlations():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT * FROM signal_observations
            WHERE resolved = 1 AND outcome IN ('up', 'down')
        """).fetchall()
        conn.close()

        if not rows:
            return jsonify({"error": "No resolved observations yet", "data": []})

        SIGNAL_COLS = [
            "btc_vs_reference",
            "momentum_1m", "momentum_5m", "momentum_15m",
            "rsi_14", "volume_ratio", "order_imbalance",
            "trade_flow_ratio", "volatility_5m", "price_acceleration",
            "poly_yes_price", "poly_divergence", "cex_poly_lag",
            "momentum_consistency", "vol_adjusted_momentum",
        ]

        results = []
        for col in SIGNAL_COLS:
            paired = [(r[col], 1 if r["outcome"] == "up" else 0)
                      for r in rows if r[col] is not None]
            if len(paired) < 5:
                continue

            v_list = [p[0] for p in paired]
            o_list = [p[1] for p in paired]
            n = len(v_list)
            mean_v = sum(v_list) / n
            mean_o = sum(o_list) / n
            std_v = math.sqrt(sum((x - mean_v)**2 for x in v_list) / n) or 1e-9
            std_o = math.sqrt(sum((x - mean_o)**2 for x in o_list) / n) or 1e-9
            corr = sum((v_list[i] - mean_v) * (o_list[i] - mean_o)
                       for i in range(n)) / (n * std_v * std_o)

            pos_wins = [o for v, o in paired if v > 0]
            neg_wins = [o for v, o in paired if v <= 0]
            wr_pos = sum(pos_wins) / len(pos_wins) if pos_wins else None
            wr_neg = sum(neg_wins) / len(neg_wins) if neg_wins else None
            edge = (wr_pos - wr_neg) if (wr_pos is not None and wr_neg is not None) else None

            results.append({
                "signal": col,
                "n": n,
                "corr": round(corr, 4),
                "wr_pos": round(wr_pos, 4) if wr_pos is not None else None,
                "wr_neg": round(wr_neg, 4) if wr_neg is not None else None,
                "edge": round(edge, 4) if edge is not None else None,
                "abs_corr": abs(corr),
            })

        results.sort(key=lambda x: x["abs_corr"], reverse=True)
        return jsonify({"total_resolved": len(rows), "data": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/summary")
def summary():
    try:
        conn = get_db()
        total    = conn.execute("SELECT COUNT(*) FROM signal_observations").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM signal_observations WHERE resolved=1").fetchone()[0]
        up       = conn.execute("SELECT COUNT(*) FROM signal_observations WHERE outcome='up'").fetchone()[0]
        down     = conn.execute("SELECT COUNT(*) FROM signal_observations WHERE outcome='down'").fetchone()[0]
        unclear  = conn.execute("SELECT COUNT(*) FROM signal_observations WHERE outcome='unclear'").fetchone()[0]
        latest   = conn.execute("SELECT ts FROM signal_observations ORDER BY id DESC LIMIT 1").fetchone()
        oldest   = conn.execute("SELECT ts FROM signal_observations ORDER BY id ASC LIMIT 1").fetchone()

        recent_avg = conn.execute("""
            SELECT AVG(momentum_5m), AVG(order_imbalance), AVG(trade_flow_ratio),
                   AVG(rsi_14), AVG(poly_yes_price), AVG(btc_vs_reference), AVG(price_now)
            FROM (SELECT * FROM signal_observations ORDER BY id DESC LIMIT 50)
        """).fetchone()
        conn.close()

        return jsonify({
            "total": total, "resolved": resolved, "unresolved": total - resolved,
            "outcomes": {"up": up, "down": down, "unclear": unclear},
            "date_range": {"oldest": oldest[0] if oldest else None, "latest": latest[0] if latest else None},
            "recent_avg": {
                "momentum_5m": round(recent_avg[0], 4) if recent_avg[0] else None,
                "order_imbalance": round(recent_avg[1], 4) if recent_avg[1] else None,
                "trade_flow_ratio": round(recent_avg[2], 4) if recent_avg[2] else None,
                "rsi_14": round(recent_avg[3], 2) if recent_avg[3] else None,
                "poly_yes_price": round(recent_avg[4], 4) if recent_avg[4] else None,
                "btc_vs_reference": round(recent_avg[5], 4) if recent_avg[5] else None,
                "price_now": round(recent_avg[6], 2) if recent_avg[6] else None,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/current")
def current():
    try:
        conn = get_db()
        row = conn.execute("""
            SELECT ts, market_slug, seconds_remaining, outcome,
                   momentum_5m, momentum_1m, rsi_14, order_imbalance,
                   trade_flow_ratio, poly_yes_price, poly_divergence,
                   btc_vs_reference, price_now, price_to_beat,
                   vol_adjusted_momentum, cex_poly_lag, volatility_5m,
                   momentum_consistency, volume_ratio,
                   signal_score, signal_side, signal_confidence, would_trade, filter_reason
            FROM signal_observations ORDER BY id DESC LIMIT 1
        """).fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "No data yet"})
        return jsonify(dict(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sparkline")
def sparkline():
    try:
        conn = get_db()
        latest = conn.execute("SELECT market_slug FROM signal_observations ORDER BY id DESC LIMIT 1").fetchone()
        if not latest or not latest[0]:
            return jsonify([])
        slug = latest[0]
        rows = conn.execute("""
            SELECT ts, btc_vs_reference, poly_yes_price, momentum_5m,
                   signal_score, would_trade, signal_side
            FROM signal_observations WHERE market_slug = ? ORDER BY id ASC
        """, (slug,)).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/window-history")
def window_history():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT market_slug, MIN(ts) as start_ts, COUNT(*) as obs_count,
                   MAX(outcome) as outcome,
                   ROUND(AVG(btc_vs_reference), 4) as avg_vs_ref,
                   ROUND(MAX(btc_vs_reference), 4) as max_vs_ref,
                   ROUND(MIN(btc_vs_reference), 4) as min_vs_ref,
                   ROUND(AVG(momentum_5m), 4) as avg_m5,
                   ROUND(MAX(poly_yes_price), 4) as final_poly,
                   ROUND(AVG(rsi_14), 1) as avg_rsi,
                   MAX(price_to_beat) as price_to_beat,
                   ROUND(AVG(signal_score), 4) as avg_score,
                   SUM(would_trade) as would_trade_count
            FROM signal_observations
            WHERE market_slug IS NOT NULL AND market_slug != ''
            GROUP BY market_slug ORDER BY MAX(ts) DESC LIMIT 20
        """).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/composite")
def composite_live():
    try:
        conn = get_db()
        row = conn.execute("""
            SELECT ts, market_slug, seconds_remaining,
                   momentum_5m, rsi_14, volume_ratio, order_imbalance,
                   trade_flow_ratio, volatility_5m, price_acceleration,
                   btc_vs_reference, cex_poly_lag, momentum_consistency,
                   vol_adjusted_momentum, poly_yes_price, poly_divergence,
                   price_now, price_to_beat
            FROM signal_observations ORDER BY id DESC LIMIT 1
        """).fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "No data yet"})
        d = dict(row)
        cex = {k: d.get(k) for k in [
            "momentum_5m", "rsi_14", "volume_ratio", "order_imbalance",
            "trade_flow_ratio", "volatility_5m", "price_acceleration",
            "btc_vs_reference", "cex_poly_lag", "momentum_consistency",
            "vol_adjusted_momentum",
        ]}
        poly = {"poly_yes_price": d.get("poly_yes_price"), "poly_divergence": d.get("poly_divergence"), "poly_spread": None}
        import sys as _sys
        _sys.path.insert(0, "/app")
        from composite_signal import get_composite_signal
        signal = get_composite_signal(cex, poly)
        return jsonify({
            "ts": d["ts"], "market_slug": d["market_slug"],
            "seconds_remaining": d["seconds_remaining"],
            "price_now": d.get("price_now"), "price_to_beat": d.get("price_to_beat"),
            "score": round(signal["score"], 4), "confidence": round(signal["confidence"], 4),
            "should_trade": signal["should_trade"], "side": signal["side"],
            "filter_reason": signal["filter_reason"],
            "breakdown": {k: {"raw": v.get("raw"), "normalized": round(v["normalized"], 4) if v.get("normalized") is not None else None, "weight": v.get("weight", 0)} for k, v in signal["breakdown"].items()},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/score-history")
def score_history():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT ts, market_slug, seconds_remaining,
                   signal_score, signal_side, signal_confidence,
                   would_trade, filter_reason,
                   outcome, resolved, btc_vs_reference, poly_yes_price,
                   momentum_5m, volume_ratio
            FROM signal_observations ORDER BY id DESC LIMIT 100
        """).fetchall()
        conn.close()
        result = []
        for row in rows:
            d = dict(row)
            score = d.get("signal_score")
            result.append({
                "ts": d["ts"], "market_slug": d.get("market_slug"),
                "seconds_remaining": d.get("seconds_remaining"),
                "score": round(score, 4) if score is not None else None,
                "should_trade": bool(d.get("would_trade")), "side": d.get("signal_side"),
                "filter_reason": d.get("filter_reason"), "outcome": d.get("outcome"),
                "resolved": d.get("resolved"), "btc_vs_reference": d.get("btc_vs_reference"),
                "poly_yes_price": d.get("poly_yes_price"), "momentum_5m": d.get("momentum_5m"),
                "volume_ratio": d.get("volume_ratio"),
            })
        return jsonify(result[::-1])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions")
def predictions():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT ts, market_slug, seconds_remaining,
                   signal_score, signal_side, signal_confidence, filter_reason,
                   outcome, resolved, btc_vs_reference, poly_yes_price, momentum_5m, volume_ratio
            FROM signal_observations WHERE would_trade = 1 ORDER BY id DESC LIMIT 200
        """).fetchall()
        conn.close()
        result = []
        for row in rows:
            d = dict(row)
            score, side, outcome = d.get("signal_score"), d.get("signal_side"), d.get("outcome")
            correct = None
            if side and outcome in ("up", "down"):
                correct = (side == "yes") == (outcome == "up")
            result.append({
                "ts": d["ts"], "market_slug": d.get("market_slug"),
                "seconds_remaining": d.get("seconds_remaining"),
                "score": round(score, 4) if score is not None else None,
                "side": side, "confidence": round(d["signal_confidence"], 4) if d.get("signal_confidence") is not None else None,
                "outcome": outcome, "resolved": d.get("resolved"), "correct": correct,
                "btc_vs_reference": d.get("btc_vs_reference"),
                "poly_yes_price": d.get("poly_yes_price"), "momentum_5m": d.get("momentum_5m"),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/accuracy")
def accuracy():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT signal_side, signal_score, outcome FROM signal_observations
            WHERE would_trade = 1 AND resolved = 1 AND outcome IN ('up','down')
        """).fetchall()
        conn.close()
        total = len(rows)
        if total == 0:
            return jsonify({"total": 0, "win_rate": None, "yes_wr": None, "no_wr": None})
        correct = sum(1 for r in rows if (r[0]=="yes" and r[2]=="up") or (r[0]=="no" and r[2]=="down"))
        yes_rows = [r for r in rows if r[0]=="yes"]
        no_rows  = [r for r in rows if r[0]=="no"]
        yes_correct = sum(1 for r in yes_rows if r[2]=="up")
        no_correct  = sum(1 for r in no_rows  if r[2]=="down")
        avg_correct = sum(r[1] for r in rows if r[1] and ((r[0]=="yes" and r[2]=="up") or (r[0]=="no" and r[2]=="down"))) / correct if correct else None
        avg_wrong   = sum(r[1] for r in rows if r[1] and not ((r[0]=="yes" and r[2]=="up") or (r[0]=="no" and r[2]=="down"))) / (total-correct) if total>correct else None
        return jsonify({
            "total": total, "correct": correct, "win_rate": round(correct/total, 4),
            "yes_total": len(yes_rows), "yes_correct": yes_correct,
            "yes_wr": round(yes_correct/len(yes_rows), 4) if yes_rows else None,
            "no_total": len(no_rows), "no_correct": no_correct,
            "no_wr": round(no_correct/len(no_rows), 4) if no_rows else None,
            "avg_score_correct": round(avg_correct, 4) if avg_correct else None,
            "avg_score_wrong": round(avg_wrong, 4) if avg_wrong else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades")
def trades():
    """
    Read executed trades from trade_log.json (written by fast_trader._log_trade_local).
    Returns a flat list, newest-first, with field names the dashboard expects.
    Mirrors the /api/logs pattern: open TRADE_PATH, parse, map, return.
    """
    n = int(request.args.get("n", 500))
    try:
        with open(TRADE_PATH) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result = []
    for t in reversed(raw):          # newest-first (mirrors ORDER BY id DESC in old DB query)
        result.append({
            "ts":                t.get("timestamp"),          # ISO — slice(0,10)=date, slice(11,19)=time
            "market_slug":       t.get("slug"),
            "trade_side":        t.get("side"),
            "signal_side":       t.get("side"),
            "trade_amount":      t.get("position_size"),
            "trade_result":      t.get("pnl"),                # None until market resolves
            "outcome":           t.get("outcome"),            # "up" | "down" | None
            "resolved":          1 if t.get("outcome") is not None else 0,
            "price_now":         None,                        # not captured in trade_log
            "btc_vs_reference":  t.get("vs_ref"),
            "momentum_5m":       t.get("momentum_pct"),
            "poly_yes_price":    t.get("poly_yes_price"),
            "seconds_remaining": t.get("time_remaining"),
            "signal_score":      t.get("score"),
            "signal_confidence": t.get("confidence"),
            "filter_reason":     None,                        # only for blocked signals, not trades
        })

    return jsonify(result[:n])


@app.route("/api/pnl-summary")
def pnl_summary():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT ts, trade_side, trade_amount, trade_result, outcome, resolved
            FROM signal_observations WHERE traded = 1
        """).fetchall()
        conn.close()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        total_pnl  = sum(r[3] for r in rows if r[3] is not None)
        today_rows = [r for r in rows if r[0] and r[0][:10] == today]
        today_pnl  = sum(r[3] for r in today_rows if r[3] is not None)
        resolved   = [r for r in rows if r[4] in ("up","down") and r[3] is not None]
        wins = sum(1 for r in resolved if r[3] > 0)
        return jsonify({
            "total_trades": len(rows), "total_pnl": round(total_pnl, 4),
            "today_trades": len(today_rows), "today_pnl": round(today_pnl, 4),
            "win_rate": round(wins/len(resolved), 4) if resolved else None,
            "resolved_trades": len(resolved),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Activity feed, logs, import status
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/logs")
def logs():
    """
    Tail the scheduler log and parse it into structured events.
    Returns last N lines as structured objects with type tags for colour-coding.
    """
    n = int(request.args.get("n", 120))
    try:
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return jsonify({"lines": [], "error": "Log file not found — scheduler may not be running"})

    tail = lines[-n:]
    parsed = []
    for raw in tail:
        line = raw.rstrip()
        if not line:
            continue

        # Classify line type for front-end colour coding
        lo = line.lower()
        if "error" in lo or "exception" in lo or "traceback" in lo:
            kind = "error"
        elif "rate limited" in lo or "429" in lo:
            kind = "ratelimit"
        elif ("→ yes" in lo or "→ no" in lo) and "score=" in lo:
            kind = "trade"   # [LIVE] would-trade line: "score=X → NO $Y"
        elif "traded" in lo or "would trade" in lo or "would_trade" in lo:
            kind = "trade"
        elif "skip import" in lo or "import quota" in lo or "cache warm" in lo:
            kind = "import"
        elif "block:" in lo or "skip:" in lo:
            kind = "block"
        elif "[cycle" in lo or "running" in lo:
            kind = "cycle"
        elif "resolved" in lo:
            kind = "resolve"
        else:
            kind = "info"

        # Extract timestamp if present
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", line)
        ts = ts_match.group(1) if ts_match else None

        parsed.append({"raw": line, "kind": kind, "ts": ts})

    return jsonify({"lines": parsed, "total_lines": len(lines)})


@app.route("/api/cycle-feed")
def cycle_feed():
    """
    Last 50 observations enriched with decision info — the core activity feed.
    Each row represents one scheduler cycle with signal + decision outcome.
    """
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT id, ts, market_slug, seconds_remaining,
                   momentum_5m, btc_vs_reference, volume_ratio,
                   poly_yes_price, price_now,
                   signal_score, signal_side, signal_confidence,
                   would_trade, filter_reason,
                   outcome, resolved
            FROM signal_observations
            ORDER BY id DESC LIMIT 50
        """).fetchall()
        conn.close()

        result = []
        for row in rows:
            d = dict(row)
            score  = d.get("signal_score")
            side   = d.get("signal_side")
            m5     = d.get("momentum_5m")
            vs_ref = d.get("btc_vs_reference")
            vol    = d.get("volume_ratio")

            # Determine display decision
            if d.get("would_trade"):
                decision = f"TRADE {(side or '').upper()}"
                decision_type = "trade"
            elif d.get("filter_reason"):
                fr = d["filter_reason"]
                if "momentum" in fr:
                    decision = "BLOCK: weak momentum"
                    decision_type = "block_momentum"
                elif "volume" in fr:
                    decision = "BLOCK: low volume"
                    decision_type = "block_volume"
                elif "neutral band" in fr:
                    decision = "BLOCK: neutral band"
                    decision_type = "block_neutral"
                elif "poly" in fr.lower():
                    decision = "BLOCK: poly priced"
                    decision_type = "block_poly"
                elif "rsi" in fr.lower():
                    decision = "BLOCK: RSI extreme"
                    decision_type = "block_rsi"
                else:
                    decision = f"BLOCK"
                    decision_type = "block"
            else:
                decision = "—"
                decision_type = "unknown"

            result.append({
                "id":           d["id"],
                "ts":           d["ts"],
                "slug_short":   (d.get("market_slug") or "")[-10:],
                "secs":         d.get("seconds_remaining"),
                "m5":           round(m5, 4) if m5 is not None else None,
                "vs_ref":       round(vs_ref, 4) if vs_ref is not None else None,
                "vol":          round(vol, 2) if vol is not None else None,
                "poly":         d.get("poly_yes_price"),
                "price_now":    d.get("price_now"),
                "score":        round(score, 3) if score is not None else None,
                "side":         side,
                "confidence":   round(d["signal_confidence"], 3) if d.get("signal_confidence") is not None else None,
                "decision":     decision,
                "decision_type":decision_type,
                "outcome":      d.get("outcome"),
                "resolved":     d.get("resolved"),
                "filter_reason":d.get("filter_reason"),
            })

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/import-status")
def import_status():
    """Cache file state + daily quota usage for the import health panel."""
    result = {
        "cache_exists":    False,
        "cache_entries":   0,
        "cached_slugs":    [],
        "current_cached":  False,
        "next_cached":     False,
        "imports_today":   0,
        "import_limit":    50,
        "spend_exists":    False,
        "spent_today":     0.0,
        "trades_today":    0,
        "daily_budget":    CONFIG.get("daily_budget", 20.0),
    }

    # Cache file
    import time
    bucket = (int(time.time()) // 300) * 300
    current_slug = f"btc-updown-5m-{bucket}"
    next_slug    = f"btc-updown-5m-{bucket + 300}"

    try:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        result["cache_exists"]   = True
        result["cache_entries"]  = len(cache)
        result["cached_slugs"]   = list(cache.keys())[-10:]  # last 10
        result["current_cached"] = current_slug in cache
        result["next_cached"]    = next_slug in cache
    except Exception:
        pass

    # Daily spend file
    try:
        with open(SPEND_PATH) as f:
            spend = json.load(f)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if spend.get("date") == today:
            result["spend_exists"]   = True
            result["imports_today"]  = spend.get("imports_today", 0)
            result["spent_today"]    = spend.get("spent", 0.0)
            result["trades_today"]   = spend.get("trades", 0)
    except Exception:
        pass

    return jsonify(result)


@app.route("/api/decision-stats")
def decision_stats():
    """Aggregate decision breakdown for the current session."""
    try:
        conn = get_db()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = conn.execute("""
            SELECT would_trade, filter_reason, signal_score, signal_side
            FROM signal_observations
            WHERE ts >= ?
        """, (today,)).fetchall()
        conn.close()

        total        = len(rows)
        would_trade  = sum(1 for r in rows if r[0])
        blocked      = total - would_trade

        reasons = {}
        for r in rows:
            if not r[0] and r[1]:
                fr = r[1]
                if "momentum" in fr:   key = "weak momentum"
                elif "volume" in fr:   key = "low volume"
                elif "neutral" in fr:  key = "neutral band"
                elif "poly" in fr.lower(): key = "poly priced"
                elif "rsi" in fr.lower():  key = "RSI extreme"
                elif "fee" in fr.lower():  key = "fee EV"
                else:                  key = "other"
                reasons[key] = reasons.get(key, 0) + 1

        scores = [r[2] for r in rows if r[2] is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        return jsonify({
            "total_today":    total,
            "would_trade":    would_trade,
            "blocked":        blocked,
            "trade_rate":     round(would_trade / total, 3) if total else 0,
            "avg_score":      round(avg_score, 3) if avg_score else None,
            "block_reasons":  reasons,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config")
def config_endpoint():
    """Expose trader/config.json to the dashboard (re-read on each call so edits take effect after restart)."""
    try:
        with open(CONFIG_PATH) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "config.json not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_dashboard(path):
    if path and os.path.exists(os.path.join("/app/dashboard", path)):
        return send_from_directory("/app/dashboard", path)
    return send_from_directory("/app/dashboard", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)