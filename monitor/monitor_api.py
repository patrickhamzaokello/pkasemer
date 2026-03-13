#!/usr/bin/env python3
"""
Pknwitq Monitor API

Serves signal data and status to the monitoring dashboard.
Runs on port 5000 inside the container (mapped to host port).

Added endpoints:
  /api/logs          — tail scheduler.log for live activity feed
  /api/cycle-feed    — last N cycles with signal + decision summary
  /api/import-status — cache file state + daily quota usage
  /api/poly-metrics  — real P&L, win rate, YES/NO breakdown from Polymarket
  /api/poly-trades   — full trade history from Polymarket Data API (paginated)
  /api/poly-positions— current open positions from Polymarket
"""

import os
import sys
import json
import sqlite3
import math
import re
import threading
import csv
import io
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request as _URLRequest
from urllib.error import HTTPError, URLError
from flask import Flask, jsonify, send_from_directory, request, session, redirect, Response, stream_with_context
from flask_cors import CORS
from werkzeug.security import check_password_hash
import time as _time

app = Flask(__name__, static_folder="/app/dashboard")
CORS(app)

# ── Authentication config ──────────────────────────────────────────────────
_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    sys.stderr.write("FATAL: SECRET_KEY environment variable is not set.\n")
    sys.stderr.write("Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"\n")
    sys.exit(1)
app.secret_key = _secret_key
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)

DB_PATH      = os.environ.get("DB_PATH", "/data/signal_research.db")
STATUS_PATH  = "/data/status.json"
LOG_PATH     = "/app/logs/collector.log"   # unified stdout+stderr via tee (scheduler + trader)
CACHE_PATH   = "/data/market_id_cache.json"
SPEND_PATH   = "/data/daily_spend.json"    # on shared volume, written by fast_trader.py
_DATA_CONFIG  = "/data/config.json"         # persistent volume — optimizer writes here
_IMAGE_CONFIG = "/app/config.json"          # image default — fallback if /data/ not seeded yet
CONFIG_PATH   = _DATA_CONFIG               # always prefer the live, optimizer-tuned copy
TRADE_PATH   = "/data/trade_log.json"
DEPLOY_PATH  = "/data/deploy_info.json"    # deploy counter + timestamp


def _bump_deploy_info():
    """Increment deploy count and record timestamp when monitor starts up."""
    try:
        try:
            with open(DEPLOY_PATH) as _f:
                info = json.load(_f)
        except Exception:
            info = {"count": 0, "last_deployed": None}
        info["count"] = info.get("count", 0) + 1
        info["last_deployed"] = datetime.now(timezone.utc).isoformat()
        with open(DEPLOY_PATH, "w") as _f:
            json.dump(info, _f)
    except Exception as e:
        sys.stderr.write(f"[deploy-tracker] failed to update {DEPLOY_PATH}: {e}\n")


_bump_deploy_info()

def _load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

CONFIG = _load_config()


USERS_DB          = os.environ.get("USERS_DB_PATH",    "/data/users.db")
KILL_SWITCH_FILE  = os.environ.get("KILL_SWITCH_PATH", "/data/kill_switch.json")

# {ip: {"count": int, "lockout_until": datetime | None}}
_login_attempts: dict = {}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_users_db():
    """Create users table if it doesn't exist yet."""
    data_dir = os.path.dirname(os.path.abspath(USERS_DB))
    os.makedirs(data_dir, exist_ok=True)
    conn = sqlite3.connect(USERS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            email         TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TEXT DEFAULT (datetime('now')),
            last_login    TEXT
        )
    """)
    conn.commit()
    conn.close()


_init_users_db()

# ─────────────────────────────────────────────────────────────────────────────
# Auth middleware + routes
# ─────────────────────────────────────────────────────────────────────────────

_EXEMPT = {"/login", "/auth/login", "/auth/logout", "/health"}


@app.before_request
def require_login():
    if request.path in _EXEMPT:
        return None
    if session.get("logged_in"):
        return None
    # Not authenticated
    if request.path.startswith("/api/"):
        return jsonify({"error": "Unauthorized", "redirect": "/login"}), 401
    return redirect("/login")


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/api/deploy-info")
def deploy_info():
    try:
        with open(DEPLOY_PATH) as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({"count": 0, "last_deployed": None})


@app.route("/login")
def login_page():
    if session.get("logged_in"):
        return redirect("/")
    return send_from_directory("/app/dashboard", "login.html")


@app.route("/auth/login", methods=["POST"])
def auth_login():
    ip = request.remote_addr
    now = datetime.now(timezone.utc)

    # Rate limit check
    attempt = _login_attempts.get(ip, {"count": 0, "lockout_until": None})
    lockout_until = attempt.get("lockout_until")
    if lockout_until and now < lockout_until:
        remaining = int((lockout_until - now).total_seconds())
        return jsonify({"error": f"Too many attempts. Try again in {remaining} seconds.",
                        "retry_after": remaining}), 429

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    # Look up user
    user = None
    try:
        conn = sqlite3.connect(USERS_DB)
        conn.row_factory = sqlite3.Row
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
    except Exception:
        pass

    if not user or not check_password_hash(user["password_hash"], password):
        count = attempt.get("count", 0) + 1
        new_lockout = None
        if count >= 5:
            new_lockout = now + timedelta(minutes=15)
            count = 0  # reset counter after lockout is issued
        _login_attempts[ip] = {"count": count, "lockout_until": new_lockout}
        return jsonify({"error": "Invalid email or password."}), 401

    # Success: clear rate limit, set session
    _login_attempts.pop(ip, None)
    session.permanent = True
    session["logged_in"] = True
    session["user_id"] = user["id"]
    session["name"] = user["name"]
    session["email"] = user["email"]

    try:
        conn = sqlite3.connect(USERS_DB)
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?",
                     (now.isoformat(), user["id"]))
        conn.commit()
        conn.close()
    except Exception:
        pass

    return jsonify({"ok": True, "name": user["name"]})


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/me")
def auth_me():
    if session.get("logged_in"):
        return jsonify({"logged_in": True, "name": session.get("name"),
                        "email": session.get("email")})
    return jsonify({"logged_in": False}), 401


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
            "rsi_14", "volume_ratio", "spread_bps", "order_imbalance",
            "trade_flow_ratio", "volatility_1m", "volatility_5m", "price_acceleration",
            "poly_yes_price", "poly_divergence", "poly_spread",
            "poly_volume_24h", "poly_order_imbalance",
            "cex_poly_lag", "momentum_consistency", "vol_adjusted_momentum",
            "seconds_remaining",
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

            # Significance: t-statistic approximation (* p<0.05, ** p<0.01)
            t_stat = corr * math.sqrt(n - 2) / math.sqrt(max(1 - corr**2, 1e-9))
            sig = "**" if abs(t_stat) > 3.0 else ("*" if abs(t_stat) > 1.96 else "")

            # Use median split for always-positive signals (volume, time, volatility)
            # so WR(lo) is never empty when no values fall at or below zero.
            if min(v_list) >= 0:
                threshold = sorted(v_list)[n // 2]
            else:
                threshold = 0.0

            hi_wins = [o for v, o in paired if v > threshold]
            lo_wins = [o for v, o in paired if v <= threshold]
            wr_hi = sum(hi_wins) / len(hi_wins) if hi_wins else None
            wr_lo = sum(lo_wins) / len(lo_wins) if lo_wins else None
            edge = (wr_hi - wr_lo) if (wr_hi is not None and wr_lo is not None) else None

            results.append({
                "signal":    col,
                "n":         n,
                "corr":      round(corr, 4),
                "sig":       sig,          # significance marker
                "threshold": round(threshold, 4),
                "wr_pos":    round(wr_hi, 4) if wr_hi is not None else None,
                "wr_neg":    round(wr_lo, 4) if wr_lo is not None else None,
                "edge":      round(edge, 4) if edge is not None else None,
                "abs_corr":  abs(corr),
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
    Read executed trades from the SQLite `trades` table (primary) with JSON fallback.
    Returns a flat list, newest-first, with all signal + resolution fields.
    """
    n = int(request.args.get("n", 500))
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT trade_id, timestamp, slug, side, score, confidence, entry_price,
                   position_size, shares, time_remaining,
                   momentum_5m, momentum_1m, momentum_15m, vs_ref, volume_ratio, rsi_14,
                   cex_poly_lag, price_acceleration, vol_adjusted_momentum,
                   poly_yes_price, market_outcome, trade_outcome, pnl, resolved, resolve_ts
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        """, (n,)).fetchall()
        conn.close()

        if rows:
            result = []
            for r in rows:
                (trade_id, ts, slug, side, score, confidence, entry_price,
                 position_size, shares, time_remaining,
                 m5, m1, m15, vs_ref, vol_ratio, rsi,
                 lag, price_accel, vol_adj_mom,
                 poly_price, market_outcome, trade_outcome, pnl, resolved, resolve_ts) = r
                result.append({
                    "ts":                    ts,
                    "trade_id":              trade_id,
                    "market_slug":           slug,
                    "trade_side":            side,
                    "signal_side":           side,
                    "trade_amount":          position_size,
                    "shares":                shares,
                    "entry_price":           entry_price,
                    "trade_result":          pnl,
                    "pnl":                   pnl,
                    "outcome":               market_outcome,
                    "trade_outcome":         trade_outcome,
                    "resolved":              resolved,
                    "resolve_ts":            resolve_ts,
                    "btc_vs_reference":      vs_ref,
                    "momentum_5m":           m5,
                    "momentum_1m":           m1,
                    "momentum_15m":          m15,
                    "volume_ratio":          vol_ratio,
                    "rsi_14":                rsi,
                    "cex_poly_lag":          lag,
                    "price_acceleration":    price_accel,
                    "vol_adjusted_momentum": vol_adj_mom,
                    "poly_yes_price":        poly_price,
                    "seconds_remaining":     time_remaining,
                    "signal_score":          score,
                    "signal_confidence":     confidence,
                })
            return jsonify(result)

        # Fallback: read from trade_log.json for pre-DB trades
        with open(TRADE_PATH) as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raw = [raw]
        result = []
        for t in reversed(raw):
            side = t.get("side")
            out  = t.get("market_outcome") or t.get("outcome")
            to   = t.get("trade_outcome") or (
                "win"  if (side == "yes" and out == "up") or (side == "no" and out == "down") else
                "loss" if out in ("up", "down") else None
            )
            result.append({
                "ts":                    t.get("timestamp"),
                "trade_id":              t.get("trade_id"),
                "market_slug":           t.get("slug"),
                "trade_side":            side,
                "signal_side":           side,
                "trade_amount":          t.get("position_size"),
                "shares":                t.get("shares"),
                "entry_price":           t.get("entry_price"),
                "trade_result":          t.get("pnl"),
                "pnl":                   t.get("pnl"),
                "outcome":               out,
                "trade_outcome":         to,
                "resolved":              t.get("resolved", 1 if out else 0),
                "btc_vs_reference":      t.get("vs_ref"),
                "momentum_5m":           t.get("momentum_5m") or t.get("momentum_pct"),
                "momentum_1m":           t.get("momentum_1m"),
                "momentum_15m":          t.get("momentum_15m"),
                "volume_ratio":          t.get("volume_ratio"),
                "rsi_14":                t.get("rsi_14"),
                "cex_poly_lag":          t.get("cex_poly_lag"),
                "price_acceleration":    t.get("price_acceleration"),
                "vol_adjusted_momentum": t.get("vol_adjusted_momentum"),
                "poly_yes_price":        t.get("poly_yes_price"),
                "seconds_remaining":     t.get("time_remaining"),
                "signal_score":          t.get("score"),
                "signal_confidence":     t.get("confidence"),
            })
        return jsonify(result[:n])
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pnl-summary")
def pnl_summary():
    """P&L summary sourced from the SQLite trades table."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        conn = get_db()
        total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        today_trades = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (today,)
        ).fetchone()[0]
        resolved_rows = conn.execute(
            "SELECT pnl, trade_outcome FROM trades WHERE resolved = 1"
        ).fetchall()
        today_pnl_row = conn.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE resolved=1 AND timestamp >= ?", (today,)
        ).fetchone()[0]
        conn.close()

        wins     = sum(1 for r in resolved_rows if r[1] == "win")
        losses   = sum(1 for r in resolved_rows if r[1] == "loss")
        total_pnl = sum(r[0] for r in resolved_rows if r[0] is not None)
        n_resolved = len(resolved_rows)
        return jsonify({
            "total_trades":    total_trades,
            "total_pnl":       round(total_pnl, 4),
            "today_trades":    today_trades,
            "today_pnl":       round(today_pnl_row or 0, 4),
            "win_rate":        round(wins / n_resolved, 4) if n_resolved else None,
            "resolved_trades": n_resolved,
            "wins":            wins,
            "losses":          losses,
        })
    except Exception as e:
        # Fallback to JSON if trades table doesn't exist yet
        try:
            with open(TRADE_PATH) as f:
                raw = json.load(f) if hasattr(f, 'read') else []
            if not isinstance(raw, list):
                raw = []
        except Exception:
            raw = []
        today_rows = [t for t in raw if (t.get("timestamp") or "")[:10] == today]
        resolved   = [t for t in raw if t.get("pnl") is not None]
        wins       = sum(1 for t in resolved if (t.get("pnl") or 0) > 0)
        return jsonify({
            "total_trades":    len(raw),
            "total_pnl":       round(sum(t["pnl"] for t in resolved), 4),
            "today_trades":    len(today_rows),
            "today_pnl":       round(sum(t.get("pnl") or 0 for t in today_rows), 4),
            "win_rate":        round(wins / len(resolved), 4) if resolved else None,
            "resolved_trades": len(resolved),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Polymarket Data API — real trade history, positions, metrics
# ─────────────────────────────────────────────────────────────────────────────

_POLY_DATA_API  = "https://data-api.polymarket.com"
_POLY_CACHE_TTL = 120   # seconds between refreshes
_poly_cache: dict = {}
_poly_cache_lock = threading.Lock()


def _poly_http(url: str, timeout: int = 20):
    try:
        req = _URLRequest(url, headers={"User-Agent": "polymarket-monitor/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        try:
            return {"error": json.loads(e.read().decode()).get("detail", str(e))}
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def _poly_cached(key: str, url: str):
    with _poly_cache_lock:
        entry = _poly_cache.get(key)
    if entry and (_time.time() - entry["ts"]) < _POLY_CACHE_TTL:
        return entry["data"]
    data = _poly_http(url)
    with _poly_cache_lock:
        _poly_cache[key] = {"ts": _time.time(), "data": data}
    return data


def _poly_wallet():
    """Resolve wallet address from environment variables."""
    for var in ("POLY_FUNDER", "POLY_WALLET_ADDRESS"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    pk = os.environ.get("POLY_PRIVATE_KEY", "").strip()
    if pk:
        try:
            from eth_account import Account
            if not pk.startswith("0x"):
                pk = "0x" + pk
            return Account.from_key(pk).address
        except Exception:
            pass
    return None


def _fetch_all_activity(address: str) -> list:
    """Paginate through all /activity records for an address (cached)."""
    cache_key = f"activity:{address}"
    with _poly_cache_lock:
        entry = _poly_cache.get(cache_key)
    if entry and (_time.time() - entry["ts"]) < _POLY_CACHE_TTL:
        return entry["data"]

    all_records, offset, page_size = [], 0, 500
    while True:
        url = f"{_POLY_DATA_API}/activity?user={address}&limit={page_size}&offset={offset}"
        result = _poly_http(url)
        if not result or (isinstance(result, dict) and result.get("error")):
            break
        page = result if isinstance(result, list) else result.get("history", [])
        if not page:
            break
        all_records.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    with _poly_cache_lock:
        _poly_cache[cache_key] = {"ts": _time.time(), "data": all_records}
    return all_records


def _ts_to_iso(ts) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _normalise_activity(raw: dict) -> dict | None:
    trade_type = (raw.get("type") or "TRADE").upper()
    if trade_type not in ("TRADE", "REDEMPTION", "REDEEM"):
        return None
    outcome    = raw.get("outcome") or ""
    side       = outcome.lower() if outcome else "yes"
    shares     = float(raw.get("shares")  or raw.get("size")    or 0)
    price      = float(raw.get("price")   or 0)
    amount     = float(raw.get("amount")  or raw.get("usdcSize") or (shares * price))
    timestamp  = int(raw.get("timestamp") or raw.get("createdAt") or 0)
    payout_raw = raw.get("payout")
    resolved   = bool(raw.get("marketClosed") or raw.get("closed") or raw.get("resolved"))
    pnl = None
    if resolved and payout_raw is not None:
        try:
            pnl = round(float(payout_raw) - amount, 4)
        except Exception:
            pass
    result_label = None
    if pnl is not None:
        result_label = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAK_EVEN")
    elif resolved:
        result_label = "RESOLVED"
    return {
        "trade_id":        raw.get("id") or raw.get("transactionHash") or "",
        "condition_id":    raw.get("conditionId") or raw.get("market") or "",
        "token_id":        raw.get("asset") or "",
        "question":        raw.get("title") or raw.get("question") or "",
        "side":            side,
        "type":            trade_type,
        "shares":          round(shares, 4),
        "price":           round(price, 4),
        "amount_paid":     round(amount, 4),
        "timestamp":       timestamp,
        "timestamp_iso":   _ts_to_iso(timestamp),
        "market_resolved": resolved,
        "payout":          float(payout_raw) if payout_raw is not None else None,
        "pnl":             pnl,
        "result":          result_label,
    }


@app.route("/api/poly-metrics")
def poly_metrics():
    """
    Performance metrics sourced from the local SQLite trades table.
    Unrealized P&L and open position count are supplemented from the
    Polymarket Data API when a wallet address is configured.
    """
    try:
        conn = get_db()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        total_trades      = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        today_trades      = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (today,)
        ).fetchone()[0]
        open_positions_db = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE resolved = 0"
        ).fetchone()[0]
        resolved_rows     = conn.execute(
            "SELECT pnl, trade_outcome, side FROM trades WHERE resolved = 1"
        ).fetchall()
        today_pnl_val     = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE resolved=1 AND timestamp >= ?", (today,)
        ).fetchone()[0] or 0
        best_pnl  = conn.execute(
            "SELECT MAX(pnl) FROM trades WHERE resolved=1 AND pnl IS NOT NULL"
        ).fetchone()[0]
        worst_pnl = conn.execute(
            "SELECT MIN(pnl) FROM trades WHERE resolved=1 AND pnl IS NOT NULL"
        ).fetchone()[0]
        conn.close()

        n_resolved = len(resolved_rows)
        wins       = [r for r in resolved_rows if r[1] == "win"]
        losses     = [r for r in resolved_rows if r[1] == "loss"]
        total_pnl  = sum(r[0] for r in resolved_rows if r[0] is not None)
        yes_trades = [r for r in resolved_rows if (r[2] or "").lower() == "yes"]
        no_trades  = [r for r in resolved_rows if (r[2] or "").lower() == "no"]
        yes_wins   = [r for r in yes_trades if r[1] == "win"]
        no_wins    = [r for r in no_trades  if r[1] == "win"]

        # Supplement with Polymarket API for unrealized P&L and live open positions
        unrealized     = 0.0
        open_positions = open_positions_db
        address = _poly_wallet()
        if address:
            try:
                pos_url    = f"{_POLY_DATA_API}/positions?user={address}&sizeThreshold=0.01&limit=500"
                pos_result = _poly_cached(f"positions:{address}", pos_url)
                positions  = pos_result if isinstance(pos_result, list) else \
                             (pos_result.get("positions", []) if isinstance(pos_result, dict) else [])
                active = [p for p in positions
                          if not p.get("closed") and float(p.get("size", 0) or 0) > 0.01]
                if active:
                    open_positions = len(active)
                    unrealized = sum(
                        float(p.get("currentValue") or 0) - float(p.get("initialValue") or 0)
                        for p in active
                    )
            except Exception:
                pass

        return jsonify({
            "total_trades":    total_trades,
            "resolved_trades": n_resolved,
            "open_positions":  open_positions,
            "total_pnl":       round(total_pnl, 4),
            "today_pnl":       round(today_pnl_val, 4),
            "today_trades":    today_trades,
            "unrealized_pnl":  round(unrealized, 4),
            "win_count":       len(wins),
            "loss_count":      len(losses),
            "win_rate":        round(len(wins) / n_resolved, 4) if n_resolved else None,
            "avg_pnl":         round(total_pnl / n_resolved, 4) if n_resolved else None,
            "best_trade_pnl":  round(best_pnl,  4) if best_pnl  is not None else None,
            "worst_trade_pnl": round(worst_pnl, 4) if worst_pnl is not None else None,
            "yes_total":       len(yes_trades),
            "yes_wins":        len(yes_wins),
            "yes_win_rate":    round(len(yes_wins) / len(yes_trades), 4) if yes_trades else None,
            "no_total":        len(no_trades),
            "no_wins":         len(no_wins),
            "no_win_rate":     round(len(no_wins) / len(no_trades), 4) if no_trades else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/poly-trades")
def poly_trades():
    """
    Full trade history from Polymarket Data API with active/closed status.
    Paginated internally; result is cached for 120 s.
    """
    address = _poly_wallet()
    if not address:
        return jsonify({"error": "Wallet address not configured", "trades": [], "total": 0}), 200

    try:
        raw_records = _fetch_all_activity(address)
        trades = [t for raw in raw_records for t in [_normalise_activity(raw)] if t]

        pos_url = f"{_POLY_DATA_API}/positions?user={address}&sizeThreshold=0.01&limit=500"
        pos_result = _poly_cached(f"positions:{address}", pos_url)
        positions = pos_result if isinstance(pos_result, list) else \
                    (pos_result.get("positions", []) if isinstance(pos_result, dict) else [])

        active_cids = {
            p.get("conditionId") or p.get("market", "")
            for p in positions
            if not p.get("closed") and float(p.get("size", 0) or 0) > 0.01
        }

        for t in trades:
            t["status"] = "active" if t["condition_id"] in active_cids else "closed"

        return jsonify({"trades": trades, "total": len(trades)})
    except Exception as e:
        return jsonify({"error": str(e), "trades": [], "total": 0}), 500


@app.route("/api/poly-positions")
def poly_positions_endpoint():
    """Current open positions from Polymarket (cached 60 s)."""
    address = _poly_wallet()
    if not address:
        return jsonify({"error": "Wallet address not configured", "positions": [], "total": 0}), 200

    try:
        url = f"{_POLY_DATA_API}/positions?user={address}&sizeThreshold=0.01&limit=500"
        result = _poly_cached(f"positions:{address}", url)
        if isinstance(result, dict) and result.get("error"):
            return jsonify({"error": result["error"], "positions": [], "total": 0}), 200

        positions = result if isinstance(result, list) else result.get("positions", [])
        out = []
        for p in positions:
            if not p:
                continue
            outcome  = (p.get("outcome") or "").lower()
            size     = float(p.get("size") or 0)
            avg_px   = float(p.get("avgPrice") or p.get("avg_price") or 0.5)
            init_val = float(p.get("initialValue") or (size * avg_px))
            cur_val  = float(p.get("currentValue") or init_val)
            out.append({
                "condition_id":   p.get("conditionId") or p.get("market") or "",
                "question":       p.get("title") or p.get("question") or "",
                "side":           outcome or "yes",
                "size":           round(size, 4),
                "entry_price":    round(avg_px, 4),
                "current_price":  round(float(p.get("curPrice") or avg_px), 4),
                "initial_value":  round(init_val, 4),
                "current_value":  round(cur_val, 4),
                "unrealized_pnl": round(cur_val - init_val, 4),
                "redeemable":     bool(p.get("redeemable", False)),
                "closed":         bool(p.get("closed", False)),
                "end_date":       p.get("endDate") or "",
            })
        return jsonify({"positions": out, "total": len(out)})
    except Exception as e:
        return jsonify({"error": str(e), "positions": [], "total": 0}), 500


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
        "import_limit":    "Infinite" if CONFIG.get("daily_import_limit") is None else CONFIG.get("daily_import_limit"),
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
    """
    Return the active config — always the optimizer-tuned /data/config.json when
    available, falling back to the image default /app/config.json.
    Re-read on every call so dashboard reflects optimizer changes without restart.
    """
    for path in (_DATA_CONFIG, _IMAGE_CONFIG):
        try:
            with open(path) as f:
                data = json.load(f)
            data["_config_source"] = path   # visible in dashboard for debugging
            return jsonify(data)
        except FileNotFoundError:
            continue
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "config.json not found"}), 404


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Analytics, operations, real-time stream
# ─────────────────────────────────────────────────────────────────────────────

OPT_HISTORY_PATH = "/data/optimizer_history.json"

@app.route("/api/optimizer-history")
def optimizer_history():
    """Return structured optimizer run history from optimizer_history.json."""
    try:
        path = OPT_HISTORY_PATH
        if not os.path.exists(path):
            return jsonify({"entries": [], "total": 0})
        with open(path) as f:
            entries = json.load(f)
        # Optionally limit to last N
        limit = int(request.args.get("limit", 200))
        entries = entries[-limit:]
        return jsonify({"entries": entries, "total": len(entries)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rolling-winrate")
def rolling_winrate():
    window = int(request.args.get("window", 30))
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT ts, signal_side, outcome
            FROM signal_observations
            WHERE would_trade = 1 AND resolved = 1 AND outcome IN ('up', 'down')
            ORDER BY id ASC
        """).fetchall()
        conn.close()
        result = []
        for i in range(len(rows)):
            w = rows[max(0, i - window + 1):i + 1]
            correct = sum(
                1 for r in w
                if (r["signal_side"] == "yes" and r["outcome"] == "up") or
                   (r["signal_side"] == "no"  and r["outcome"] == "down")
            )
            result.append({"ts": rows[i]["ts"], "win_rate": round(correct / len(w), 4), "n": len(w)})
        return jsonify({"window": window, "total": len(rows), "data": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/drawdown")
def drawdown():
    try:
        with open(TRADE_PATH) as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raw = [raw]
    except FileNotFoundError:
        return jsonify({"cumulative": [], "max_drawdown": 0, "total_pnl": 0, "peak_pnl": 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    resolved = sorted(
        [t for t in raw if t.get("pnl") is not None],
        key=lambda t: t.get("timestamp", ""),
    )
    if not resolved:
        return jsonify({"cumulative": [], "max_drawdown": 0, "total_pnl": 0, "peak_pnl": 0})

    cumulative, running, peak, max_dd = [], 0.0, 0.0, 0.0
    for t in resolved:
        running += t["pnl"]
        peak = max(peak, running)
        dd = peak - running
        max_dd = max(max_dd, dd)
        cumulative.append({"ts": t["timestamp"], "pnl": round(running, 4), "drawdown": round(dd, 4)})

    return jsonify({
        "cumulative":   cumulative,
        "max_drawdown": round(max_dd, 4),
        "total_pnl":    round(running, 4),
        "peak_pnl":     round(peak, 4),
    })


@app.route("/api/daily-pnl")
def daily_pnl():
    """Daily P&L grouped by resolved-trade date from the SQLite trades table."""
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT DATE(timestamp) AS day,
                   ROUND(SUM(pnl), 4) AS pnl,
                   COUNT(*) AS trades,
                   SUM(CASE WHEN trade_outcome = 'win'  THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN trade_outcome = 'loss' THEN 1 ELSE 0 END) AS losses
            FROM trades
            WHERE resolved = 1 AND pnl IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY day ASC
        """).fetchall()
        conn.close()
        days = [
            {"day": r[0], "pnl": r[1], "trades": r[2], "wins": r[3], "losses": r[4]}
            for r in rows
        ]
        return jsonify({"days": days})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/market-breakdown")
def market_breakdown():
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT market_slug,
                   COUNT(*) as total_cycles,
                   SUM(would_trade) as trade_signals,
                   ROUND(AVG(signal_score), 4) as avg_score,
                   MAX(ts) as last_seen
            FROM signal_observations
            WHERE market_slug IS NOT NULL AND market_slug != ''
            GROUP BY market_slug ORDER BY last_seen DESC LIMIT 30
        """).fetchall()
        wr_rows = conn.execute("""
            SELECT market_slug, signal_side, outcome
            FROM signal_observations
            WHERE would_trade = 1 AND resolved = 1
              AND outcome IN ('up', 'down') AND market_slug IS NOT NULL
        """).fetchall()
        conn.close()

        wr = {}
        for r in wr_rows:
            s = r["market_slug"]
            if s not in wr:
                wr[s] = {"total": 0, "correct": 0}
            wr[s]["total"] += 1
            if (r["signal_side"] == "yes" and r["outcome"] == "up") or \
               (r["signal_side"] == "no"  and r["outcome"] == "down"):
                wr[s]["correct"] += 1

        result = []
        for r in rows:
            slug = r["market_slug"]
            w = wr.get(slug, {})
            wt, wc = w.get("total", 0), w.get("correct", 0)
            result.append({
                "market_slug":    slug,
                "slug_short":     slug.split("-")[-1],
                "total_cycles":   r["total_cycles"],
                "trade_signals":  r["trade_signals"] or 0,
                "avg_score":      r["avg_score"],
                "win_rate":       round(wc / wt, 4) if wt > 0 else None,
                "resolved_trades": wt,
                "last_seen":      r["last_seen"],
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signal-heatmap")
def signal_heatmap():
    n = int(request.args.get("n", 50))
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT ts, market_slug, signal_score, would_trade, signal_side,
                   momentum_1m, momentum_5m, momentum_15m, rsi_14, volume_ratio,
                   order_imbalance, trade_flow_ratio, volatility_5m,
                   price_acceleration, btc_vs_reference, cex_poly_lag,
                   momentum_consistency, vol_adjusted_momentum, poly_yes_price
            FROM signal_observations ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return jsonify([dict(r) for r in reversed(rows)])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/optimize-weights")
def optimize_weights():
    """Scan score thresholds (0.50–0.80) to find optimal entry point."""
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT signal_side, outcome, signal_score
            FROM signal_observations
            WHERE resolved = 1 AND outcome IN ('up', 'down') AND signal_score IS NOT NULL
        """).fetchall()
        conn.close()

        if len(rows) < 10:
            return jsonify({"error": "Need ≥10 resolved observations for calibration."})

        scan = []
        for t_int in range(50, 81):
            t = t_int / 100.0
            lo = 1.0 - t
            trades = [(r["signal_side"], r["outcome"]) for r in rows
                      if r["signal_score"] > t or r["signal_score"] < lo]
            correct = sum(1 for side, out in trades
                          if (side == "yes" and out == "up") or (side == "no" and out == "down"))
            n = len(trades)
            wr = correct / n if n > 0 else 0.0
            scan.append({"threshold": t, "trades": n, "correct": correct,
                         "win_rate": round(wr, 4), "ev": round(wr * 2 - 1, 4)})

        candidates = [s for s in scan if s["trades"] >= 5]
        best = max(candidates, key=lambda x: x["win_rate"]) if candidates else None
        return jsonify({
            "total_resolved":          len(rows),
            "scan":                    scan,
            "recommended_threshold":   best["threshold"] if best else None,
            "recommended_wr":          best["win_rate"] if best else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/daily-digest")
def daily_digest():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        conn = get_db()
        stats = conn.execute("""
            SELECT COUNT(*) as cycles, SUM(would_trade) as signals,
                   ROUND(AVG(signal_score), 4) as avg_score, MAX(ts) as last_cycle
            FROM signal_observations WHERE ts >= ?
        """, (today,)).fetchone()
        top_block = conn.execute("""
            SELECT filter_reason, COUNT(*) as cnt
            FROM signal_observations
            WHERE ts >= ? AND would_trade = 0 AND filter_reason IS NOT NULL
            GROUP BY filter_reason ORDER BY cnt DESC LIMIT 1
        """, (today,)).fetchone()
        conn.close()
    except Exception:
        stats = top_block = None

    today_trades, pnl, wins, resolved_count = [], 0.0, 0, 0
    try:
        with open(TRADE_PATH) as f:
            all_trades = json.load(f)
        today_trades = [t for t in all_trades if (t.get("timestamp") or "")[:10] == today]
        resolved = [t for t in today_trades
                    if t.get("outcome") in ("up", "down") and t.get("pnl") is not None]
        pnl = sum(t["pnl"] for t in resolved)
        wins = sum(1 for t in resolved if t["pnl"] > 0)
        resolved_count = len(resolved)
    except Exception:
        pass

    return jsonify({
        "date":           today,
        "cycles_today":   stats["cycles"]  if stats else 0,
        "signals_today":  int(stats["signals"] or 0) if stats else 0,
        "avg_score":      stats["avg_score"] if stats else None,
        "last_cycle":     stats["last_cycle"] if stats else None,
        "top_block":      top_block["filter_reason"] if top_block else None,
        "trades_today":   len(today_trades),
        "pnl_today":      round(pnl, 4),
        "resolved_today": resolved_count,
        "win_rate_today": round(wins / resolved_count, 4) if resolved_count > 0 else None,
    })


@app.route("/api/kill-switch", methods=["GET", "POST"])
def kill_switch():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        active = bool(data.get("active", False))
        try:
            with open(KILL_SWITCH_FILE, "w") as f:
                json.dump({"active": active,
                           "set_at": datetime.now(timezone.utc).isoformat()}, f)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"active": active, "ok": True})

    try:
        with open(KILL_SWITCH_FILE) as f:
            d = json.load(f)
        return jsonify({"active": d.get("active", False), "set_at": d.get("set_at")})
    except FileNotFoundError:
        return jsonify({"active": False, "set_at": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset-observations", methods=["POST"])
def reset_observations():
    """
    DELETE all rows from signal_observations (and optionally window_refs cache).
    Requires a JSON body: {"confirm": "RESET"} to prevent accidental calls.
    Use this when collected data is corrupt and you want to start fresh.
    Trades table is NOT touched — live trade history is preserved.
    """
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    if data.get("confirm") != "RESET":
        return jsonify({"error": 'Must send {"confirm": "RESET"} in body'}), 400
    try:
        conn = get_db()
        before = conn.execute("SELECT COUNT(*) FROM signal_observations").fetchone()[0]
        conn.execute("DELETE FROM signal_observations")
        # Reset the SQLite auto-increment counter so IDs restart from 1
        conn.execute("DELETE FROM sqlite_sequence WHERE name='signal_observations'")
        conn.commit()
        conn.close()

        # Also clear the window reference price cache so stale refs don't pollute the next run
        window_refs_path = os.path.join(os.environ.get("DATA_DIR", "/data"), "window_refs.json")
        cleared_refs = False
        try:
            if os.path.exists(window_refs_path):
                os.remove(window_refs_path)
                cleared_refs = True
        except Exception:
            pass

        return jsonify({
            "ok": True,
            "deleted_rows": before,
            "window_refs_cleared": cleared_refs,
            "message": f"Deleted {before} observations. Collector will start fresh on next cycle.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def save_config():
    """
    Merge-update the live config.

    Send only the keys you want to change — existing keys are preserved.
    Writes to /data/config.json (persistent volume) so changes survive rebuilds
    and are picked up by the trader on the next cycle automatically.

    Example:
        curl -X POST http://localhost:5000/api/config \
             -H 'Content-Type: application/json' \
             -d '{"max_position": 5.0, "min_momentum_pct": 0.10}'
    """
    try:
        updates = request.get_json(silent=True)
        if not updates or not isinstance(updates, dict):
            return jsonify({"error": "Body must be a JSON object of key→value pairs"}), 400

        # Load the current live config
        current = {}
        for path in (_DATA_CONFIG, _IMAGE_CONFIG):
            try:
                with open(path) as f:
                    current = json.load(f)
                break
            except FileNotFoundError:
                continue

        # Strip internal metadata injected by GET endpoint
        updates.pop("_config_source", None)

        # Deep-merge: for nested sections merge key-by-key so optimizer-tuned
        # keys the dashboard didn't touch are preserved
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key].update(value)
            else:
                current[key] = value

        # Always write to the persistent volume
        with open(_DATA_CONFIG, "w") as f:
            json.dump(current, f, indent=2)

        return jsonify({"ok": True, "updated": list(updates.keys())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stream")
def event_stream():
    """Server-Sent Events: pushes new observation row whenever id changes."""
    def gen():
        last_id = None
        while True:
            try:
                conn = get_db()
                row = conn.execute("""
                    SELECT id, ts, market_slug, seconds_remaining,
                           signal_score, signal_side, would_trade, filter_reason,
                           btc_vs_reference, momentum_5m, poly_yes_price
                    FROM signal_observations ORDER BY id DESC LIMIT 1
                """).fetchone()
                conn.close()
                if row:
                    d = dict(row)
                    if d["id"] != last_id:
                        last_id = d["id"]
                        yield f"data: {json.dumps(d)}\n\n"
                    else:
                        yield ": keep-alive\n\n"
                else:
                    yield ": keep-alive\n\n"
            except Exception:
                yield ": keep-alive\n\n"
            _time.sleep(5)

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.route("/api/export/<table_name>")
def export_csv(table_name):
    """Export a database table as a downloadable CSV file."""
    ALLOWED_TABLES = {
        "observations": ("signal_observations", "signal_observations.csv"),
        "trades":        ("trades",              "trades.csv"),
    }
    if table_name not in ALLOWED_TABLES:
        return jsonify({"error": f"Unknown table '{table_name}'. Choose: {list(ALLOWED_TABLES)}"}), 400

    db_table, filename = ALLOWED_TABLES[table_name]
    try:
        conn = get_db()
        rows = conn.execute(f"SELECT * FROM {db_table} ORDER BY id ASC").fetchall()
        conn.close()

        output = io.StringIO()
        writer = csv.writer(output)
        if rows:
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(list(row))

        csv_bytes = output.getvalue().encode("utf-8")
        return Response(
            csv_bytes,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(csv_bytes)),
            },
        )
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