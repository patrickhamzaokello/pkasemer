#!/usr/bin/env python3
"""
FastLoop Monitor API

Serves signal data and status to the monitoring dashboard.
Runs on port 5000 inside the container (mapped to host port).
"""

import os
import json
import sqlite3
import math
from datetime import datetime, timezone
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="/app/dashboard")
CORS(app)

DB_PATH = os.environ.get("DB_PATH", "/data/signal_research.db")
STATUS_PATH = "/data/status.json"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
                   price_now
            FROM signal_observations
            ORDER BY id DESC LIMIT 200
        """).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/correlations")
def correlations():
    """Compute live signal correlations from resolved observations."""
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
            "momentum_1m", "momentum_5m", "momentum_15m",
            "rsi_14", "volume_ratio", "order_imbalance",
            "trade_flow_ratio", "volatility_5m", "price_acceleration",
            "poly_yes_price", "poly_divergence", "cex_poly_lag",
            "momentum_consistency", "vol_adjusted_momentum",
        ]

        outcomes = [1 if r["outcome"] == "up" else 0 for r in rows]
        results = []

        for col in SIGNAL_COLS:
            vals = [r[col] for r in rows if r[col] is not None]
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
        return jsonify({
            "total_resolved": len(rows),
            "data": results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/summary")
def summary():
    """High-level DB summary stats."""
    try:
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM signal_observations").fetchone()[0]
        resolved = conn.execute(
            "SELECT COUNT(*) FROM signal_observations WHERE resolved=1").fetchone()[0]
        up = conn.execute(
            "SELECT COUNT(*) FROM signal_observations WHERE outcome='up'").fetchone()[0]
        down = conn.execute(
            "SELECT COUNT(*) FROM signal_observations WHERE outcome='down'").fetchone()[0]
        unclear = conn.execute(
            "SELECT COUNT(*) FROM signal_observations WHERE outcome='unclear'").fetchone()[0]
        latest = conn.execute(
            "SELECT ts FROM signal_observations ORDER BY id DESC LIMIT 1").fetchone()
        oldest = conn.execute(
            "SELECT ts FROM signal_observations ORDER BY id ASC LIMIT 1").fetchone()

        # Signal averages from last 50 obs
        recent_avg = conn.execute("""
            SELECT
                AVG(momentum_5m) as avg_m5,
                AVG(order_imbalance) as avg_oi,
                AVG(trade_flow_ratio) as avg_tfr,
                AVG(rsi_14) as avg_rsi,
                AVG(poly_yes_price) as avg_poly
            FROM (SELECT * FROM signal_observations ORDER BY id DESC LIMIT 50)
        """).fetchone()
        conn.close()

        return jsonify({
            "total": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "outcomes": {"up": up, "down": down, "unclear": unclear},
            "date_range": {
                "oldest": oldest[0] if oldest else None,
                "latest": latest[0] if latest else None,
            },
            "recent_avg": {
                "momentum_5m": round(recent_avg[0], 4) if recent_avg[0] else None,
                "order_imbalance": round(recent_avg[1], 4) if recent_avg[1] else None,
                "trade_flow_ratio": round(recent_avg[2], 4) if recent_avg[2] else None,
                "rsi_14": round(recent_avg[3], 2) if recent_avg[3] else None,
                "poly_yes_price": round(recent_avg[4], 4) if recent_avg[4] else None,
            }
        })
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