#!/usr/bin/env python3
"""
FastLoop Scheduler

Runs signal collection and trading 24/7 (Polymarket operates continuously).

Modes (set via FASTLOOP_MODE env var):
  collect   — run signal_research.py --collect (default)
  trade     — run fast_trader.py --live
  both      — collect + trade in parallel
  dry       — run fast_trader.py --dry-run (no real trades)

Environment variables:
  FASTLOOP_MODE         collect | trade | both | dry
  FASTLOOP_INTERVAL     seconds between cycles (default: 20)
  FASTLOOP_ASSET        BTC | ETH | SOL (default: BTC)
  FASTLOOP_WINDOW       5m | 15m (default: 5m)
  SIMMER_API_KEY        required for trade/both modes
  LOG_LEVEL             INFO | DEBUG (default: INFO)
"""

import os
import sys
import time
import signal
import logging
import sqlite3
import subprocess
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MODE = os.environ.get("FASTLOOP_MODE", "collect").lower()
INTERVAL = int(os.environ.get("FASTLOOP_INTERVAL", "20"))
ASSET = os.environ.get("FASTLOOP_ASSET", "BTC").upper()
WINDOW = os.environ.get("FASTLOOP_WINDOW", "5m")
DB_PATH = os.environ.get("DB_PATH", "/data/signal_research.db")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/app/logs/scheduler.log", mode="a"),
    ],
)
log = logging.getLogger("scheduler")


# ─────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────

def scheduler_status_str(now=None):
    """Human-readable running status."""
    if now is None:
        now = datetime.now(timezone.utc)
    return f"RUNNING — {now.strftime('%a %H:%M UTC')}"


# ─────────────────────────────────────────────
# Cycle execution
# ─────────────────────────────────────────────

def run_collector():
    """Run one collection cycle via signal_research.py."""
    try:
        # Import and run directly (same process, more efficient than subprocess)
        sys.path.insert(0, "/app")
        from signal_research import collect_one, init_db
        conn = init_db(DB_PATH)
        symbol = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}.get(ASSET, "BTCUSDT")
        collect_one(conn, asset=ASSET, window=WINDOW, symbol=symbol)
        conn.close()
        return True
    except Exception as e:
        log.error(f"Collector error: {e}", exc_info=True)
        return False


def run_trader(dry=False):
    """Run one trading cycle via fast_trader.py."""
    try:
        sys.path.insert(0, "/app")
        from fast_trader import run_fast_market_strategy
        run_fast_market_strategy(dry_run=dry, quiet=False)
        return True
    except Exception as e:
        log.error(f"Trader error: {e}", exc_info=True)
        return False


def run_resolver():
    """Resolve pending outcomes every 10 minutes."""
    try:
        sys.path.insert(0, "/app")
        from signal_research import resolve_outcomes, init_db
        conn = init_db(DB_PATH)
        n = resolve_outcomes(conn)
        if n > 0:
            log.info(f"Resolved {n} outcomes")
        conn.close()
    except Exception as e:
        log.error(f"Resolver error: {e}", exc_info=True)


# ─────────────────────────────────────────────
# Stats for monitoring
# ─────────────────────────────────────────────

def write_status(cycle, last_action, market_open):
    """Write current status to /data/status.json for the dashboard."""
    import json
    now = datetime.now(timezone.utc)

    stats = {
        "ts": now.isoformat(),
        "cycle": cycle,
        "mode": MODE,
        "asset": ASSET,
        "window": WINDOW,
        "market_open": True,
        "market_status": scheduler_status_str(now),
        "last_action": last_action,
        "interval_s": INTERVAL,
    }

    # DB stats
    try:
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM signal_observations").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM signal_observations WHERE resolved=1").fetchone()[0]
        up = conn.execute("SELECT COUNT(*) FROM signal_observations WHERE outcome='up'").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM signal_observations WHERE outcome='down'").fetchone()[0]
        unclear = conn.execute("SELECT COUNT(*) FROM signal_observations WHERE outcome='unclear'").fetchone()[0]
        recent = conn.execute("""
            SELECT ts, momentum_5m, poly_yes_price, order_imbalance, trade_flow_ratio, seconds_remaining
            FROM signal_observations ORDER BY id DESC LIMIT 20
        """).fetchall()
        conn.close()

        stats["db"] = {
            "total": total,
            "resolved": resolved,
            "unresolved": total - resolved,
            "outcomes": {"up": up, "down": down, "unclear": unclear},
        }
        stats["recent"] = [
            {
                "ts": r[0],
                "m5": round(r[1], 4) if r[1] else None,
                "poly": round(r[2], 4) if r[2] else None,
                "oi": round(r[3], 4) if r[3] else None,
                "tfr": round(r[4], 4) if r[4] else None,
                "secs": round(r[5], 0) if r[5] else None,
            }
            for r in recent
        ]
    except Exception as e:
        stats["db_error"] = str(e)

    try:
        with open("/data/status.json", "w") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass


# ─────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────

_running = True

def _handle_signal(signum, frame):
    global _running
    log.info(f"Received signal {signum}, shutting down gracefully...")
    _running = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info(f"FastLoop Scheduler starting")
    log.info(f"  Mode:     {MODE}")
    log.info(f"  Asset:    {ASSET} {WINDOW}")
    log.info(f"  Interval: {INTERVAL}s")
    log.info(f"  DB:       {DB_PATH}")
    log.info(f"  Running:  24/7 (Polymarket is always open)")
    log.info("=" * 60)

    cycle = 0
    last_resolve = datetime.now(timezone.utc)
    last_action = "starting"

    while _running:
        now = datetime.now(timezone.utc)
        cycle += 1
        log.info(f"[cycle {cycle}] {scheduler_status_str(now)}")

        # Run based on mode
        if MODE in ("collect", "both", "dry"):
            ok = run_collector()
            last_action = f"collect {'ok' if ok else 'err'} @ {now.strftime('%H:%M:%S')}"

        if MODE in ("trade", "both"):
            ok = run_trader(dry=False)
            last_action = f"trade {'ok' if ok else 'err'} @ {now.strftime('%H:%M:%S')}"

        if MODE == "dry":
            ok = run_trader(dry=True)
            last_action = f"dry-run {'ok' if ok else 'err'} @ {now.strftime('%H:%M:%S')}"

        # Resolve outcomes every 10 minutes
        if (now - last_resolve).total_seconds() > 600:
            run_resolver()
            last_resolve = now

        write_status(cycle, last_action, True)


        # Sleep with interrupt awareness
        for _ in range(INTERVAL):
            if not _running:
                break
            time.sleep(1)

    log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()