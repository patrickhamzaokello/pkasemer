#!/usr/bin/env python3
"""
Simmer Pknwitq Trading Skill

Trades Polymarket BTC 5-minute fast markets using CEX price momentum.
Default signal: Binance BTCUSDT candles. Agents can customize signal source.

Usage:
    python fast_trader.py              # Dry run (show opportunities, no trades)
    python fast_trader.py --live       # Execute real trades
    python fast_trader.py --positions  # Show current fast market positions
    python fast_trader.py --quiet      # Only output on trades/errors

Requires:
    POLY_PRIVATE_KEY   Polygon wallet private key
    POLY_API_KEY       Polymarket CLOB API key (or derived automatically)
    POLY_API_SECRET    Polymarket CLOB API secret
    POLY_API_PASSPHRASE  Polymarket CLOB API passphrase

Changes from original:
    - Fixed import_fast_market_market(): result was fetched but never parsed —
      every trade attempt was silently failing.
    - Fixed _CACHE_FILE path: now uses DATA_DIR pattern instead of hardcoded /data/
    - Removed discover_fast_market_markets / find_best_fast_market — now in market_utils.py
      to eliminate the circular import with signal_research.py.
"""

import os
import sys
import json
import sqlite3
import argparse
import time
from datetime import datetime, timezone, timedelta  # timedelta kept for future use
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from pathlib import Path

from composite_signal import get_composite_signal, _calc_cex_poly_lag, detect_market_regime, check_hour_gate
from signal_research import extract_cex_signals, extract_poly_signals, get_window_reference_price, _load_ref_cache
from market_utils import discover_fast_market_markets, find_best_fast_market

# Force line-buffered stdout for non-TTY environments (cron, Docker, OpenClaw)
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

try:
    from tradejournal import log_trade
    JOURNAL_AVAILABLE = True
except ImportError:
    try:
        from skills.tradejournal import log_trade
        JOURNAL_AVAILABLE = True
    except ImportError:
        JOURNAL_AVAILABLE = False
        def log_trade(*args, **kwargs):
            pass

# =============================================================================
# Constants
# =============================================================================

TRADE_SOURCE = "sdk:pknwitq"
SMART_SIZING_PCT = 0.05
MIN_SHARES_PER_ORDER = 6.0
# These are defaults; overridden each cycle from config.json
MIN_SCORE_TO_IMPORT = 0.65
IMPORT_DAILY_LIMIT  = 1000
MIN_ENTRY_PRICE     = 0.35
MIN_LIQUIDITY_RATIO = 0.20


ASSET_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

COINGECKO_ASSETS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

# =============================================================================
# Data directory — consistent with signal_research.py
# =============================================================================

_DATA_DIR = "/data" if os.path.isdir("/data") else str(Path(__file__).parent)
_CACHE_FILE = Path(_DATA_DIR) / "market_id_cache.json"

# =============================================================================
# Configuration (config.json > env vars > defaults)
# =============================================================================

CONFIG_SCHEMA = {
    "min_momentum_pct":        {"default": 0.15,      "env": "SIMMER_SPRINT_MOMENTUM",     "type": float},
    "max_position":            {"default": 5.0,       "env": "SIMMER_SPRINT_MAX_POSITION", "type": float},
    "signal_source":           {"default": "binance", "env": "SIMMER_SPRINT_SIGNAL",       "type": str},
    "min_time_remaining":      {"default": 120,       "env": "SIMMER_SPRINT_MIN_TIME",     "type": int},
    "max_time_remaining":      {"default": 420,       "env": "SIMMER_SPRINT_MAX_TIME",     "type": int},
    "asset":                   {"default": "BTC",     "env": "SIMMER_SPRINT_ASSET",        "type": str},
    "window":                  {"default": "5m",      "env": "SIMMER_SPRINT_WINDOW",       "type": str},
    "volume_confidence":       {"default": True,      "env": "SIMMER_SPRINT_VOL_CONF",     "type": bool},
    "daily_budget":            {"default": 10.0,      "env": "SIMMER_SPRINT_DAILY_BUDGET", "type": float},
    "composite_threshold":     {"default": 0.60,      "env": "SIMMER_SPRINT_COMP_THRESH",  "type": float},
    "max_position_per_window": {"default": 3.50,      "env": "SIMMER_MAX_PER_WINDOW",      "type": float},
    "max_no_score":            {"default": 0.284,     "env": "SIMMER_MAX_NO_SCORE",        "type": float},
    "webhook_url":             {"default": "",        "env": "SIMMER_WEBHOOK_URL",         "type": str},
    "telegram_bot_token":      {"default": "",        "env": "TELEGRAM_BOT_TOKEN",         "type": str},
    "telegram_chat_id":        {"default": "",        "env": "TELEGRAM_CHAT_ID",           "type": str},
}


def _load_config(schema, skill_file, config_filename="config.json"):
    config_path = Path(skill_file).parent / config_filename
    file_cfg = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                file_cfg = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    result = {}
    for key, spec in schema.items():
        if key in file_cfg:
            result[key] = file_cfg[key]
        elif spec.get("env") and os.environ.get(spec["env"]):
            val = os.environ.get(spec["env"])
            type_fn = spec.get("type", str)
            try:
                result[key] = (
                    val.lower() in ("true", "1", "yes")
                    if type_fn is bool
                    else type_fn(val)
                )
            except (ValueError, TypeError):
                result[key] = spec.get("default")
        else:
            result[key] = spec.get("default")
    return result


def _get_config_path(skill_file, config_filename="config.json"):
    return Path(skill_file).parent / config_filename


def _update_config(updates, skill_file, config_filename="config.json"):
    config_path = Path(skill_file).parent / config_filename
    existing = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    existing.update(updates)
    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)
    return existing


# Load config at module level
cfg = _load_config(CONFIG_SCHEMA, __file__)
# raw_cfg: full config.json including signal_weights, rsi thresholds,
# poly gate bounds, etc. Passed to get_composite_signal so all config
# values take effect without needing entries in CONFIG_SCHEMA.
try:
    with open(_get_config_path(__file__)) as _f:
        raw_cfg = json.load(_f)
except Exception:
    raw_cfg = cfg  # fall back to schema-filtered cfg
ENTRY_THRESHOLD    = cfg["entry_threshold"]
MIN_MOMENTUM_PCT   = cfg["min_momentum_pct"]
MAX_POSITION_USD   = cfg["max_position"]
SIGNAL_SOURCE      = cfg["signal_source"]
LOOKBACK_MINUTES   = cfg["lookback_minutes"]
MIN_TIME_REMAINING = cfg["min_time_remaining"]
MAX_TIME_REMAINING = cfg.get("max_time_remaining", 420)
ASSET              = cfg["asset"].upper()
WINDOW             = cfg["window"]
VOLUME_CONFIDENCE  = cfg["volume_confidence"]
DAILY_BUDGET       = cfg["daily_budget"]
MIN_LIQUIDITY_RATIO = 0.20  # minimum volume ratio for order execution

# =============================================================================
# Daily Budget & Import Tracking
# =============================================================================

# daily_spend.json lives on /data (shared volume) so monitor container can read it
_SPEND_FILE = Path(_DATA_DIR) / "daily_spend.json"


def _load_daily_spend(*_):
    """Load today's spend record, resetting automatically on a new UTC day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _SPEND_FILE.exists():
        try:
            with open(_SPEND_FILE) as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return {"date": today, "spent": 0.0, "trades": 0, "imports_today": 0}


def _save_daily_spend(spend_data):
    with open(_SPEND_FILE, "w") as f:
        json.dump(spend_data, f, indent=2)


def _get_import_count_today():
    return _load_daily_spend().get("imports_today", 0)


def _increment_import_count():
    spend = _load_daily_spend()
    spend["imports_today"] = spend.get("imports_today", 0) + 1
    _save_daily_spend(spend)


# =============================================================================
# Market ID Cache
# =============================================================================

def _load_market_cache():
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_market_cache(cache):
    try:
        _CACHE_FILE.write_text(json.dumps(cache))
    except Exception:
        pass


_market_id_cache = _load_market_cache()


# =============================================================================
# Local Trade Log
# =============================================================================

_TRADE_LOG_FILE   = Path(_DATA_DIR) / "trade_log.json"
_KILL_SWITCH_FILE = Path(_DATA_DIR) / "kill_switch.json"


def _is_kill_switch_active() -> bool:
    """Return True if the UI emergency-stop has been triggered."""
    try:
        if _KILL_SWITCH_FILE.exists():
            return json.loads(_KILL_SWITCH_FILE.read_text()).get("active", False)
    except Exception:
        pass
    return False


def _send_webhook(url: str, slug: str, side: str, score: float,
                  position_size: float, shares: float, entry_price: float) -> None:
    """POST a trade alert to a Telegram/Discord-compatible webhook URL."""
    if not url or not url.strip():
        return
    arrow = "🟢" if side == "yes" else "🔴"
    payload = json.dumps({
        "text": (
            f"{arrow} TRADE: {side.upper()} | {slug}\n"
            f"Score: {score:.3f} | ${position_size:.2f} @ ${entry_price:.3f} ({shares:.1f} shares)"
        )
    }).encode()
    try:
        req = Request(url.strip(), data=payload,
                      headers={"Content-Type": "application/json"}, method="POST")
        urlopen(req, timeout=5)
    except Exception:
        pass  # webhook failures are non-fatal


def _fmt_slug(slug):
    """
    Format a market slug for uniform log display.

    btc-updown-5m-1772962200  →  BTC-5m@962200
    eth-updown-15m-1772962200 →  ETH-15m@962200

    Falls back to the last 20 chars for any unrecognised format.
    """
    parts = slug.split("-updown-", 1)
    if len(parts) == 2:
        coin = parts[0].upper()
        win_ts = parts[1].split("-", 1)
        if len(win_ts) == 2:
            window, ts = win_ts
            return f"{coin}-{window}@{ts[-6:]}"
    return slug[-20:]


def _send_telegram(token: str, chat_id: str, slug: str, side: str, score: float,
                   position_size: float, shares: float, entry_price: float,
                   remaining: float, lag: float):
    """Send a live-trade alert via Telegram Bot API. Returns message_id or None."""
    if not token or not token.strip():
        print("[telegram] SKIP entry alert: TELEGRAM_BOT_TOKEN not set", flush=True)
        return None
    if not chat_id or not chat_id.strip():
        print("[telegram] SKIP entry alert: TELEGRAM_CHAT_ID not set", flush=True)
        return None
    arrow  = "\U0001f7e2" if side == "yes" else "\U0001f534"  # 🟢 / 🔴
    bar    = "\u2501" * 20  # ━━━━━━━━━━━━━━━━━━━━
    market = slug if len(slug) <= 40 else slug[:37] + "..."
    text   = (
        f"{arrow} <b>LIVE TRADE \u2014 {side.upper()}</b>\n"
        f"<code>{bar}</code>\n"
        f"Market: <code>{market}</code>\n"
        f"Score:  <b>{score:.3f}</b>\n"
        f"Entry:  ${entry_price:.3f}  ({shares:.1f} shares)\n"
        f"Size:   <b>${position_size:.2f}</b>  |  {remaining:.0f}s left\n"
        f"Lag:    {lag:+.3f}"
    )
    payload = json.dumps({"chat_id": chat_id.strip(), "text": text,
                          "parse_mode": "HTML"}).encode()
    try:
        url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
        req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        msg_id = data.get("result", {}).get("message_id")
        print(f"[telegram] entry alert sent → {side.upper()} ${position_size:.2f} (msg_id={msg_id})", flush=True)
        return msg_id
    except Exception as _tg_err:
        print(f"[telegram] entry alert FAILED: {_tg_err}", flush=True)
        return None


def _send_telegram_outcome(token: str, chat_id: str, trade: dict,
                           reply_to_message_id=None) -> None:
    """Send a trade resolution alert via Telegram Bot API."""
    if not token or not token.strip():
        print("[telegram] SKIP outcome alert: TELEGRAM_BOT_TOKEN not set", flush=True)
        return
    if not chat_id or not chat_id.strip():
        print("[telegram] SKIP outcome alert: TELEGRAM_CHAT_ID not set", flush=True)
        return
    outcome  = trade.get("outcome", "?")
    pnl      = trade.get("pnl", 0.0) or 0.0
    side     = trade.get("side", "?")
    slug     = trade.get("slug", "?")
    size     = trade.get("position_size", 0.0) or 0.0
    entry    = trade.get("entry_price", 0.0) or 0.0
    market   = slug if len(slug) <= 40 else slug[:37] + "..."

    if outcome == "win":
        emoji = "\u2705"  # ✅
        label = "WIN"
    elif outcome == "loss":
        emoji = "\u274c"  # ❌
        label = "LOSS"
    else:
        emoji = "\u2753"  # ❓
        label = outcome.upper()

    pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
    bar     = "\u2501" * 20
    text    = (
        f"{emoji} <b>RESOLVED \u2014 {label}</b>\n"
        f"<code>{bar}</code>\n"
        f"Market: <code>{market}</code>\n"
        f"Side:   <b>{side.upper()}</b>  |  Entry: ${entry:.3f}\n"
        f"Size:   ${size:.2f}\n"
        f"PnL:    <b>{pnl_str}</b>"
    )
    body = {"chat_id": chat_id.strip(), "text": text, "parse_mode": "HTML"}
    if reply_to_message_id:
        body["reply_to_message_id"] = reply_to_message_id
    payload = json.dumps(body).encode()
    try:
        url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
        req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        urlopen(req, timeout=8)
        print(f"[telegram] outcome alert sent → {label} {pnl_str}", flush=True)
    except Exception as _tg_err:
        print(f"[telegram] outcome alert FAILED: {_tg_err}", flush=True)


def _log_trade_local(trade_id, side, score, confidence, entry_price, position_size,
                      shares, slug, remaining, cex_signals, poly_signals, lag=0.0):
    """Append trade to local JSON log and write to SQLite trades table."""
    now = datetime.now(timezone.utc)
    record = {
        "timestamp":              now.isoformat(),
        "hour_utc":               now.hour,
        "trade_id":               trade_id,
        "source":                 TRADE_SOURCE,
        "thesis":                 f"{side.upper()} score={score:.3f} conf={confidence:.3f}",
        "asset":                  ASSET,
        "signal_source":          SIGNAL_SOURCE,
        "slug":                   slug,
        "side":                   side,
        "score":                  round(score, 4),
        "confidence":             round(confidence, 4),
        "entry_price":            round(entry_price, 4),
        "position_size":          round(position_size, 4),
        "shares":                 round(shares, 2),
        "time_remaining":         int(remaining),
        # CEX signals at trade time
        "momentum_5m":            round(cex_signals.get("momentum_5m", 0) or 0, 4),
        "momentum_1m":            round(cex_signals.get("momentum_1m", 0) or 0, 4),
        "momentum_15m":           round(cex_signals.get("momentum_15m", 0) or 0, 4),
        "vs_ref":                 round(cex_signals.get("btc_vs_reference", 0) or 0, 4),
        "volume_ratio":           round(cex_signals.get("volume_ratio", 1.0) or 1.0, 3),
        "rsi_14":                 round(cex_signals.get("rsi_14", 50) or 50, 2),
        "cex_poly_lag":           round(lag, 4),
        "price_acceleration":     round(cex_signals.get("price_acceleration", 0) or 0, 4),
        "vol_adjusted_momentum":  round(cex_signals.get("vol_adjusted_momentum", 0) or 0, 4),
        # Poly signals
        "poly_yes_price":         round(poly_signals.get("poly_yes_price", 0.5) or 0.5, 4),
        # Resolution — filled in by resolve_trade_outcomes
        "market_outcome":         None,
        "outcome":                None,
        "pnl":                    None,
        "resolved":               0,
    }

    # ── JSON log (append) ─────────────────────────────────────────────────────
    trades = []
    if _TRADE_LOG_FILE.exists():
        try:
            trades = json.loads(_TRADE_LOG_FILE.read_text())
        except Exception:
            trades = []
    trades.append(record)
    _TRADE_LOG_FILE.write_text(json.dumps(trades, indent=2))

    # ── SQLite trades table (upsert; enables in-place outcome updates) ────────
    try:
        from signal_research import upsert_trade, init_db
        db_path = os.environ.get("DB_PATH", "/data/signal_research.db")
        conn = init_db(db_path)
        upsert_trade(conn, record)
        conn.close()
    except Exception as _db_err:
        print(f"  [trade-db] write failed: {_db_err}", flush=True)

def warm_import_cache(asset="BTC"):
    from market_utils import get_fast_market_slugs
    slugs = get_fast_market_slugs(asset, include_next=True)
    print(f"  [cache warm] checking {len(slugs)} slugs for {asset}...", flush=True)
    for slug in slugs:
        if slug in _market_id_cache:
            print(f"  [cache warm] {_fmt_slug(slug):<15} already cached ✓", flush=True)
            continue
        # Single attempt only — no retries during warm
        market_id, err = import_fast_market_market(slug, max_retries=1)
        if market_id:
            print(f"  [cache warm] {_fmt_slug(slug):<15} imported ✓", flush=True)
        else:
            print(f"  [cache warm] {_fmt_slug(slug):<15} skipped: {err}", flush=True)


def run_redeemer():
    """Check for redeemable positions and redeem them."""
    try:
        positions = get_client().get_positions()
        from dataclasses import asdict
        for p in positions:
            pos = asdict(p)
            if pos.get("redeemable"):
                side = "yes" if pos.get("shares_yes", 0) > 0 else "no"
                result = get_client().redeem(
                    market_id=pos["market_id"],
                    side=side
                )
                if result.get("success"):
                    print(f"Redeemed {pos['question'][:40]} {side.upper()}")
    except Exception as e:
        print(f"Redeemer error: {e}")


def backfill_trade_outcomes():
    """Fetch trade history from Polymarket Data API and update local trade_log.json."""
    try:
        poly_trades = get_client().get_trade_history_index(limit=200)
    except Exception as e:
        print(f"  Backfill fetch error: {e}", flush=True)
        return

    if not _TRADE_LOG_FILE.exists():
        return

    local_trades = json.loads(_TRADE_LOG_FILE.read_text())
    updated = 0

    # Load tg_message_log once for thread lookup
    _tg_msg_lookup = {}
    try:
        from signal_research import get_tg_entry_msg, init_db
        db_path  = os.environ.get("DB_PATH", "/data/signal_research.db")
        _bf_conn = init_db(db_path)
        for trade in local_trades:
            tid = trade.get("trade_id")
            if tid:
                _tg_msg_lookup[tid] = get_tg_entry_msg(_bf_conn, tid)
        _bf_conn.close()
    except Exception:
        pass

    for trade in local_trades:
        tid = trade.get("trade_id")
        if tid and tid in poly_trades and trade.get("outcome") is None:
            pt = poly_trades[tid]
            if pt.get("pnl") is not None:
                trade["pnl"]     = round(pt["pnl"], 4)
                trade["outcome"] = "win" if trade["pnl"] > 0 else "loss"
                updated += 1
                _send_telegram_outcome(
                    token=cfg.get("telegram_bot_token", ""),
                    chat_id=cfg.get("telegram_chat_id", ""),
                    trade=trade,
                    reply_to_message_id=_tg_msg_lookup.get(tid),
                )

    if updated:
        _TRADE_LOG_FILE.write_text(json.dumps(local_trades, indent=2))
        print(f"  Backfilled {updated} trade outcomes", flush=True)
# =============================================================================
# API Helpers
# =============================================================================

_client = None


def get_client():
    global _client
    if _client is None:
        try:
            from polymarket_sdk import get_poly_client
        except ImportError as e:
            print(f"Error: polymarket_sdk not found. {e}")
            sys.exit(1)
        _client = get_poly_client()
    return _client


def _api_request(url, method="GET", data=None, headers=None, timeout=15):
    try:
        req_headers = headers or {}
        if "User-Agent" not in req_headers:
            req_headers["User-Agent"] = "simmer-pknwitq/1.0"
        body = None
        if data:
            body = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        req = Request(url, data=body, headers=req_headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        try:
            error_body = json.loads(e.read().decode("utf-8"))
            return {"error": error_body.get("detail", str(e)), "status_code": e.code}
        except Exception:
            return {"error": str(e), "status_code": e.code}
    except URLError as e:
        return {"error": f"Connection error: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# CEX Price Signals (local, not imported from signal_research to keep clean)
# =============================================================================

def get_binance_momentum(symbol="BTCUSDT", lookback_minutes=5):
    url = (
        f"https://api.binance.com/api/v3/klines" 
        # f"https://api.binance.us/api/v3/klines",
        f"?symbol={symbol}&interval=1m&limit={lookback_minutes}"
    )
    result = _api_request(url)
    if not result or isinstance(result, dict):
        return None
    try:
        candles = result
        if len(candles) < 2:
            return None
        price_then    = float(candles[0][1])
        price_now     = float(candles[-1][4])
        momentum_pct  = ((price_now - price_then) / price_then) * 100
        volumes       = [float(c[5]) for c in candles]
        avg_volume    = sum(volumes) / len(volumes)
        latest_volume = volumes[-1]
        volume_ratio  = latest_volume / avg_volume if avg_volume > 0 else 1.0
        return {
            "momentum_pct":  momentum_pct,
            "direction":     "up" if momentum_pct > 0 else "down",
            "price_now":     price_now,
            "price_then":    price_then,
            "avg_volume":    avg_volume,
            "latest_volume": latest_volume,
            "volume_ratio":  volume_ratio,
            "candles":       len(candles),
        }
    except (IndexError, ValueError, KeyError):
        return None


def get_coingecko_momentum(asset="bitcoin"):
    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={asset}&vs_currencies=usd"
    )
    result = _api_request(url)
    if not result or (isinstance(result, dict) and result.get("error")):
        return None
    price_now = result.get(asset, {}).get("usd")
    if not price_now:
        return None
    return {
        "momentum_pct":  0,
        "direction":     "neutral",
        "price_now":     price_now,
        "price_then":    price_now,
        "avg_volume":    0,
        "latest_volume": 0,
        "volume_ratio":  1.0,
        "candles":       0,
    }


def get_momentum(asset="BTC", source="binance", lookback=5):
    if source == "binance":
        return get_binance_momentum(ASSET_SYMBOLS.get(asset, "BTCUSDT"), lookback)
    elif source == "coingecko":
        return get_coingecko_momentum(COINGECKO_ASSETS.get(asset, "bitcoin"))
    return None


# =============================================================================
# Polymarket API: Import & Trade
# =============================================================================

def import_fast_market_market(slug, max_retries=1):
    """
    Resolve a Polymarket market slug to a condition_id via polymarket_sdk.

    Returns (market_id, None) on success or (None, error_str) on failure.
    Cache hits are returned immediately without an API call.

    Retries up to max_retries times on rate errors with exponential backoff:
      attempt 1: wait 10s
      attempt 2: wait 20s
      attempt 3: wait 40s
    """
    global _market_id_cache

    # Return from cache immediately — no quota consumed
    if slug in _market_id_cache:
        return _market_id_cache[slug], None

    # Guard against exceeding the daily import limit
    imports_used = _get_import_count_today()
    # if imports_used >= IMPORT_DAILY_LIMIT:
    #     return None, f"Daily import limit reached ({imports_used}/{IMPORT_DAILY_LIMIT}) — waiting for tomorrow"

    print(f"  Import quota: {imports_used}/{IMPORT_DAILY_LIMIT} used today", flush=True)

    url = f"https://polymarket.com/event/{slug}"

    for attempt in range(max_retries):
        try:
            result = get_client().import_market(url)
        except Exception as e:
            err = str(e)
            if "429" in err:
                wait = 10 * (2 ** attempt)   # 10s, 20s, 40s
                print(
                    f"  Rate limited (attempt {attempt+1}/{max_retries}) "
                    f"— waiting {wait}s before retry...",
                    flush=True,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
                # All retries exhausted
                print(f"  Rate limited on all {max_retries} attempts — skipping this cycle", flush=True)
                return None, "429_rate_limited"
            return None, err

        # Extract the Simmer-specific market_id from the response.
        # The SDK may return a dict or an object depending on SDK version.
        if isinstance(result, dict):
            market_id = result.get("market_id") or result.get("id")
        else:
            market_id = getattr(result, "market_id", None) or getattr(result, "id", None)

        if not market_id:
            return None, f"No market_id in import response: {result}"

        # Cache so future cycles skip the quota entirely
        _market_id_cache[slug] = market_id
        _save_market_cache(_market_id_cache)
        _increment_import_count()

        return market_id, None

    return None, "429_rate_limited"


def get_positions():
    try:
        positions = get_client().get_positions()
        from dataclasses import asdict
        return [asdict(p) for p in positions]
    except Exception:
        return []



def get_portfolio():
    try:
        return get_client().get_portfolio()
    except Exception as e:
        return {"error": str(e)}


def execute_trade(market_id, side, amount):
    try:
        result = get_client().trade(
            market_id=market_id,
            side=side,
            amount=amount,
            source=TRADE_SOURCE,
        )
        return {
            "success":       result.success,
            "trade_id":      result.trade_id,
            "shares_bought": result.shares_bought,
            "shares":        result.shares_bought,
            "error":         result.error,
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_position_size(max_size, smart_sizing=False):
    if not smart_sizing:
        return max_size
    portfolio = get_portfolio()
    if not portfolio or portfolio.get("error"):
        return max_size
    balance = portfolio.get("balance_usdc", 0)
    if balance <= 0:
        return max_size
    return min(balance * SMART_SIZING_PCT, max_size)


# =============================================================================
# Main Strategy Logic
# =============================================================================

def run_fast_market_strategy(
    dry_run=True,
    positions_only=False,
    show_config=False,
    smart_sizing=False,
    quiet=False,
):
    # ── Reload config each cycle so optimizer changes take effect immediately ─
    global cfg, raw_cfg
    global MIN_MOMENTUM_PCT, MAX_POSITION_USD, SIGNAL_SOURCE
    global MIN_TIME_REMAINING, MAX_TIME_REMAINING, ASSET
    global WINDOW, VOLUME_CONFIDENCE, DAILY_BUDGET
    global MIN_SCORE_TO_IMPORT, IMPORT_DAILY_LIMIT, MIN_ENTRY_PRICE, MIN_LIQUIDITY_RATIO
    cfg = _load_config(CONFIG_SCHEMA, __file__)
    try:
        with open(_get_config_path(__file__)) as _f:
            raw_cfg = json.load(_f)
    except Exception:
        raw_cfg = cfg
    MIN_MOMENTUM_PCT    = cfg["min_momentum_pct"]
    MAX_POSITION_USD    = cfg["max_position"]
    SIGNAL_SOURCE       = cfg["signal_source"]
    MIN_TIME_REMAINING  = cfg["min_time_remaining"]
    MAX_TIME_REMAINING  = cfg.get("max_time_remaining", 420)
    ASSET               = cfg["asset"].upper()
    WINDOW              = cfg["window"]
    VOLUME_CONFIDENCE   = cfg["volume_confidence"]
    DAILY_BUDGET        = cfg["daily_budget"]
    MIN_SCORE_TO_IMPORT = raw_cfg.get("min_score_to_import", 0.65)
    IMPORT_DAILY_LIMIT  = raw_cfg.get("daily_import_limit", 1000)
    MIN_ENTRY_PRICE     = raw_cfg.get("min_entry_price", 0.35)
    MIN_LIQUIDITY_RATIO = raw_cfg.get("min_liquidity_ratio", 0.20)

    now_str  = datetime.now(timezone.utc).strftime("%H:%M:%S")
    mode_tag = "[DRY]" if dry_run else "[LIVE]"

    def log(msg, force=False):
        if not quiet or force:
            print(msg, flush=True)

    # ── Kill switch ───────────────────────────────────────────────────────────
    if not dry_run and _is_kill_switch_active():
        log(f"{mode_tag} {now_str} | KILL SWITCH ACTIVE — trading halted this cycle", force=True)
        return

    # ── Consecutive loss cooldown ─────────────────────────────────────────────
    # After N resolved losses in a row, skip trading for one cycle to avoid
    # feeding losses during an adverse or noisy market environment.
    if raw_cfg.get("loss_cooldown_enabled", True):
        _streak = int(raw_cfg.get("loss_cooldown_streak", 3))
        if _TRADE_LOG_FILE.exists():
            try:
                _all_trades = json.loads(_TRADE_LOG_FILE.read_text())
                _resolved = [t for t in _all_trades if t.get("outcome") in ("win", "loss")]
                if len(_resolved) >= _streak:
                    _recent = [t["outcome"] for t in _resolved[-_streak:]]
                    if all(o == "loss" for o in _recent):
                        log(
                            f"{mode_tag} {now_str} | COOLDOWN: last {_streak} resolved trades "
                            f"all losses — skipping cycle to protect capital",
                            force=True,
                        )
                        return
            except Exception:
                pass

    # ── Config view ───────────────────────────────────────────────────────────
    if show_config:
        daily_spend = _load_daily_spend()
        print(f"Config: {_get_config_path(__file__)}")
        print(
            f"  asset={ASSET}  window={WINDOW}  max_pos=${MAX_POSITION_USD:.2f}  "
            f"budget=${DAILY_BUDGET:.2f} (${daily_spend['spent']:.2f} spent)"
        )
        return

    # Validate API key early
    get_client()

    # ── Positions view ────────────────────────────────────────────────────────
    if positions_only:
        positions = get_positions()
        fast_positions = [
            p for p in positions
            if "up or down" in (p.get("question", "") or "").lower()
        ]
        if not fast_positions:
            print("  No open fast market positions")
        else:
            for pos in fast_positions:
                print(
                    f"  {pos.get('question','')[:55]:55} "
                    f"YES={pos.get('shares_yes', 0):.1f} "
                    f"NO={pos.get('shares_no', 0):.1f} "
                    f"P&L=${pos.get('pnl', 0):.2f}"
                )
        return

    daily_spend = _load_daily_spend()

    # ── Step 1: Discover markets ──────────────────────────────────────────────
    markets = discover_fast_market_markets(ASSET, WINDOW)
    if not markets:
        log(f"{mode_tag} {now_str} | no active {ASSET} markets")
        return

    # ── Step 2: Pick best market ──────────────────────────────────────────────
    best = find_best_fast_market(markets, min_time_remaining=MIN_TIME_REMAINING)
    if not best:
        log(f"{mode_tag} {now_str} | no market with >{MIN_TIME_REMAINING}s left")
        return

    end_time  = best.get("end_time")
    remaining = (end_time - datetime.now(timezone.utc)).total_seconds() if end_time else 0
    slug_short = _fmt_slug(best.get("slug", ""))

    # Parse YES price; fall back to 0.5 only on genuine parse failure
    try:
        prices = json.loads(best.get("outcome_prices", "[]"))
        market_yes_price = float(prices[0]) if prices else 0.5
        if not (0 < market_yes_price < 1):
            market_yes_price = 0.5
    except (json.JSONDecodeError, IndexError, ValueError):
        market_yes_price = 0.5

    fee_rate_bps = best.get("fee_rate_bps", 0)
    fee_rate     = fee_rate_bps / 10000

    # ── Step 3: Collect signals ───────────────────────────────────────────────
    # Done BEFORE the time guard so vs_ref is always shown in the SKIP log.
    # CEX klines are served from the 15s in-process cache populated by the
    # collector 5s earlier — no extra API calls.
    symbol = ASSET_SYMBOLS.get(ASSET, "BTCUSDT")
    cex_signals = extract_cex_signals(symbol)
    if not cex_signals:
        log(f"{mode_tag} {now_str} | {slug_short} {remaining:.0f}s | ERROR: CEX fetch failed", force=True)
        return

    poly_signals = extract_poly_signals(best)

    now_utc     = datetime.now(timezone.utc)
    market_slug = best.get("slug", "")

    # Determine if the window just opened. Use event_start if available,
    # otherwise fall back to: window is "just opened" if no cached reference
    # price exists yet for this slug. This fixes the 47% vs_ref=0 rate caused
    # by event_start being None in the Gamma API response.
    event_start = best.get("event_start")
    if event_start is not None:
        window_just_opened = (now_utc - event_start).total_seconds() < 60
    else:
        # No event_start from API -- treat as newly opened if not yet cached
        window_just_opened = market_slug not in _load_ref_cache()

    # Use Poly own priceToBeat if populated -- more accurate than our Binance snapshot.
    # Falls back to our window_refs cache when Poly does not populate it (the common case).
    poly_price_to_beat = best.get("price_to_beat")
    if poly_price_to_beat:
        try:
            price_to_beat = float(poly_price_to_beat)
        except (ValueError, TypeError):
            price_to_beat = get_window_reference_price(
                market_slug,
                cex_price_now=cex_signals.get("price_now"),
                window_open=window_just_opened,
            )
    else:
        price_to_beat = get_window_reference_price(
            market_slug,
            cex_price_now=cex_signals.get("price_now"),
            window_open=window_just_opened,
        )
    if price_to_beat and cex_signals.get("price_now"):
        cex_signals["btc_vs_reference"] = (
            (cex_signals["price_now"] - price_to_beat) / price_to_beat * 100
        )

    vs_ref     = cex_signals.get("btc_vs_reference")
    vs_ref_str = f"{vs_ref:+.4f}%" if vs_ref is not None else "n/a"
    m5         = cex_signals.get("momentum_5m", 0) or 0
    vol_r      = cex_signals.get("volume_ratio", 1.0) or 1.0
    poly_p     = poly_signals.get("poly_yes_price", 0.5) or 0.5
    cex_lag_val = _calc_cex_poly_lag(cex_signals, poly_signals) or 0.0
    cex_lag_str = f"{cex_lag_val:+.3f}"

    # ── Time guard (after signals so vs_ref shows the real value) ─────────────
    if remaining > MAX_TIME_REMAINING:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:.0f}s | "
            f"m5={m5:+.3f}% vs_ref={vs_ref_str} poly={poly_p:.3f} vol={vol_r:.2f}x | "
            f"SKIP: too early ({remaining:.0f}s > {MAX_TIME_REMAINING}s max)"
        )
        return

    # Block trade if dominant signal (btc_vs_reference, weight=0.315) is missing.
    # A score built without it is unreliable -- better to skip than trade blind.
    if vs_ref is None:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"m5={m5:+.3f}% vs_ref=UNSEEDED poly={poly_p:.3f} lag={cex_lag_str} vol={vol_r:.2f}x | "
            f"score=n/a BLOCK: btc_vs_reference not seeded"
        )
        return

    # ── Step 4: Composite signal ──────────────────────────────────────────────
    signal     = get_composite_signal(cex_signals, poly_signals, config=raw_cfg)
    score      = signal["score"]
    confidence = signal["confidence"]
    regime       = signal.get("regime", "normal")
    hour_acc     = signal.get("hour_accuracy", 0.0)
    hour_acc_str = f"{hour_acc:.0%}"
    session      = signal.get("session", "?")

    if not signal["should_trade"]:
        reason = signal.get("filter_reason", "neutral band")
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"[{regime}/{session}/{hour_acc_str}] m5={m5:+.3f}% vs_ref={vs_ref_str} poly={poly_p:.3f} "
            f"lag={cex_lag_str} vol={vol_r:.2f}x | "
            f"score={score:.3f} BLOCK: {reason}"
        )
        return

    side         = signal["side"]
    position_pct = signal["position_pct"]

    # ── YES confidence gate ───────────────────────────────────────────────────
    # min_yes_conf: minimum confidence required to enter a YES trade.
    # Confidence = how far score is from 0.5 (0 = coin-flip, 1 = max certainty).
    # Protects against low-conviction YES trades in noisy markets.
    min_yes_conf = raw_cfg.get("min_yes_conf", 0.0)
    if side == "yes" and min_yes_conf > 0 and confidence < min_yes_conf:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"score={score:.3f} conf={confidence:.3f} → YES BLOCK: "
            f"confidence {confidence:.3f} < min_yes_conf {min_yes_conf:.3f}"
        )
        return

    # ── NO side score cap ─────────────────────────────────────────────────────
    MAX_NO_SCORE = cfg.get("max_no_score", 0.284)
    if side == "no" and score > MAX_NO_SCORE:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"score={score:.3f} → NO BLOCK: score {score:.3f} > max_no_score {MAX_NO_SCORE:.3f}"
        )
        return

    # ── Slow-market entry price tightening ───────────────────────────────────
    # In a quiet market, the polymarket price near 0.50 gives better payout
    # ratios (you earn more per dollar if the rare move materialises). Block
    # entries where the market has already drifted far from 0.50 in a slow
    # regime — the edge is gone and the payout is poor.
    if regime == "slow":
        slow_max_entry_yes = raw_cfg.get("slow_max_entry_yes", 0.54)
        slow_min_entry_no  = raw_cfg.get("slow_min_entry_no",  0.46)
        if side == "yes" and market_yes_price > slow_max_entry_yes:
            log(
                f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
                f"[slow] score={score:.3f} → YES BLOCK: market already at "
                f"{market_yes_price:.3f} > {slow_max_entry_yes:.3f} slow-market cap"
            )
            return
        if side == "no" and market_yes_price < slow_min_entry_no:
            log(
                f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
                f"[slow] score={score:.3f} → NO BLOCK: market already at "
                f"{market_yes_price:.3f} < {slow_min_entry_no:.3f} slow-market floor"
            )
            return

    # Hard liquidity floor — FAK orders fail on thin books regardless of signal.
    # Skip sentinels: vol_r <= 0.05 means new candle with sub-second volume accumulated
    # (not a thin book); vol_r == 1.0 is the fallback when avg_vol has not computed yet.
    if vol_r > 0.05 and vol_r != 1.0 and vol_r < MIN_LIQUIDITY_RATIO:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"m5={m5:+.3f}% vs_ref={vs_ref_str} poly={poly_p:.3f} vol={vol_r:.2f}x | "
            f"score={score:.3f} BLOCK: insufficient liquidity ({vol_r:.2f}x < {MIN_LIQUIDITY_RATIO}x min)"
        )
        return

    # ── Fee-aware EV check ────────────────────────────────────────────────────
    entry_price = market_yes_price if side == "yes" else (1 - market_yes_price)

    if fee_rate > 0:
        win_profit   = (1 - entry_price) * (1 - fee_rate)
        breakeven_wr = entry_price / (win_profit + entry_price)
        implied_wr   = score if side == "yes" else (1 - score)
        # Tighter edge requirement in slow markets: score must beat breakeven by 0.05
        min_edge = 0.05 if regime == "slow" else 0.03
        if implied_wr < breakeven_wr + min_edge:
            log(
                f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
                f"[{regime}] score={score:.3f} → {side.upper()} BLOCK: fee EV too thin "
                f"(implied={implied_wr:.1%} < breakeven={breakeven_wr:.1%} + {min_edge:.0%})"
            )
            return

    # ── Payout ratio gate — wins must structurally exceed losses ─────────────
    # payout_ratio = amount won per dollar if correct / dollar lost if wrong
    #   = (1 - entry_price) / entry_price
    # Buying at 0.55 → ratio 0.82x  (win earns LESS than the loss — negative)
    # Buying at 0.50 → ratio 1.00x  (breakeven ignoring win rate)
    # Buying at 0.43 → ratio 1.33x  (each win covers 1.33 losses)
    # Minimum required payout ratio is configurable; default = 1.0 (at least
    # break-even in payout before win-rate edge kicks in).
    payout_ratio = (1 - entry_price) / entry_price if entry_price > 0 else 0
    # Slow markets get a stricter requirement: need payout to cover more losses
    # since lower signal reliability means fewer wins per loss.
    if regime == "slow":
        min_payout = raw_cfg.get("slow_min_payout_ratio", 1.10)
    else:
        min_payout = raw_cfg.get("min_payout_ratio", 0.90)
    if payout_ratio < min_payout:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"[{regime}] score={score:.3f} → {side.upper()} BLOCK: "
            f"payout ratio {payout_ratio:.2f}x < {min_payout:.2f}x min "
            f"(entry={entry_price:.3f})"
        )
        return

    # ── Position sizing ───────────────────────────────────────────────────────
    position_size    = calculate_position_size(MAX_POSITION_USD * position_pct, smart_sizing)
    # max_no_size: hard cap on NO-side trade size to limit downside on contrarian bets
    if side == "no":
        max_no_size = raw_cfg.get("max_no_size", MAX_POSITION_USD)
        position_size = min(position_size, max_no_size)
    remaining_budget = DAILY_BUDGET - daily_spend["spent"]

    if remaining_budget <= 0:
        log(
            f"{mode_tag} {now_str} | {slug_short} | BLOCK: daily budget exhausted "
            f"(${daily_spend['spent']:.2f}/${DAILY_BUDGET:.2f})",
            force=True,
        )
        return

    # ── Per-window cap: check BEFORE sizing/bump so position_size is unmodified ─
    slug           = best["slug"]
    window_key     = f"window_{slug}"
    window_spent   = daily_spend.get(window_key, 0.0)
    max_per_window = cfg.get("max_position_per_window", MAX_POSITION_USD)
    if window_spent + position_size > max_per_window:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"score={score:.3f} → {side.upper()} BLOCK: window cap "
            f"(${window_spent:.2f} spent + ${position_size:.2f} proposed > ${max_per_window:.2f} limit)"
        )
        return

    position_size = min(position_size, remaining_budget)

    if position_size < 0.50:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"score={score:.3f} → {side.upper()} BLOCK: size ${position_size:.2f} < $0.50 min"
        )
        return

    if entry_price <= 0:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"BLOCK: invalid entry_price={entry_price:.4f}",
            force=True,
        )
        return

    # Never buy shares priced below 35¢ — market has already priced in the move
    if entry_price < MIN_ENTRY_PRICE:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"score={score:.3f} → {side.upper()} BLOCK: "
            f"entry ${entry_price:.2f} < ${MIN_ENTRY_PRICE} min (market already priced in)"
        )
        return

    best_ask_raw = best.get("bestAsk")
    best_bid_raw = best.get("bestBid")
    if best_ask_raw and best_bid_raw:
        try:
            yes_ask = float(best_ask_raw)
            yes_bid = float(best_bid_raw)
            # NO ask = 1 - YES bid (complementary market)
            ask_price = yes_ask if side == "yes" else (1 - yes_bid)
            ask_price = max(ask_price, entry_price)  # ask always >= mid
        except (ValueError, TypeError):
            ask_price = entry_price / 0.90
    else:
        ask_price = entry_price / 0.90  # fallback: assume 10% spread

    min_order_usdc = MIN_SHARES_PER_ORDER * ask_price  # 5 shares at ask price

    if position_size < min_order_usdc:
        if remaining_budget >= min_order_usdc and min_order_usdc <= MAX_POSITION_USD:
            calc_size     = position_size  # save before bump for log
            position_size = min_order_usdc  # bump up to meet minimum
            log(
                f"  SIZE BUMP: ${calc_size:.2f} → ${min_order_usdc:.2f} "
                f"(min {MIN_SHARES_PER_ORDER:.0f} shares @ ${ask_price:.3f}; "
                f"max_pos=${MAX_POSITION_USD:.2f})"
            )
        else:
            reason = []
            if remaining_budget < min_order_usdc:
                reason.append(
                    f"insufficient budget (${remaining_budget:.2f} < ${min_order_usdc:.2f})"
                )
            if min_order_usdc > MAX_POSITION_USD:
                reason.append(
                    f"exceeds max position (${min_order_usdc:.2f} > ${MAX_POSITION_USD:.2f})"
                )

            log(
                f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
                f"score={score:.3f} → {side.upper()} BLOCK: "
                f"min {MIN_SHARES_PER_ORDER} shares = ${min_order_usdc:.2f} | "
                + " & ".join(reason)
            )
            return

    # ── Step 5: Import & execute ──────────────────────────────────────────────
    log(
        f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
        f"[{regime}/{session}/{hour_acc_str}] m5={m5:+.3f}% vs_ref={vs_ref_str} poly={poly_p:.3f} "
        f"lag={cex_lag_str} vol={vol_r:.2f}x payout={payout_ratio:.2f}x | "
        f"score={score:.3f} conf={confidence:.2f} → {side.upper()} ${position_size:.2f}"
    )

    # Gate: only consume an import slot on strong signals
    if slug not in _market_id_cache:
        if score < MIN_SCORE_TO_IMPORT and score > (1 - MIN_SCORE_TO_IMPORT):
            log(f"  SKIP import: score {score:.3f} too weak to spend import quota")
            return

    market_id, import_error = import_fast_market_market(slug)
    if not market_id:
        if import_error != "429_rate_limited":
            log(f"  ERROR import failed: {import_error}", force=True)
        return

    # ── Trade execution ───────────────────────────────────────────────────────
    if dry_run:
        est_shares = position_size / entry_price
        log(
            f"  [DRY RUN] Would buy {side.upper()} ${position_size:.2f} "
            f"(~{est_shares:.1f} shares @ ${entry_price:.3f})"
        )
    else:
        result = execute_trade(market_id, side, position_size)

        # Recover from SDK token-cache miss on container restart:
        # The SDK loses its internal token_id mapping on restart even though our
        # market_id_cache.json still has the slug, so import_market is skipped
        # and the trade fails.  Fix: evict + re-import + retry once.
        if result and "No token_id" in (result.get("error") or ""):
            log(f"  SDK token cache miss — evicting slug and re-importing...", force=True)
            _market_id_cache.pop(slug, None)
            _save_market_cache(_market_id_cache)
            market_id2, import_err2 = import_fast_market_market(slug)
            if market_id2:
                result = execute_trade(market_id2, side, position_size)
            else:
                log(f"  ERROR re-import failed: {import_err2}", force=True)
                return

        if result and result.get("success"):
            shares   = result.get("shares_bought") or result.get("shares") or 0
            # CLOB FOK responses sometimes return makingAmount=0 even on a fill;
            # estimate from spend / entry_price as a fallback so logs are meaningful.
            if shares == 0 and entry_price > 0:
                shares = round(position_size / entry_price, 1)
            trade_id = result.get("trade_id")
            log(f"  TRADED {shares:.1f} {side.upper()} shares @ ${entry_price:.3f}", force=True)
            daily_spend["spent"]              += position_size
            daily_spend["trades"]             += 1
            daily_spend[window_key]            = window_spent + position_size
            _save_daily_spend(daily_spend)

            if trade_id:
                try:
                    _log_trade_local(
                        trade_id=trade_id,
                        side=side,
                        score=score,
                        confidence=confidence,
                        entry_price=entry_price,
                        position_size=position_size,
                        shares=shares,
                        slug=best["slug"],
                        remaining=remaining,
                        cex_signals=cex_signals,
                        poly_signals=poly_signals,
                        lag=cex_lag_val,
                    )
                except Exception as e:
                    log(f"  WARNING: trade log failed: {e}", force=True)
            _send_webhook(
                url=cfg.get("webhook_url", ""),
                slug=best["slug"],
                side=side,
                score=score,
                position_size=position_size,
                shares=shares,
                entry_price=entry_price,
            )
            tg_msg_id = _send_telegram(
                token=cfg.get("telegram_bot_token", ""),
                chat_id=cfg.get("telegram_chat_id", ""),
                slug=best["slug"],
                side=side,
                score=score,
                position_size=position_size,
                shares=shares,
                entry_price=entry_price,
                remaining=remaining,
                lag=cex_lag_val,
            )
            if tg_msg_id and trade_id:
                try:
                    from signal_research import store_tg_entry_msg, init_db
                    db_path = os.environ.get("DB_PATH", "/data/signal_research.db")
                    _tg_conn = init_db(db_path)
                    store_tg_entry_msg(_tg_conn, trade_id, tg_msg_id)
                    _tg_conn.close()
                except Exception as _tg_db_err:
                    print(f"[telegram] failed to store msg_id: {_tg_db_err}", flush=True)
        else:
            error = result.get("error", "Unknown") if result else "no response"
            log(f"  ERROR trade failed: {error}", force=True)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simmer Pknwitq Trading Skill")
    parser.add_argument("--live",         action="store_true", help="Execute real trades")
    parser.add_argument("--dry-run",      action="store_true", help="Show opportunities without trading (default)")
    parser.add_argument("--positions",    action="store_true", help="Show current positions")
    parser.add_argument("--config",       action="store_true", help="Show current config")
    parser.add_argument("--set",          action="append", metavar="KEY=VALUE", help="Update config value")
    parser.add_argument("--smart-sizing", action="store_true", help="Portfolio-based position sizing")
    parser.add_argument("--quiet", "-q",  action="store_true", help="Only output on trades/errors")
    args = parser.parse_args()

    if args.set:
        updates = {}
        for item in args.set:
            if "=" not in item:
                print(f"Invalid --set format: {item}. Use KEY=VALUE")
                sys.exit(1)
            key, val = item.split("=", 1)
            if key not in CONFIG_SCHEMA:
                print(f"Unknown config key: {key}. Valid: {', '.join(CONFIG_SCHEMA)}")
                sys.exit(1)
            type_fn = CONFIG_SCHEMA[key].get("type", str)
            try:
                updates[key] = (
                    val.lower() in ("true", "1", "yes")
                    if type_fn is bool
                    else type_fn(val)
                )
            except ValueError:
                print(f"Invalid value for {key}: {val}")
                sys.exit(1)
        print(f"✅ Config updated: {json.dumps(_update_config(updates, __file__))}")
        sys.exit(0)

    run_fast_market_strategy(
        dry_run=not args.live,
        positions_only=args.positions,
        show_config=args.config,
        smart_sizing=args.smart_sizing,
        quiet=args.quiet,
    )
