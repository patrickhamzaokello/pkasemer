#!/usr/bin/env python3
"""
Simmer FastLoop Trading Skill

Trades Polymarket BTC 5-minute fast markets using CEX price momentum.
Default signal: Binance BTCUSDT candles. Agents can customize signal source.

Usage:
    python fast_trader.py              # Dry run (show opportunities, no trades)
    python fast_trader.py --live       # Execute real trades
    python fast_trader.py --positions  # Show current fast market positions
    python fast_trader.py --quiet      # Only output on trades/errors

Requires:
    SIMMER_API_KEY environment variable (get from simmer.markets/dashboard)

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
import argparse
import time
from datetime import datetime, timezone, timedelta  # timedelta kept for future use
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from pathlib import Path

from composite_signal import get_composite_signal
from signal_research import extract_cex_signals, extract_poly_signals, get_window_reference_price
from market_utils import discover_fast_market_markets, find_best_fast_market

# Force line-buffered stdout for non-TTY environments (cron, Docker, OpenClaw)
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv()



# =============================================================================
# Constants
# =============================================================================

TRADE_SOURCE = "sdk:fastloop"
SMART_SIZING_PCT = 0.05
MIN_SHARES_PER_ORDER = 5
MIN_SCORE_TO_IMPORT = 0.65
IMPORT_DAILY_LIMIT = 50           # Leave 1 buffer from the 10/day free tier
MIN_ENTRY_PRICE      = 0.35     # skip if market already priced in the move


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
    "entry_threshold":    {"default": 0.05,      "env": "SIMMER_SPRINT_ENTRY",        "type": float},
    "min_momentum_pct":   {"default": 0.15,      "env": "SIMMER_SPRINT_MOMENTUM",     "type": float},
    "max_position":       {"default": 5.0,       "env": "SIMMER_SPRINT_MAX_POSITION", "type": float},
    "signal_source":      {"default": "binance", "env": "SIMMER_SPRINT_SIGNAL",       "type": str},
    "lookback_minutes":   {"default": 5,         "env": "SIMMER_SPRINT_LOOKBACK",     "type": int},
    "min_time_remaining": {"default": 30,        "env": "SIMMER_SPRINT_MIN_TIME",     "type": int},
    "asset":              {"default": "BTC",     "env": "SIMMER_SPRINT_ASSET",        "type": str},
    "window":             {"default": "5m",      "env": "SIMMER_SPRINT_WINDOW",       "type": str},
    "volume_confidence":  {"default": True,      "env": "SIMMER_SPRINT_VOL_CONF",     "type": bool},
    "daily_budget":       {"default": 10.0,      "env": "SIMMER_SPRINT_DAILY_BUDGET", "type": float},
    "composite_threshold":{"default": 0.60,      "env": "SIMMER_SPRINT_COMP_THRESH",  "type": float},
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
ENTRY_THRESHOLD    = cfg["entry_threshold"]
MIN_MOMENTUM_PCT   = cfg["min_momentum_pct"]
MAX_POSITION_USD   = cfg["max_position"]
SIGNAL_SOURCE      = cfg["signal_source"]
LOOKBACK_MINUTES   = cfg["lookback_minutes"]
MIN_TIME_REMAINING = cfg["min_time_remaining"]
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


def warm_import_cache(asset="BTC"):
    """
    Pre-import the current and next 5m market slugs at startup.
    Called once when the scheduler starts so that live trade cycles
    always find a cached market_id and never hit the import endpoint
    mid-signal.

    Uses the same import quota tracking — skips if limit already reached.
    Logs clearly so you can see what was cached vs skipped.
    """
    from market_utils import get_fast_market_slugs
    slugs = get_fast_market_slugs(asset, include_next=True)
    print(f"  [cache warm] checking {len(slugs)} slugs for {asset}...", flush=True)
    for slug in slugs:
        if slug in _market_id_cache:
            print(f"  [cache warm] {slug[-19:]} already cached ✓", flush=True)
            continue
        market_id, err = import_fast_market_market(slug)
        if market_id:
            print(f"  [cache warm] {slug[-19:]} imported → {market_id[:8]}... ✓", flush=True)
        elif err == "429_rate_limited":
            print(f"  [cache warm] {slug[-19:]} rate limited — will cache on first signal", flush=True)
        else:
            print(f"  [cache warm] {slug[-19:]} skipped: {err}", flush=True)
        time.sleep(12)  # 6/min limit = 1 per 10s; 12s gives comfortable headroom


# =============================================================================
# API Helpers
# =============================================================================

_client = None


def get_client():
    global _client
    if _client is None:
        try:
            from simmer_sdk import SimmerClient
        except ImportError:
            print("Error: simmer-sdk not installed. Run: pip install simmer-sdk")
            sys.exit(1)
        api_key = os.environ.get("SIMMER_API_KEY")
        if not api_key:
            print("Error: SIMMER_API_KEY environment variable not set")
            sys.exit(1)
        _client = SimmerClient(api_key=api_key, venue="polymarket")
    return _client


def _api_request(url, method="GET", data=None, headers=None, timeout=15):
    try:
        req_headers = headers or {}
        if "User-Agent" not in req_headers:
            req_headers["User-Agent"] = "simmer-fastloop/1.0"
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
# Simmer API: Import & Trade
# =============================================================================

def import_fast_market_market(slug, max_retries=3):
    """
    Import a Polymarket market via Simmer SDK.

    Returns (market_id, None) on success or (None, error_str) on failure.
    Cache hits are returned immediately without consuming the import quota.

    Retries up to max_retries times on 429 with exponential backoff:
      attempt 1: wait 10s
      attempt 2: wait 20s
      attempt 3: wait 40s
    After all retries exhausted returns (None, "429_rate_limited") so the
    caller can log it as a soft skip rather than a hard error.
    """
    global _market_id_cache

    # Return from cache immediately — no quota consumed
    if slug in _market_id_cache:
        return _market_id_cache[slug], None

    # Guard against exceeding the daily import limit
    imports_used = _get_import_count_today()
    if imports_used >= IMPORT_DAILY_LIMIT:
        return None, f"Daily import limit reached ({imports_used}/{IMPORT_DAILY_LIMIT}) — waiting for tomorrow"

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
    now_str  = datetime.now(timezone.utc).strftime("%H:%M:%S")
    mode_tag = "[DRY]" if dry_run else "[LIVE]"

    def log(msg, force=False):
        if not quiet or force:
            print(msg, flush=True)

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
    slug_short = best.get("slug", "")[-19:]

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
    symbol = ASSET_SYMBOLS.get(ASSET, "BTCUSDT")
    cex_signals = extract_cex_signals(symbol)
    if not cex_signals:
        log(f"{mode_tag} {now_str} | {slug_short} {remaining:.0f}s | ERROR: CEX fetch failed", force=True)
        return

    poly_signals = extract_poly_signals(best)

    # Only seed the reference price in the first ~60s after window opens.
    # If seeded on every cycle, a restart mid-window resets btc_vs_reference to 0.
    event_start = best.get("event_start")
    now_utc = datetime.now(timezone.utc)
    window_just_opened = bool(
        event_start and (now_utc - event_start).total_seconds() < 60
    )
    price_to_beat = get_window_reference_price(
        best.get("slug", ""),
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

    # ── Step 4: Composite signal ──────────────────────────────────────────────
    signal     = get_composite_signal(cex_signals, poly_signals, config=cfg)
    score      = signal["score"]
    confidence = signal["confidence"]

    if not signal["should_trade"]:
        reason = signal.get("filter_reason", "neutral band")
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"m5={m5:+.3f}% vs_ref={vs_ref_str} poly={poly_p:.3f} vol={vol_r:.2f}x | "
            f"score={score:.3f} BLOCK: {reason}"
        )
        return

    side         = signal["side"]
    position_pct = signal["position_pct"]

    # Hard liquidity floor — FAK orders fail on thin books regardless of signal
    if vol_r < MIN_LIQUIDITY_RATIO:
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
        if implied_wr < breakeven_wr + 0.03:
            log(
                f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
                f"score={score:.3f} → {side.upper()} BLOCK: fee EV negative "
                f"(implied={implied_wr:.1%} < breakeven={breakeven_wr:.1%})"
            )
            return

    # ── Position sizing ───────────────────────────────────────────────────────
    position_size    = calculate_position_size(MAX_POSITION_USD * position_pct, smart_sizing)
    remaining_budget = DAILY_BUDGET - daily_spend["spent"]

    if remaining_budget <= 0:
        log(
            f"{mode_tag} {now_str} | {slug_short} | BLOCK: daily budget exhausted "
            f"(${daily_spend['spent']:.2f}/${DAILY_BUDGET:.2f})",
            force=True,
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
            position_size = min_order_usdc  # bump up to meet minimum
        else:
            log(
                f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
                f"score={score:.3f} → {side.upper()} BLOCK: "
                f"can't meet {MIN_SHARES_PER_ORDER}-share min ${min_order_usdc:.2f} "
                f"(budget=${remaining_budget:.2f})"
            )
            return

    # ── Step 5: Import & execute ──────────────────────────────────────────────
    log(
        f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
        f"m5={m5:+.3f}% vs_ref={vs_ref_str} vol={vol_r:.2f}x | "
        f"score={score:.3f} conf={confidence:.2f} → {side.upper()} ${position_size:.2f}"
    )

    # Gate: only consume an import slot on strong signals
    slug = best["slug"]
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
        if result and result.get("success"):
            shares   = result.get("shares_bought") or result.get("shares") or 0
            trade_id = result.get("trade_id")
            log(f"  TRADED {shares:.1f} {side.upper()} shares @ ${entry_price:.3f}", force=True)
            daily_spend["spent"]  += position_size
            daily_spend["trades"] += 1
            _save_daily_spend(daily_spend)
            if trade_id and JOURNAL_AVAILABLE:
                log_trade(
                    trade_id=trade_id,
                    source=TRADE_SOURCE,
                    thesis=f"{side.upper()} score={score:.3f} conf={confidence:.3f}",
                    confidence=round(confidence, 2),
                    asset=ASSET,
                    momentum_pct=round(cex_signals.get("momentum_5m", 0), 3),
                    volume_ratio=round(cex_signals.get("volume_ratio", 1.0), 2),
                    signal_source=SIGNAL_SOURCE,
                )
        else:
            error = result.get("error", "Unknown") if result else "no response"
            log(f"  ERROR trade failed: {error}", force=True)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simmer FastLoop Trading Skill")
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