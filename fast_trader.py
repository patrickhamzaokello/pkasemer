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
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from composite_signal import get_composite_signal
from signal_research import extract_cex_signals, extract_poly_signals, get_window_reference_price
import time
from datetime import datetime, timezone

# Coin slugs exactly as Polymarket uses them
COIN_SLUGS = {
    "BTC": "btc",
    "ETH": "eth", 
    "SOL": "sol",
    "XRP": "xrp",
}


# Force line-buffered stdout for non-TTY environments (cron, Docker, OpenClaw)
sys.stdout.reconfigure(line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

# Optional: Trade Journal integration
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
# Configuration (config.json > env vars > defaults)
# =============================================================================

CONFIG_SCHEMA = {
    "entry_threshold":   {"default": 0.05,    "env": "SIMMER_SPRINT_ENTRY",       "type": float},
    "min_momentum_pct":  {"default": 0.15,    "env": "SIMMER_SPRINT_MOMENTUM",    "type": float},
    "max_position":      {"default": 5.0,     "env": "SIMMER_SPRINT_MAX_POSITION","type": float},
    "signal_source":     {"default": "binance","env": "SIMMER_SPRINT_SIGNAL",     "type": str},
    "lookback_minutes":  {"default": 5,       "env": "SIMMER_SPRINT_LOOKBACK",    "type": int},
    "min_time_remaining":{"default": 30,      "env": "SIMMER_SPRINT_MIN_TIME",    "type": int},
    "asset":             {"default": "BTC",   "env": "SIMMER_SPRINT_ASSET",       "type": str},
    "window":            {"default": "5m",    "env": "SIMMER_SPRINT_WINDOW",      "type": str},
    "volume_confidence": {"default": True,    "env": "SIMMER_SPRINT_VOL_CONF",    "type": bool},
    "daily_budget":      {"default": 10.0,    "env": "SIMMER_SPRINT_DAILY_BUDGET","type": float},
    "composite_threshold":{"default": 0.60,  "env": "SIMMER_SPRINT_COMP_THRESH", "type": float},
}

TRADE_SOURCE = "sdk:fastloop"
SMART_SIZING_PCT = 0.05
MIN_SHARES_PER_ORDER = 5

ASSET_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
ASSET_PATTERNS = {
    "BTC": ["bitcoin up or down"],
    "ETH": ["ethereum up or down"],
    "SOL": ["solana up or down"],
}

def get_fast_market_slugs(asset="BTC", include_next=True):
    """
    Generate Polymarket 5m fast market slugs using Unix timestamp bucketing.
    Format: {coin}-updown-5m-{unix_ts_rounded_to_5min}
    
    Returns current bucket slug and optionally the next one.
    """
    now = int(time.time())
    current_bucket = (now // 300) * 300
    next_bucket = current_bucket + 300

    coin = COIN_SLUGS.get(asset.upper(), asset.lower())
    slugs = [f"{coin}-updown-5m-{current_bucket}"]
    if include_next:
        slugs.append(f"{coin}-updown-5m-{next_bucket}")
    return slugs


def _load_config(schema, skill_file, config_filename="config.json"):
    from pathlib import Path
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
                result[key] = val.lower() in ("true","1","yes") if type_fn == bool else type_fn(val)
            except (ValueError, TypeError):
                result[key] = spec.get("default")
        else:
            result[key] = spec.get("default")
    return result


def _get_config_path(skill_file, config_filename="config.json"):
    from pathlib import Path
    return Path(skill_file).parent / config_filename


def _update_config(updates, skill_file, config_filename="config.json"):
    from pathlib import Path
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


# Load config
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


# =============================================================================
# Daily Budget Tracking
# =============================================================================

def _get_spend_path(skill_file):
    from pathlib import Path
    return Path(skill_file).parent / "daily_spend.json"


def _load_daily_spend(skill_file):
    spend_path = _get_spend_path(skill_file)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if spend_path.exists():
        try:
            with open(spend_path) as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return {"date": today, "spent": 0.0, "trades": 0}


def _save_daily_spend(skill_file, spend_data):
    spend_path = _get_spend_path(skill_file)
    with open(spend_path, "w") as f:
        json.dump(spend_data, f, indent=2)


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
# Market Discovery
# =============================================================================

def discover_fast_market_markets(asset="BTC", window="5m"):
    now = datetime.now(timezone.utc)
    markets = []

    for slug in get_fast_market_slugs(asset, include_next=True):
        gamma_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
        result = _api_request(gamma_url)

        if not result or isinstance(result, dict) or len(result) == 0:
            continue

        event = result[0]

        # eventStartTime is on the EVENT object, not the market
        event_start = None
        for key in ("startTime", "eventStartTime"):
            raw = event.get(key)
            if raw and "T" in str(raw):
                try:
                    event_start = datetime.fromisoformat(
                        raw.replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                    break
                except ValueError:
                    pass

        # Skip if start is more than 15 min away (market not yet tradeable)
        if event_start:
            secs_until_start = (event_start - now).total_seconds()
            if secs_until_start > 900:
                continue

        for m in (event.get("markets") or []):
            if m.get("closed", False):
                continue
            if not m.get("acceptingOrders", True):
                continue

            # endDate is on the MARKET object with full ISO timestamp
            end_time = None
            for key in ("endDate",):
                raw = m.get(key)
                if raw and "T" in str(raw):
                    try:
                        end_time = datetime.fromisoformat(
                            raw.replace("Z", "+00:00")
                        ).astimezone(timezone.utc)
                        break
                    except ValueError:
                        pass

            if not end_time:
                continue

            # Skip if expired more than 5 min ago
            if (now - end_time).total_seconds() > 300:
                continue

            # Parse fee rate — real field is makerBaseFee (bps)
            fee_bps = int(
                m.get("makerBaseFee") or
                m.get("feeRateBps") or
                m.get("fee_rate_bps") or 0
            )

            markets.append({
                "question":        m.get("question") or event.get("title") or "",
                "slug":            slug,
                "condition_id":    m.get("conditionId", ""),
                "end_time":        end_time,
                "event_start":     event_start,
                "outcomes":        m.get("outcomes", []),
                "outcome_prices":  m.get("outcomePrices", "[]"),
                "fee_rate_bps":    fee_bps,
                # Live price fields (used directly — no CLOB or secondary fetch needed)
                "lastTradePrice":  m.get("lastTradePrice"),
                "bestBid":         m.get("bestBid"),
                "bestAsk":         m.get("bestAsk"),
                "spread":          m.get("spread"),
                "volumeClob":      m.get("volumeClob"),
                "volume24hr":      m.get("volume24hr"),
                # Reference price at window start (from Chainlink via eventMetadata)
                "price_to_beat":   (event.get("eventMetadata") or {}).get("priceToBeat"),
            })

    return markets


def find_best_fast_market(markets):
    """
    Pick the best market to trade.
    
    Strategy:
    - If current window has > 60s left AND event_start has passed → trade it
    - Otherwise prefer the NEXT window (300s out) so we enter fresh
    - Never trade with < MIN_TIME_REMAINING seconds left
    """
    now = datetime.now(timezone.utc)
    candidates = []

    for m in markets:
        event_start = m.get("event_start")
        end_time = m.get("end_time")

        if not end_time:
            continue

        remaining = (end_time - now).total_seconds()

        # Hard floor — never enter a dying market
        if remaining < MIN_TIME_REMAINING:
            continue

        # If event_start is in the future, market window hasn't opened yet
        # Still include it but note it as "pending"
        window_open = True
        secs_until_start = 0
        if event_start:
            secs_until_start = (event_start - now).total_seconds()
            if secs_until_start > 900:  # more than 15 min away — skip entirely
                continue
            if secs_until_start > 0:
                window_open = False  # window not yet open, but within 15 min

        # Score: strongly prefer markets whose window is already open.
        # priceToBeat (from Chainlink) is only set once the window starts,
        # so we MUST be in an active window to get the btc_vs_reference signal.
        # Pre-window markets are a fallback only — never beat an active window.
        if window_open and remaining > 120:
            score = remaining + 600  # large bonus: active window always beats pre-window
        elif window_open:
            score = remaining * 0.5  # in-window but almost expired — low priority
        else:
            score = remaining * 0.8  # pre-window fallback — no priceToBeat yet

        candidates.append((score, m))

    if not candidates:
        return None

    # Highest score wins
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# =============================================================================
# CEX Price Signal
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
        price_then  = float(candles[0][1])
        price_now   = float(candles[-1][4])
        momentum_pct = ((price_now - price_then) / price_then) * 100
        volumes      = [float(c[5]) for c in candles]
        avg_volume   = sum(volumes) / len(volumes)
        latest_volume = volumes[-1]
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
        return {
            "momentum_pct":   momentum_pct,
            "direction":      "up" if momentum_pct > 0 else "down",
            "price_now":      price_now,
            "price_then":     price_then,
            "avg_volume":     avg_volume,
            "latest_volume":  latest_volume,
            "volume_ratio":   volume_ratio,
            "candles":        len(candles),
        }
    except (IndexError, ValueError, KeyError):
        return None


def get_coingecko_momentum(asset="bitcoin", lookback_minutes=5):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={asset}&vs_currencies=usd"
    result = _api_request(url)
    if not result or (isinstance(result, dict) and result.get("error")):
        return None
    price_now = result.get(asset, {}).get("usd")
    if not price_now:
        return None
    return {
        "momentum_pct": 0, "direction": "neutral",
        "price_now": price_now, "price_then": price_now,
        "avg_volume": 0, "latest_volume": 0, "volume_ratio": 1.0, "candles": 0,
    }


COINGECKO_ASSETS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}


def get_momentum(asset="BTC", source="binance", lookback=5):
    if source == "binance":
        return get_binance_momentum(ASSET_SYMBOLS.get(asset, "BTCUSDT"), lookback)
    elif source == "coingecko":
        return get_coingecko_momentum(COINGECKO_ASSETS.get(asset, "bitcoin"), lookback)
    return None


# =============================================================================
# Import & Trade
# =============================================================================

def import_fast_market_market(slug):
    url = f"https://polymarket.com/event/{slug}"
    try:
        result = get_client().import_market(url)
    except Exception as e:
        return None, str(e)
    if not result:
        return None, "No response from import endpoint"
    if result.get("error"):
        return None, result.get("error", "Unknown error")
    status   = result.get("status")
    market_id = result.get("market_id")
    if status == "resolved":
        alts = result.get("active_alternatives", [])
        return None, f"Market resolved. Try: {alts[0].get('id')}" if alts else "Market resolved, no alternatives"
    if status in ("imported", "already_exists"):
        return market_id, None
    return None, f"Unexpected status: {status}"


def get_market_details(market_id):
    try:
        market = get_client().get_market_by_id(market_id)
        if not market:
            return None
        from dataclasses import asdict
        return asdict(market)
    except Exception:
        return None


def get_portfolio():
    try:
        return get_client().get_portfolio()
    except Exception as e:
        return {"error": str(e)}


def get_positions():
    try:
        positions = get_client().get_positions()
        from dataclasses import asdict
        return [asdict(p) for p in positions]
    except Exception:
        return []


def execute_trade(market_id, side, amount):
    try:
        result = get_client().trade(
            market_id=market_id, side=side,
            amount=amount, source=TRADE_SOURCE,
        )
        return {
            "success":      result.success,
            "trade_id":     result.trade_id,
            "shares_bought":result.shares_bought,
            "shares":       result.shares_bought,
            "error":        result.error,
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

def run_fast_market_strategy(dry_run=True, positions_only=False, show_config=False,
                              smart_sizing=False, quiet=False):

    def log(msg, force=False):
        if not quiet or force:
            print(msg)

    log("⚡ Simmer FastLoop Trading Skill")
    log("=" * 50)

    if dry_run:
        log("\n  [DRY RUN] No trades will be executed. Use --live to enable trading.")

    daily_spend = _load_daily_spend(__file__)

    log(f"\n⚙️  Configuration:")
    log(f"  Asset:            {ASSET}")
    log(f"  Window:           {WINDOW}")
    log(f"  Entry threshold:  {ENTRY_THRESHOLD}")
    log(f"  Min momentum:     {MIN_MOMENTUM_PCT}%")
    log(f"  Max position:     ${MAX_POSITION_USD:.2f}")
    log(f"  Signal source:    {SIGNAL_SOURCE}")
    log(f"  Lookback:         {LOOKBACK_MINUTES} minutes")
    log(f"  Min time left:    {MIN_TIME_REMAINING}s")
    log(f"  Volume weighting: {'✓' if VOLUME_CONFIDENCE else '✗'}")
    log(f"  Daily budget:     ${DAILY_BUDGET:.2f} (${daily_spend['spent']:.2f} spent, {daily_spend['trades']} trades)")

    if show_config:
        log(f"\n  Config file: {_get_config_path(__file__)}")
        log(f"  Edit config.json directly or use --set key=value")
        return

    # Validate API key early
    get_client()

    # ── Positions view ────────────────────────────────────────────────────────
    if positions_only:
        log("\n📊 Sprint Positions:")
        positions = get_positions()
        fast_positions = [p for p in positions if "up or down" in (p.get("question","") or "").lower()]
        if not fast_positions:
            log("  No open fast market positions")
        else:
            for pos in fast_positions:
                log(f"  • {pos.get('question','Unknown')[:60]}")
                log(f"    YES: {pos.get('shares_yes',0):.1f} | NO: {pos.get('shares_no',0):.1f} | P&L: ${pos.get('pnl',0):.2f}")
        return

    if smart_sizing:
        portfolio = get_portfolio()
        if portfolio and not portfolio.get("error"):
            log(f"\n💰 Balance: ${portfolio.get('balance_usdc', 0):.2f}")

    # ── Step 1: Discover markets ──────────────────────────────────────────────
    log(f"\n🔍 Discovering {ASSET} fast markets...")
    markets = discover_fast_market_markets(ASSET, WINDOW)
    log(f"  Found {len(markets)} active fast markets")

    if not markets:
        log("  No active fast markets found")
        log("📊 Summary: No markets available", force=True)
        return

    # ── Step 2: Pick best market ──────────────────────────────────────────────
    best = find_best_fast_market(markets)
    if not best:
        log(f"  No markets with >{MIN_TIME_REMAINING}s remaining")
        log("📊 Summary: No tradeable markets (too close to expiry)", force=True)
        return

    end_time  = best.get("end_time")
    remaining = (end_time - datetime.now(timezone.utc)).total_seconds() if end_time else 0

    log(f"\n🎯 Selected: {best['question']}")
    log(f"  Expires in: {remaining:.0f}s")

    # Parse current market odds
    try:
        prices = json.loads(best.get("outcome_prices", "[]"))
        market_yes_price = float(prices[0]) if prices else 0.5
    except (json.JSONDecodeError, IndexError, ValueError):
        market_yes_price = 0.5

    log(f"  Current YES price: ${market_yes_price:.3f}")

    fee_rate_bps = best.get("fee_rate_bps", 0)
    fee_rate     = fee_rate_bps / 10000
    if fee_rate > 0:
        log(f"  Fee rate: {fee_rate:.0%}")

    # ── Step 3: Collect signals ───────────────────────────────────────────────
    log(f"\n📈 Collecting signals ({ASSET})...")
    symbol = ASSET_SYMBOLS.get(ASSET, "BTCUSDT")

    cex_signals = extract_cex_signals(symbol)
    if not cex_signals:
        log("  ❌ Failed to fetch CEX data", force=True)
        log("📊 Summary: Signal fetch failed", force=True)
        return

    # best dict already has live price data from the events API (lastTradePrice,
    # bestBid, bestAsk, spread) — no secondary fetch or CLOB call needed
    poly_signals = extract_poly_signals(best)

    # btc_vs_reference: how far is BTC from the window's opening reference price?
    # We cache our own Binance price at first window observation (API never provides priceToBeat).
    event_start = best.get("event_start")
    window_open = bool(event_start and datetime.now(timezone.utc) >= event_start)
    price_to_beat = get_window_reference_price(
        best.get("slug", ""),
        cex_price_now=cex_signals.get("price_now"),
        window_open=window_open,
    )
    if price_to_beat and cex_signals.get("price_now"):
        cex_signals["btc_vs_reference"] = (
            (cex_signals["price_now"] - price_to_beat) / price_to_beat * 100
        )

    vs_ref = cex_signals.get("btc_vs_reference")
    vs_ref_str = f"{vs_ref:+.3f}%" if vs_ref is not None else "n/a"
    ptb_str = f"{price_to_beat:.2f}" if price_to_beat else "n/a"
    log(f"  m5={cex_signals.get('momentum_5m', 0):+.3f}%  "
        f"vs_ref={vs_ref_str}  "
        f"poly={poly_signals.get('poly_yes_price', 0.5):.3f}  "
        f"OI={cex_signals.get('order_imbalance', 0):+.3f}  "
        f"TFR={cex_signals.get('trade_flow_ratio', 0.5):.2f}  "
        f"RSI={cex_signals.get('rsi_14', 0):.0f}  "
        f"ptb={ptb_str}")

    # ── Step 4: Composite signal ──────────────────────────────────────────────
    log(f"\n🧠 Computing composite signal...")
    signal = get_composite_signal(cex_signals, poly_signals, config=cfg)

    score      = signal["score"]
    confidence = signal["confidence"]
    log(f"  Score:      {score:.4f}")
    log(f"  Confidence: {confidence:.4f}")

    if not signal["should_trade"]:
        reason = signal.get("filter_reason", "score within neutral band")
        log(f"  ⏸️  No trade: {reason}")
        log(f"📊 Summary: No trade — {reason}", force=True)
        return

    side         = signal["side"]          # 'yes' or 'no'
    position_pct = signal["position_pct"]  # 0–1 scale

    # ── Fee-aware EV check ────────────────────────────────────────────────────
    entry_price = market_yes_price if side == "yes" else (1 - market_yes_price)
    if fee_rate > 0:
        win_profit    = (1 - entry_price) * (1 - fee_rate)
        breakeven_wr  = entry_price / (win_profit + entry_price)
        implied_wr    = score if side == "yes" else (1 - score)
        if implied_wr < breakeven_wr + 0.03:
            log(f"  ⏸️  Fee-adjusted EV negative "
                f"(implied WR {implied_wr:.1%} < breakeven {breakeven_wr:.1%})")
            log("📊 Summary: No trade — fees eat the edge", force=True)
            return

    # ── Position sizing ───────────────────────────────────────────────────────
    position_size = calculate_position_size(MAX_POSITION_USD * position_pct, smart_sizing)

    remaining_budget = DAILY_BUDGET - daily_spend["spent"]
    if remaining_budget <= 0:
        log(f"  ⏸️  Daily budget exhausted (${daily_spend['spent']:.2f}/${DAILY_BUDGET:.2f})")
        log("📊 Summary: No trade — daily budget exhausted", force=True)
        return
    if position_size > remaining_budget:
        position_size = remaining_budget
        log(f"  Budget cap: capped at ${position_size:.2f}")
    if position_size < 0.50:
        log(f"  ⏸️  Position ${position_size:.2f} < $0.50 minimum")
        log("📊 Summary: No trade — position too small", force=True)
        return
    if entry_price > 0 and (MIN_SHARES_PER_ORDER * entry_price) > position_size:
        log(f"  ⚠️  ${position_size:.2f} too small for {MIN_SHARES_PER_ORDER} shares @ ${entry_price:.2f}")
        log("📊 Summary: No trade — below minimum order size", force=True)
        return

    log(f"\n  ✅ Signal: {side.upper()}  score={score:.4f}  "
        f"confidence={confidence:.2f}  size=${position_size:.2f}", force=True)

    # ── Step 5: Import & execute ──────────────────────────────────────────────
    log(f"\n🔗 Importing to Simmer...", force=True)
    market_id, import_error = import_fast_market_market(best["slug"])

    if not market_id:
        log(f"  ❌ Import failed: {import_error}", force=True)
        log("📊 Summary: Import failed", force=True)
        return

    log(f"  ✅ Market ID: {market_id[:16]}...", force=True)

    traded = False
    if dry_run:
        est_shares = position_size / entry_price if entry_price > 0 else 0
        log(f"  [DRY RUN] Would buy {side.upper()} ${position_size:.2f} "
            f"(~{est_shares:.1f} shares @ ${entry_price:.3f})", force=True)
        traded = True  # count dry-run as "attempted"
    else:
        log(f"  Executing {side.upper()} trade for ${position_size:.2f}...", force=True)
        result = execute_trade(market_id, side, position_size)

        if result and result.get("success"):
            shares   = result.get("shares_bought") or result.get("shares") or 0
            trade_id = result.get("trade_id")
            log(f"  ✅ Bought {shares:.1f} {side.upper()} shares @ ${entry_price:.3f}", force=True)
            traded = True

            # Update daily spend
            daily_spend["spent"]  += position_size
            daily_spend["trades"] += 1
            _save_daily_spend(__file__, daily_spend)

            # Trade journal
            if trade_id and JOURNAL_AVAILABLE:
                log_trade(
                    trade_id=trade_id,
                    source=TRADE_SOURCE,
                    thesis=f"{side.upper()} signal: score={score:.3f} confidence={confidence:.3f}",
                    confidence=round(confidence, 2),
                    asset=ASSET,
                    momentum_pct=round(cex_signals.get("momentum_5m", 0), 3),
                    volume_ratio=round(cex_signals.get("volume_ratio", 1.0), 2),
                    signal_source=SIGNAL_SOURCE,
                )
        else:
            error = result.get("error", "Unknown error") if result else "No response"
            log(f"  ❌ Trade failed: {error}", force=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    action = "DRY RUN" if dry_run else ("TRADED" if traded else "FAILED")
    print(f"\n📊 Summary:")
    print(f"  Market:  {best['question'][:55]}")
    print(f"  Signal:  {side.upper()}  score={score:.4f}  conf={confidence:.3f}")
    print(f"  Price:   YES ${market_yes_price:.3f}  |  entry @ ${entry_price:.3f}")
    print(f"  Action:  {action}  ${position_size:.2f}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simmer FastLoop Trading Skill")
    parser.add_argument("--live",         action="store_true", help="Execute real trades")
    parser.add_argument("--dry-run",      action="store_true", help="Show opportunities without trading (default)")
    parser.add_argument("--positions",    action="store_true", help="Show current positions")
    parser.add_argument("--config",       action="store_true", help="Show current config")
    parser.add_argument("--set",          action="append", metavar="KEY=VALUE", help="Update config")
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
                updates[key] = val.lower() in ("true","1","yes") if type_fn == bool else type_fn(val)
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