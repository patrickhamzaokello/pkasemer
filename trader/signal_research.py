#!/usr/bin/env python3
"""
FastLoop Signal Research Framework

Collects and logs multi-factor signal candidates alongside actual Polymarket
outcomes. Run this in dry-run mode to build a dataset, then analyze with
analyze_signals.py to find what actually has predictive power.

Usage:
    python signal_research.py --collect          # Log signals + outcomes to DB
    python signal_research.py --analyze          # Show correlation report
    python signal_research.py --analyze --min-n 50  # Only show signals with 50+ observations
    python signal_research.py --live             # Trade using composite score
    python signal_research.py --export signals.csv  # Export raw data

Signal candidates tracked:
  CEX Signals (Binance):
    - momentum_1m, momentum_5m, momentum_15m     % price change
    - rsi_14                                      overbought/oversold
    - volume_ratio                                current vs avg volume
    - spread_bps                                  bid-ask spread (order book)
    - order_imbalance                             buy pressure vs sell pressure
    - trade_flow_ratio                            recent trades buy/sell ratio
    - volatility_1m, volatility_5m               realized vol (std of returns)
    - price_acceleration                          momentum of momentum

  Polymarket Signals:
    - poly_yes_price                              current YES price
    - poly_divergence                             yes_price - 0.50
    - poly_spread                                 best_ask - best_bid on CLOB
    - poly_volume_24h                             recent market liquidity
    - poly_order_imbalance                        CLOB buy vs sell depth

  Derived / Combo Signals:
    - btc_vs_reference                            (price_now - ref) / ref * 100
    - cex_poly_lag                                momentum direction vs poly pricing
    - momentum_consistency                        # of recent candles agreeing on direction
    - vol_adjusted_momentum                       momentum / recent volatility

Changes from original:
    - collect_one() now imports from market_utils instead of fast_trader —
      eliminates the circular import (fast_trader → signal_research → fast_trader).
    - btc_vs_reference seeding now only fires in the first 60s after window open,
      preventing a restart mid-window from zeroing out the dominant signal.
    - _determine_outcome() validates that outcomes[0] is labelled "up" before
      assuming outcomePrices[0] corresponds to the Up token.
    - _CACHE_FILE and DB_PATH use _DATA_DIR for consistent path resolution between
      Docker (/data) and local (script directory) environments.
"""

import os
import sys
import json
import math
import sqlite3
import argparse
import time
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Paths — consistent with fast_trader.py
# =============================================================================

_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.environ.get("DB_PATH", os.path.join(_DATA_DIR, "signal_research.db"))

BINANCE_WS_SNAPSHOT = "https://api.binance.com/api/v3/depth"
BINANCE_TRADES      = "https://api.binance.com/api/v3/trades"
BINANCE_KLINES      = "https://api.binance.com/api/v3/klines"
POLY_CLOB           = "https://clob.polymarket.com"
GAMMA_API           = "https://gamma-api.polymarket.com"

# =============================================================================
# Window Reference Price Cache
#
# Records Binance price at first observation of each 5m window.
# Polymarket's eventMetadata.priceToBeat is never populated via the API,
# so we maintain our own reference to compute btc_vs_reference.
#
# FIX: seeding now only fires in the first 60s after window open (controlled
# by the caller passing window_open=True). Seeding on every cycle caused a
# restart mid-window to zero out btc_vs_reference for the rest of the window.
# =============================================================================

_REF_CACHE_FILE = os.path.join(_DATA_DIR, "window_refs.json")


def _load_ref_cache():
    try:
        with open(_REF_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_ref_cache(cache):
    try:
        os.makedirs(os.path.dirname(_REF_CACHE_FILE), exist_ok=True)
        with open(_REF_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def get_window_reference_price(slug, cex_price_now=None, window_open=False):
    """
    Return the cached Binance reference price for this market window slug.

    window_open should be True ONLY in the first ~60s after the window starts.
    When True and no cached price exists, records cex_price_now as the reference.

    Returns None if:
    - The window hasn't opened yet (window_open=False and no cache entry)
    - No price is available to record
    """
    if not slug:
        return None
    cache = _load_ref_cache()
    if slug in cache:
        return float(cache[slug])
    if window_open and cex_price_now:
        cache[slug] = cex_price_now
        # Keep last 60 entries (~5 hours of 5-minute windows)
        if len(cache) > 60:
            for k in sorted(cache.keys())[:-48]:
                del cache[k]
        _save_ref_cache(cache)
        return cex_price_now
    return None


# =============================================================================
# DB Setup
# =============================================================================

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            market_slug TEXT,
            market_condition_id TEXT,
            seconds_remaining REAL,
            outcome TEXT,             -- 'up' or 'down' (filled in after resolution)
            resolved INTEGER DEFAULT 0,

            -- CEX signals
            momentum_1m REAL,
            momentum_5m REAL,
            momentum_15m REAL,
            rsi_14 REAL,
            volume_ratio REAL,
            spread_bps REAL,
            order_imbalance REAL,
            trade_flow_ratio REAL,
            volatility_1m REAL,
            volatility_5m REAL,
            price_acceleration REAL,
            price_now REAL,

            -- Market reference
            price_to_beat REAL,

            -- Polymarket signals (sourced from Gamma API)
            poly_yes_price REAL,
            poly_divergence REAL,
            poly_spread REAL,
            poly_volume_24h REAL,
            poly_order_imbalance REAL,

            -- Derived
            btc_vs_reference REAL,
            cex_poly_lag REAL,
            momentum_consistency REAL,
            vol_adjusted_momentum REAL,

            -- Meta
            traded INTEGER DEFAULT 0,
            trade_side TEXT,
            trade_amount REAL,
            trade_result REAL
        )
    """)
    # Migrate existing DBs that predate these columns
    for col, typedef in [
        ("price_to_beat",      "REAL"),
        ("btc_vs_reference",   "REAL"),
        ("signal_score",       "REAL"),
        ("signal_side",        "TEXT"),
        ("signal_confidence",  "REAL"),
        ("would_trade",        "INTEGER"),
        ("filter_reason",      "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE signal_observations ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


# =============================================================================
# Data Fetchers
# =============================================================================

def _get(url, timeout=8):
    try:
        req = Request(url, headers={"User-Agent": "fastloop-research/1.0"})
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def fetch_binance_klines(symbol="BTCUSDT", interval="1m", limit=20):
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
    return _get(url)


def fetch_binance_orderbook(symbol="BTCUSDT", limit=20):
    url = f"{BINANCE_WS_SNAPSHOT}?symbol={symbol}&limit={limit}"
    return _get(url)


def fetch_binance_recent_trades(symbol="BTCUSDT", limit=500):
    url = f"{BINANCE_TRADES}?symbol={symbol}&limit={limit}"
    return _get(url)


def fetch_poly_clob_orderbook(condition_id):
    """Fetch Polymarket CLOB order book for a condition."""
    url = f"{POLY_CLOB}/book?token_id={condition_id}"
    return _get(url)


def fetch_poly_market(condition_id):
    """Fetch Polymarket market details from Gamma."""
    url = f"{GAMMA_API}/markets?conditionId={condition_id}"
    result = _get(url)
    if result and isinstance(result, list) and result:
        return result[0]
    return None


# =============================================================================
# Signal Calculators
# =============================================================================

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_volatility(returns):
    """Realized volatility from a list of returns."""
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def calc_order_imbalance(bids, asks, levels=5):
    """
    Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    Range: -1 (all ask pressure) to +1 (all bid pressure)
    """
    bid_vol = sum(float(b[1]) for b in bids[:levels])
    ask_vol = sum(float(a[1]) for a in asks[:levels])
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def calc_trade_flow_ratio(trades):
    """
    Ratio of buy-initiated volume to total volume.
    Binance marks isBuyerMaker: True means buyer was maker (seller hit the bid = sell flow).
    """
    buy_vol  = sum(float(t["qty"]) for t in trades if not t.get("isBuyerMaker", True))
    sell_vol = sum(float(t["qty"]) for t in trades if t.get("isBuyerMaker", True))
    total = buy_vol + sell_vol
    if total == 0:
        return 0.5
    return buy_vol / total


def extract_cex_signals(symbol="BTCUSDT"):
    """Pull all CEX signals. Returns dict of signal_name -> value."""
    signals = {}

    k1 = fetch_binance_klines(symbol, "1m", 20)
    k5 = fetch_binance_klines(symbol, "5m", 20)

    if k1 and len(k1) >= 2:
        closes_1m  = [float(c[4]) for c in k1]
        volumes_1m = [float(c[5]) for c in k1]

        signals["momentum_1m"] = (closes_1m[-1] - closes_1m[-2]) / closes_1m[-2] * 100
        signals["price_now"]   = closes_1m[-1]
        signals["rsi_14"]      = calc_rsi(closes_1m)

        avg_vol = sum(volumes_1m[:-1]) / max(len(volumes_1m) - 1, 1)
        signals["volume_ratio"] = volumes_1m[-1] / avg_vol if avg_vol > 0 else 1.0

        returns_1m = [(closes_1m[i] - closes_1m[i-1]) / closes_1m[i-1]
                      for i in range(1, len(closes_1m))]
        signals["volatility_1m"] = calc_volatility(returns_1m[-5:])

        direction = 1 if signals["momentum_1m"] > 0 else -1
        recent_moves = [1 if closes_1m[i] > closes_1m[i-1] else -1
                        for i in range(max(1, len(closes_1m)-5), len(closes_1m))]
        agreement = sum(1 for m in recent_moves if m == direction)
        signals["momentum_consistency"] = agreement / len(recent_moves) if recent_moves else 0.5

    if k5 and len(k5) >= 2:
        closes_5m = [float(c[4]) for c in k5]

        signals["momentum_5m"] = (closes_5m[-1] - closes_5m[-2]) / closes_5m[-2] * 100

        returns_5m = [(closes_5m[i] - closes_5m[i-1]) / closes_5m[i-1]
                      for i in range(1, len(closes_5m))]
        signals["volatility_5m"] = calc_volatility(returns_5m[-5:])

        if len(closes_5m) >= 3:
            m_recent = (closes_5m[-1] - closes_5m[-2]) / closes_5m[-2] * 100
            m_prior  = (closes_5m[-2] - closes_5m[-3]) / closes_5m[-3] * 100
            signals["price_acceleration"] = m_recent - m_prior

    k15 = fetch_binance_klines(symbol, "15m", 5)
    if k15 and len(k15) >= 2:
        closes_15m = [float(c[4]) for c in k15]
        signals["momentum_15m"] = (closes_15m[-1] - closes_15m[-2]) / closes_15m[-2] * 100

    if signals.get("momentum_5m") and signals.get("volatility_5m") and signals["volatility_5m"] > 0:
        signals["vol_adjusted_momentum"] = signals["momentum_5m"] / signals["volatility_5m"]

    ob = fetch_binance_orderbook(symbol, 20)
    if ob and ob.get("bids") and ob.get("asks"):
        bids = ob["bids"]
        asks = ob["asks"]
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2
        signals["spread_bps"]      = ((best_ask - best_bid) / mid) * 10000
        signals["order_imbalance"] = calc_order_imbalance(bids, asks)

    trades = fetch_binance_recent_trades(symbol, 500)
    if trades:
        signals["trade_flow_ratio"] = calc_trade_flow_ratio(trades)

    return signals


def extract_poly_signals(market_data):
    """
    Extract Polymarket signals from a market dict (Gamma events API format).

    Uses live fields directly from the API — no CLOB call needed:
      lastTradePrice  → poly_yes_price
      bestBid/bestAsk → poly_spread
      volumeClob      → poly_volume_24h
    """
    signals = {}
    if not market_data:
        return signals

    last_trade = market_data.get("lastTradePrice")
    best_bid   = market_data.get("bestBid")
    best_ask   = market_data.get("bestAsk")

    yes_price = None
    if last_trade is not None:
        try:
            yes_price = float(last_trade)
        except (ValueError, TypeError):
            pass

    if yes_price is None and best_bid is not None and best_ask is not None:
        try:
            yes_price = (float(best_bid) + float(best_ask)) / 2
        except (ValueError, TypeError):
            pass

    if yes_price is not None:
        signals["poly_yes_price"]  = yes_price
        signals["poly_divergence"] = yes_price - 0.50

    if best_bid is not None and best_ask is not None:
        try:
            signals["poly_spread"] = float(best_ask) - float(best_bid)
        except (ValueError, TypeError):
            pass
    elif market_data.get("spread") is not None:
        try:
            signals["poly_spread"] = float(market_data["spread"])
        except (ValueError, TypeError):
            pass

    vol = market_data.get("volumeClob") or market_data.get("volume24hr")
    if vol is not None:
        try:
            signals["poly_volume_24h"] = float(vol)
        except (ValueError, TypeError):
            signals["poly_volume_24h"] = 0.0

    return signals


def derive_combo_signals(cex, poly, price_to_beat=None):
    """Derive cross-signal features."""
    derived = {}

    if cex.get("momentum_5m") is not None and poly.get("poly_divergence") is not None:
        poly_div = poly["poly_divergence"]
        derived["cex_poly_lag"] = cex["momentum_5m"] * (0.15 - max(-0.15, min(0.15, poly_div))) / 0.15

    if price_to_beat and cex.get("price_now"):
        derived["btc_vs_reference"] = (
            (cex["price_now"] - price_to_beat) / price_to_beat * 100
        )

    return derived


# =============================================================================
# Outcome Resolution
# =============================================================================

def resolve_outcomes(conn, symbol="BTCUSDT"):
    """
    For unresolved observations, check if market has resolved and fill in outcome.

    Resolution strategy (in priority order):
      1. market.winners field — explicit winner token label ("Up" / "Down")
      2. outcomePrices — YES token final price (>0.8 = up, <0.2 = down)
         Only used after validating that outcomes[0] is labelled "Up".
      3. outcomes array with winner flag
      4. Mark as 'unclear' if closed but can't determine direction
    """
    cursor = conn.execute("""
        SELECT id, market_condition_id, market_slug, ts
        FROM signal_observations
        WHERE resolved = 0 OR outcome = 'unclear'
        ORDER BY ts ASC LIMIT 100
    """)
    rows = cursor.fetchall()
    resolved_count = 0
    skipped_open = 0

    for row in rows:
        obs_id, cond_id, slug, ts_str = row
        if not slug and not cond_id:
            continue

        try:
            obs_time = datetime.fromisoformat(ts_str)
        except ValueError:
            continue

        # Only attempt resolution if observation is at least 10 minutes old
        age_seconds = (datetime.now(timezone.utc) - obs_time).total_seconds()
        if age_seconds < 600:
            skipped_open += 1
            continue

        # Prefer slug-based lookup — conditionId lookup returns wrong markets
        # for fast-market condition IDs (Polymarket API bug).
        market = None
        if slug:
            data = _get(f"{GAMMA_API}/events?slug={slug}")
            if data and isinstance(data, list) and data:
                mkts = data[0].get("markets") or []
                if mkts:
                    market = mkts[0]

        if market is None and cond_id:
            market = fetch_poly_market(cond_id)

        if not market:
            continue

        if not market.get("closed", False):
            skipped_open += 1
            continue

        outcome = _determine_outcome(market)
        conn.execute("""
            UPDATE signal_observations
            SET resolved = 1, outcome = ?
            WHERE id = ?
        """, (outcome, obs_id))
        resolved_count += 1

    conn.commit()
    if skipped_open:
        print(f"  ({skipped_open} observations still open / too recent)")
    return resolved_count


def _determine_outcome(market):
    """
    Extract up/down outcome from a resolved Gamma market dict.
    Tries multiple fields since Gamma API is inconsistent post-resolution.

    FIX: outcomePrices strategy now validates that outcomes[0] is labelled "Up"
    before assuming prices[0] corresponds to the Up token. Prevents silently
    flipping outcomes on markets with different outcome orderings.
    """
    # Strategy 1: explicit winners array e.g. ["Up"] or ["Down"]
    winners = market.get("winners") or market.get("winner")
    if winners:
        if isinstance(winners, list):
            winners = winners[0] if winners else ""
        w = str(winners).lower()
        if "up" in w:
            return "up"
        if "down" in w:
            return "down"

    # Strategy 2: outcomePrices — only safe when we know which token is which.
    # Validate that outcomes[0] is "Up" before treating prices[0] as Up probability.
    try:
        outcomes_raw = market.get("outcomes", "[]")
        if isinstance(outcomes_raw, str):
            outcomes_list = json.loads(outcomes_raw)
        else:
            outcomes_list = outcomes_raw

        if isinstance(outcomes_list, list) and len(outcomes_list) >= 2:
            first_label = str(outcomes_list[0]).lower()
            second_label = str(outcomes_list[1]).lower()

            # Only proceed if we can confirm the token ordering
            if "up" in first_label and "down" in second_label:
                prices = json.loads(market.get("outcomePrices", "[]"))
                if len(prices) >= 2:
                    yes_price = float(prices[0])
                    no_price  = float(prices[1])
                    if yes_price > 0.8:
                        return "up"
                    if no_price > 0.8:
                        return "down"
                    if yes_price > no_price and yes_price > 0.6:
                        return "up"
                    if no_price > yes_price and no_price > 0.6:
                        return "down"
            elif "down" in first_label and "up" in second_label:
                # Reversed ordering — swap interpretation
                prices = json.loads(market.get("outcomePrices", "[]"))
                if len(prices) >= 2:
                    down_price = float(prices[0])
                    up_price   = float(prices[1])
                    if up_price > 0.8:
                        return "up"
                    if down_price > 0.8:
                        return "down"
            # Unknown ordering — skip outcomePrices entirely
    except (ValueError, json.JSONDecodeError, IndexError, TypeError):
        pass

    # Strategy 3: outcomes array with winner flag
    outcomes = market.get("outcomes")
    if outcomes:
        try:
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            for o in outcomes:
                if isinstance(o, dict) and o.get("winner"):
                    label = str(o.get("value") or o.get("label") or "").lower()
                    if "up" in label:
                        return "up"
                    if "down" in label:
                        return "down"
        except Exception:
            pass

    return "unclear"


# =============================================================================
# Signal Analysis / Correlation Report
# =============================================================================

SIGNAL_COLUMNS = [
    "momentum_1m", "momentum_5m", "momentum_15m",
    "rsi_14", "volume_ratio", "spread_bps",
    "order_imbalance", "trade_flow_ratio",
    "volatility_1m", "volatility_5m", "price_acceleration",
    "poly_yes_price", "poly_divergence", "poly_spread",
    "poly_volume_24h", "poly_order_imbalance",
    "btc_vs_reference", "cex_poly_lag", "momentum_consistency", "vol_adjusted_momentum",
    "seconds_remaining",
]


def analyze_signals(conn, min_n=20):
    """
    For each signal column, compute:
    - Point-biserial correlation with outcome (up=1, down=0)
    - Win rate when signal is positive vs negative
    - Mean value for up vs down outcomes
    """
    cursor = conn.execute("""
        SELECT * FROM signal_observations
        WHERE resolved = 1 AND outcome IN ('up', 'down')
    """)
    rows = cursor.fetchall()
    col_names = [d[0] for d in cursor.description]

    if not rows:
        print("No resolved observations yet. Run --collect for a while first.")
        return

    outcomes = []
    data = {col: [] for col in SIGNAL_COLUMNS}

    for row in rows:
        rec = dict(zip(col_names, row))
        outcome_val = 1 if rec["outcome"] == "up" else 0
        outcomes.append(outcome_val)
        for col in SIGNAL_COLUMNS:
            data[col].append(rec.get(col))

    print(f"\n{'─'*72}")
    print(f"  SIGNAL CORRELATION REPORT  ({len(outcomes)} resolved observations)")
    print(f"{'─'*72}")
    print(f"  {'Signal':<28} {'N':>5}  {'Corr':>7}  {'WR(+)':>7}  {'WR(-)':>7}  {'Edge':>7}")
    print(f"{'─'*72}")

    results = []
    for col in SIGNAL_COLUMNS:
        vals = data[col]
        paired = [(v, o) for v, o in zip(vals, outcomes) if v is not None]
        if len(paired) < min_n:
            continue
        v_list = [p[0] for p in paired]
        o_list = [p[1] for p in paired]

        n      = len(v_list)
        mean_v = sum(v_list) / n
        mean_o = sum(o_list) / n
        std_v  = math.sqrt(sum((x - mean_v)**2 for x in v_list) / n) or 1e-9
        std_o  = math.sqrt(sum((x - mean_o)**2 for x in o_list) / n) or 1e-9
        corr   = sum((v_list[i] - mean_v) * (o_list[i] - mean_o) for i in range(n)) / (n * std_v * std_o)

        pos_wins = [o for v, o in paired if v > 0]
        neg_wins = [o for v, o in paired if v <= 0]
        wr_pos = sum(pos_wins) / len(pos_wins) if pos_wins else float('nan')
        wr_neg = sum(neg_wins) / len(neg_wins) if neg_wins else float('nan')
        edge   = wr_pos - wr_neg

        results.append((abs(corr), col, len(paired), corr, wr_pos, wr_neg, edge))

    results.sort(reverse=True)
    for _, col, n, corr, wr_pos, wr_neg, edge in results:
        wr_pos_str = f"{wr_pos:.1%}" if not math.isnan(wr_pos) else "  n/a"
        wr_neg_str = f"{wr_neg:.1%}" if not math.isnan(wr_neg) else "  n/a"
        print(f"  {col:<28} {n:>5}  {corr:>+7.3f}  {wr_pos_str:>7}  {wr_neg_str:>7}  {edge:>+7.3f}")

    print(f"{'─'*72}")
    print(f"\n  Interpretation:")
    print(f"    Corr > +0.10  : signal has directional predictive power")
    print(f"    Edge > +0.05  : positive signal wins 5%+ more often than negative")
    print(f"    Best signals for the composite score = high |corr| AND high edge\n")


def export_csv(conn, path):
    """Export all observations to CSV."""
    cursor = conn.execute("SELECT * FROM signal_observations ORDER BY ts")
    rows   = cursor.fetchall()
    cols   = [d[0] for d in cursor.description]
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {path}")


# =============================================================================
# Collection Loop
# =============================================================================

def _now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def collect_one(conn, asset="BTC", window="5m", symbol="BTCUSDT"):
    # FIX: import from market_utils, not fast_trader — eliminates circular import
    from market_utils import discover_fast_market_markets, find_best_fast_market

    markets = discover_fast_market_markets(asset, window)
    if not markets:
        print(f"[{_now()}] No active markets found")
        return

    best = find_best_fast_market(markets)
    if not best:
        print(f"[{_now()}] No markets with enough time remaining")
        return

    cond_id = best.get("condition_id", "")
    now = datetime.now(timezone.utc)

    end_time = best.get("end_time")
    if not end_time:
        print(f"[{_now()}] SKIP: no end_time on best market")
        return

    expired_secs = (now - end_time).total_seconds()
    if expired_secs > 600:
        print(f"[{_now()}] SKIP: market ended {expired_secs/60:.0f}m ago — {best.get('slug','')[:50]}")
        return

    seconds_remaining = (end_time - now).total_seconds()

    cex  = extract_cex_signals(symbol)
    poly = extract_poly_signals(best)

    # FIX: only seed reference price in the first 60s after window opens.
    # Seeding on every cycle would reset btc_vs_reference to 0 on restart.
    event_start = best.get("event_start")
    window_just_opened = bool(
        event_start and (now - event_start).total_seconds() < 60
    )
    price_to_beat = get_window_reference_price(
        best.get("slug", ""),
        cex_price_now=cex.get("price_now"),
        window_open=window_just_opened,
    )

    derived     = derive_combo_signals(cex, poly, price_to_beat)
    all_signals = {**cex, **poly, **derived}

    row = {
        "ts":                   now.isoformat(),
        "market_slug":          best.get("slug", ""),
        "market_condition_id":  cond_id,
        "seconds_remaining":    seconds_remaining,
        "price_now":            all_signals.get("price_now"),
        "price_to_beat":        price_to_beat,
        **{k: all_signals.get(k) for k in SIGNAL_COLUMNS if k != "seconds_remaining"},
    }

    cursor = conn.execute("""
        INSERT INTO signal_observations
        (ts, market_slug, market_condition_id, seconds_remaining,
         momentum_1m, momentum_5m, momentum_15m, rsi_14, volume_ratio,
         spread_bps, order_imbalance, trade_flow_ratio, volatility_1m,
         volatility_5m, price_acceleration, price_now, price_to_beat,
         poly_yes_price, poly_divergence, poly_spread, poly_volume_24h,
         poly_order_imbalance, btc_vs_reference, cex_poly_lag,
         momentum_consistency, vol_adjusted_momentum)
        VALUES
        (:ts, :market_slug, :market_condition_id, :seconds_remaining,
         :momentum_1m, :momentum_5m, :momentum_15m, :rsi_14, :volume_ratio,
         :spread_bps, :order_imbalance, :trade_flow_ratio, :volatility_1m,
         :volatility_5m, :price_acceleration, :price_now, :price_to_beat,
         :poly_yes_price, :poly_divergence, :poly_spread, :poly_volume_24h,
         :poly_order_imbalance, :btc_vs_reference, :cex_poly_lag,
         :momentum_consistency, :vol_adjusted_momentum)
    """, row)
    obs_id = cursor.lastrowid

    # Compute composite signal and store prediction alongside observation
    sig = None
    try:
        from composite_signal import get_composite_signal
        cex_for_sig = {k: all_signals.get(k) for k in [
            "momentum_5m", "rsi_14", "volume_ratio", "order_imbalance",
            "trade_flow_ratio", "volatility_5m", "price_acceleration",
            "btc_vs_reference", "cex_poly_lag", "momentum_consistency",
            "vol_adjusted_momentum", "momentum_1m", "momentum_15m",
        ]}
        poly_for_sig = {
            "poly_yes_price":  all_signals.get("poly_yes_price"),
            "poly_divergence": all_signals.get("poly_divergence"),
            "poly_spread":     all_signals.get("poly_spread"),
        }
        sig = get_composite_signal(cex_for_sig, poly_for_sig)
        conn.execute("""
            UPDATE signal_observations
            SET signal_score = ?, signal_side = ?, signal_confidence = ?,
                would_trade = ?, filter_reason = ?
            WHERE id = ?
        """, (
            round(sig["score"], 6),
            sig.get("side"),
            round(sig["confidence"], 6),
            1 if sig["should_trade"] else 0,
            sig.get("filter_reason"),
            obs_id,
        ))
    except Exception:
        pass

    conn.commit()

    m5         = all_signals.get("momentum_5m", 0) or 0
    poly_p     = all_signals.get("poly_yes_price", 0.5) or 0.5
    vs_ref     = all_signals.get("btc_vs_reference")
    vol_r      = all_signals.get("volume_ratio", 1.0) or 1.0
    vs_ref_str = f"{vs_ref:+.4f}%" if vs_ref is not None else "n/a"

    score_str = ""
    if sig is not None:
        try:
            score_str = f"score={sig['score']:.3f} → {(sig.get('side') or 'neutral').upper()}"
            if sig["should_trade"]:
                score_str += f" (conf={sig['confidence']:.2f}) WOULD_TRADE"
            elif sig.get("filter_reason"):
                score_str += f" | {sig['filter_reason']}"
        except Exception:
            pass

    print(
        f"[{_now()}] {best.get('slug','')[-19:]:19} {seconds_remaining:4.0f}s | "
        f"m5={m5:+.3f}% vs_ref={vs_ref_str} poly={poly_p:.3f} vol={vol_r:.2f}x | "
        f"{score_str}"
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastLoop Signal Research")
    parser.add_argument("--collect",  action="store_true",
                        help="Collect signals continuously (Ctrl-C to stop)")
    parser.add_argument("--analyze",  action="store_true",
                        help="Show correlation report")
    parser.add_argument("--resolve",  action="store_true",
                        help="Resolve pending outcomes")
    parser.add_argument("--export",   metavar="FILE",
                        help="Export observations to CSV")
    parser.add_argument("--min-n",    type=int, default=20,
                        help="Min observations to show in report")
    parser.add_argument("--interval", type=int, default=30,
                        help="Collection interval in seconds (default: 30)")
    parser.add_argument("--asset",    default="BTC")
    parser.add_argument("--window",   default="5m")
    args = parser.parse_args()

    conn = init_db()

    if args.resolve:
        n = resolve_outcomes(conn)
        print(f"Resolved {n} outcomes")

    if args.export:
        export_csv(conn, args.export)

    if args.analyze:
        resolve_outcomes(conn)
        analyze_signals(conn, args.min_n)

    if args.collect:
        symbol = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}.get(args.asset, "BTCUSDT")
        print(f"Collecting signals every {args.interval}s. Ctrl-C to stop.")
        print(f"DB: {DB_PATH}\n")
        try:
            while True:
                try:
                    collect_one(conn, args.asset, args.window, symbol)
                except Exception as e:
                    print(f"[{_now()}] Error: {e}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")