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
    - poly_mid_vs_last                            mid price vs last trade

  Derived / Combo Signals:
    - cex_poly_lag                                momentum direction vs poly pricing
    - momentum_consistency                        # of recent candles agreeing on direction
    - vol_adjusted_momentum                       momentum / recent volatility
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

DB_PATH = os.path.join(os.path.dirname(__file__), "signal_research.db")
BINANCE_WS_SNAPSHOT = "https://api.binance.com/api/v3/depth"
BINANCE_TRADES = "https://api.binance.com/api/v3/trades"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
POLY_CLOB = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

# ─────────────────────────────────────────────
# Window Reference Price Cache
# Records Binance price at first observation of each 5m window.
# Polymarket's eventMetadata.priceToBeat is never populated via the API,
# so we maintain our own reference to compute btc_vs_reference.
# Stored at /data/window_refs.json (Docker) or ./window_refs.json (local).
# ─────────────────────────────────────────────

_DATA_DIR = "/data" if os.path.isdir("/data") else os.path.dirname(os.path.abspath(__file__))
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
    On the first cycle after window open, records cex_price_now and returns it.
    Returns None if the window hasn't opened yet or no price available.
    """
    if not slug:
        return None
    cache = _load_ref_cache()
    if slug in cache:
        return float(cache[slug])
    if window_open and cex_price_now:
        cache[slug] = cex_price_now
        # Keep last 48 entries (~4 hours of 5-minute windows)
        if len(cache) > 60:
            for k in sorted(cache.keys())[:-48]:
                del cache[k]
        _save_ref_cache(cache)
        return cex_price_now
    return None


# ─────────────────────────────────────────────
# DB Setup
# ─────────────────────────────────────────────

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
            price_to_beat REAL,       -- Chainlink BTC price at window start (from eventMetadata)

            -- Polymarket signals (sourced from Gamma API, no CLOB needed)
            poly_yes_price REAL,      -- lastTradePrice from API
            poly_divergence REAL,
            poly_spread REAL,         -- bestAsk - bestBid from API
            poly_volume_24h REAL,
            poly_order_imbalance REAL,

            -- Derived
            btc_vs_reference REAL,    -- (price_now - price_to_beat) / price_to_beat * 100
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
    for col, typedef in [("price_to_beat", "REAL"), ("btc_vs_reference", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE signal_observations ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


# ─────────────────────────────────────────────
# Data Fetchers
# ─────────────────────────────────────────────

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


def fetch_binance_recent_trades(symbol="BTCUSDT", limit=500):  # was 100
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


# ─────────────────────────────────────────────
# Signal Calculators
# ─────────────────────────────────────────────

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
    """Annualized realized volatility from a list of returns."""
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
    Binance marks isBuyerMaker: True means the buyer was maker (so seller hit the bid = sell flow).
    """
    buy_vol = sum(float(t["qty"]) for t in trades if not t.get("isBuyerMaker", True))
    sell_vol = sum(float(t["qty"]) for t in trades if t.get("isBuyerMaker", True))
    total = buy_vol + sell_vol
    if total == 0:
        return 0.5
    return buy_vol / total


def extract_cex_signals(symbol="BTCUSDT"):
    """
    Pull all CEX signals. Returns dict of signal_name -> value.
    """
    signals = {}

    # Klines for momentum, RSI, volatility
    k1 = fetch_binance_klines(symbol, "1m", 20)
    k5 = fetch_binance_klines(symbol, "5m", 20)

    if k1 and len(k1) >= 2:
        closes_1m = [float(c[4]) for c in k1]
        opens_1m = [float(c[1]) for c in k1]
        volumes_1m = [float(c[5]) for c in k1]

        # Momentum
        signals["momentum_1m"] = (closes_1m[-1] - closes_1m[-2]) / closes_1m[-2] * 100
        signals["price_now"] = closes_1m[-1]

        # RSI
        signals["rsi_14"] = calc_rsi(closes_1m)

        # Volume ratio (latest vs 10-candle avg)
        avg_vol = sum(volumes_1m[:-1]) / max(len(volumes_1m) - 1, 1)
        signals["volume_ratio"] = volumes_1m[-1] / avg_vol if avg_vol > 0 else 1.0

        # Volatility (std of 1m returns)
        returns_1m = [(closes_1m[i] - closes_1m[i-1]) / closes_1m[i-1]
                      for i in range(1, len(closes_1m))]
        signals["volatility_1m"] = calc_volatility(returns_1m[-5:])

        # Momentum consistency: fraction of last N candles that agree with latest direction
        direction = 1 if signals["momentum_1m"] > 0 else -1
        recent_moves = [1 if closes_1m[i] > closes_1m[i-1] else -1
                        for i in range(max(1, len(closes_1m)-5), len(closes_1m))]
        agreement = sum(1 for m in recent_moves if m == direction)
        signals["momentum_consistency"] = agreement / len(recent_moves) if recent_moves else 0.5

    if k5 and len(k5) >= 2:
        closes_5m = [float(c[4]) for c in k5]

        # Last close vs previous close = actual 5-minute price change
        signals["momentum_5m"] = (closes_5m[-1] - closes_5m[-2]) / closes_5m[-2] * 100

        returns_5m = [(closes_5m[i] - closes_5m[i-1]) / closes_5m[i-1]
                      for i in range(1, len(closes_5m))]
        signals["volatility_5m"] = calc_volatility(returns_5m[-5:])

        # Price acceleration: is momentum accelerating or decelerating?
        if len(closes_5m) >= 3:
            m_recent = (closes_5m[-1] - closes_5m[-2]) / closes_5m[-2] * 100
            m_prior = (closes_5m[-2] - closes_5m[-3]) / closes_5m[-3] * 100
            signals["price_acceleration"] = m_recent - m_prior

    k15 = fetch_binance_klines(symbol, "15m", 5)
    if k15 and len(k15) >= 2:
        closes_15m = [float(c[4]) for c in k15]
        # Last close vs previous close = actual 15-minute price change
        signals["momentum_15m"] = (closes_15m[-1] - closes_15m[-2]) / closes_15m[-2] * 100

    # Vol-adjusted momentum
    if signals.get("momentum_5m") and signals.get("volatility_5m") and signals["volatility_5m"] > 0:
        signals["vol_adjusted_momentum"] = signals["momentum_5m"] / signals["volatility_5m"]

    # Order book
    ob = fetch_binance_orderbook(symbol, 20)
    if ob and ob.get("bids") and ob.get("asks"):
        bids = ob["bids"]
        asks = ob["asks"]

        # Spread in bps
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2
        signals["spread_bps"] = ((best_ask - best_bid) / mid) * 10000

        # Order imbalance
        signals["order_imbalance"] = calc_order_imbalance(bids, asks)

    # Trade flow
    trades = fetch_binance_recent_trades(symbol, 500)
    if trades:
        signals["trade_flow_ratio"] = calc_trade_flow_ratio(trades)

    return signals


def extract_poly_signals(market_data):
    """
    Extract Polymarket signals from a market dict (Gamma events API format).

    Uses live fields directly from the API — no CLOB call needed:
      lastTradePrice  → poly_yes_price  (most current traded price)
      bestBid/bestAsk → poly_spread     (live order book spread)
      spread          → poly_spread     (fallback if bestBid/bestAsk absent)
      volumeClob      → poly_volume_24h
    """
    signals = {}
    if not market_data:
        return signals

    # Live price: lastTradePrice is the most current signal.
    # Fall back to bestBid/bestAsk midpoint, then outcomePrices (stale — avoid).
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
        signals["poly_yes_price"] = yes_price
        signals["poly_divergence"] = yes_price - 0.50

    # Spread: prefer bestBid/bestAsk, fall back to spread field
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

    # Volume
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

    # CEX/Poly lag: how much of the CEX signal is NOT yet in Polymarket pricing
    if cex.get("momentum_5m") is not None and poly.get("poly_divergence") is not None:
        derived["cex_poly_lag"] = cex["momentum_5m"] * (1.0 - poly["poly_divergence"] * 2)

    # BTC vs reference: (current_price - window_start_price) / window_start_price * 100
    # This is the direct answer to "is Bitcoin up or down since the window started?"
    # Positive = Up is currently winning. Most predictive signal available.
    if price_to_beat and cex.get("price_now"):
        derived["btc_vs_reference"] = (
            (cex["price_now"] - price_to_beat) / price_to_beat * 100
        )

    return derived


# ─────────────────────────────────────────────
# Outcome Resolution
# ─────────────────────────────────────────────

def resolve_outcomes(conn, symbol="BTCUSDT"):
    """
    For unresolved observations, check if market has resolved and fill in outcome.

    Resolution strategy (in priority order):
      1. market.winners field — explicit winner token label ("Up" / "Down")
      2. outcomePrices — YES token final price (>0.8 = up, <0.2 = down)
      3. outcomes array with winner flag
      4. Mark as 'unclear' if closed but can't determine direction

    Note: secs_remaining may be inflated (pre-created market bug).
    We ignore it and just check market age + closed flag.
    """
    cursor = conn.execute("""
        SELECT id, market_condition_id, market_slug, ts, seconds_remaining, price_now
        FROM signal_observations
        WHERE resolved = 0
        ORDER BY ts ASC LIMIT 100
    """)
    rows = cursor.fetchall()
    resolved_count = 0
    skipped_open = 0

    for row in rows:
        obs_id, cond_id, slug, ts_str, secs_remaining, price_at_obs = row
        if not slug and not cond_id:
            continue

        # Only attempt resolution if observation is at least 10 minutes old
        try:
            obs_time = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        age_seconds = (datetime.now(timezone.utc) - obs_time).total_seconds()
        if age_seconds < 600:
            skipped_open += 1
            continue

        # Prefer slug-based lookup via events endpoint — conditionId lookup returns
        # wrong markets (Polymarket API bug with fast-market condition IDs).
        market = None
        if slug:
            data = _get(f"{GAMMA_API}/events?slug={slug}")
            if data and isinstance(data, list) and data:
                mkts = data[0].get("markets") or []
                if mkts:
                    market = mkts[0]

        # Fallback to conditionId only if slug lookup failed
        if market is None and cond_id:
            market = fetch_poly_market(cond_id)

        if not market:
            continue

        closed = market.get("closed", False)
        if not closed:
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

    # Strategy 2: outcomePrices — winning token resolves to ~1.0
    try:
        prices = json.loads(market.get("outcomePrices", "[]"))
        if len(prices) >= 2:
            yes_price = float(prices[0])
            no_price = float(prices[1])
            if yes_price > 0.8:
                return "up"
            if no_price > 0.8:
                return "down"
            if yes_price > no_price and yes_price > 0.6:
                return "up"
            if no_price > yes_price and no_price > 0.6:
                return "down"
    except (ValueError, json.JSONDecodeError, IndexError):
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


# ─────────────────────────────────────────────
# Signal Analysis / Correlation Report
# ─────────────────────────────────────────────

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

        # Point-biserial correlation
        n = len(v_list)
        mean_v = sum(v_list) / n
        mean_o = sum(o_list) / n
        std_v = math.sqrt(sum((x - mean_v)**2 for x in v_list) / n) or 1e-9
        std_o = math.sqrt(sum((x - mean_o)**2 for x in o_list) / n) or 1e-9
        corr = sum((v_list[i] - mean_v) * (o_list[i] - mean_o) for i in range(n)) / (n * std_v * std_o)

        # Win rates by signal direction
        pos_wins = [o for v, o in paired if v > 0]
        neg_wins = [o for v, o in paired if v <= 0]
        wr_pos = sum(pos_wins) / len(pos_wins) if pos_wins else float('nan')
        wr_neg = sum(neg_wins) / len(neg_wins) if neg_wins else float('nan')
        edge = wr_pos - wr_neg

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
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {path}")


# ─────────────────────────────────────────────
# Collection Loop
# ─────────────────────────────────────────────

def _is_valid_market(gamma_mkt):
    """
    Validate a Gamma market record is current and tradeable.
    Only trusts endDate with a full ISO timestamp (contains 'T').
    Never uses createdAt — unreliable for recurring markets.
    """
    if not gamma_mkt:
        return False, "no gamma data"

    now = datetime.now(timezone.utc)

    # Try fields in order of reliability — endDate first, it has the full timestamp
    for key in ("endDate", "end_date"):
        raw = gamma_mkt.get(key)
        if raw and "T" in str(raw):
            try:
                end = datetime.fromisoformat(
                    str(raw).replace("Z", "+00:00")
                ).astimezone(timezone.utc)

                expired_secs = (now - end).total_seconds()
                if expired_secs > 600:
                    return False, f"market ended {expired_secs/60:.0f}m ago"

                future_secs = (end - now).total_seconds()
                if future_secs > 900:
                    return False, f"market resolves in {future_secs/60:.0f}m (too far ahead)"

                return True, "ok"
            except ValueError:
                continue

    return False, "no endDate with full timestamp found"


def collect_one(conn, asset="BTC", window="5m", symbol="BTCUSDT"):
    from fast_trader import discover_fast_market_markets, find_best_fast_market

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
    event_start = best.get("event_start")
    window_open = bool(event_start and now >= event_start)

    # Fetch CEX signals first — we need price_now to populate the reference cache
    cex = extract_cex_signals(symbol)
    poly = extract_poly_signals(best)

    # price_to_beat: our cached Binance price at first observation after window opens.
    # Polymarket's eventMetadata.priceToBeat is never set via the API, so we record
    # our own reference on the first cycle after startTime passes.
    price_to_beat = get_window_reference_price(
        best.get("slug", ""),
        cex_price_now=cex.get("price_now"),
        window_open=window_open,
    )

    derived = derive_combo_signals(cex, poly, price_to_beat)
    all_signals = {**cex, **poly, **derived}

    row = {
        "ts": now.isoformat(),
        "market_slug": best.get("slug", ""),
        "market_condition_id": cond_id,
        "seconds_remaining": seconds_remaining,
        "price_now": all_signals.get("price_now"),
        "price_to_beat": price_to_beat,
        **{k: all_signals.get(k) for k in SIGNAL_COLUMNS if k != "seconds_remaining"},
    }

    conn.execute("""
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
    conn.commit()

    m5 = all_signals.get("momentum_5m", 0) or 0
    poly_p = all_signals.get("poly_yes_price", 0.5) or 0.5
    oi = all_signals.get("order_imbalance", 0) or 0
    tfr = all_signals.get("trade_flow_ratio", 0.5) or 0.5
    rsi = all_signals.get("rsi_14") or 0
    vs_ref = all_signals.get("btc_vs_reference")
    vs_ref_str = f"{vs_ref:+.3f}%" if vs_ref is not None else "n/a"
    ptb_str = f"{price_to_beat:.2f}" if price_to_beat else "n/a"
    print(f"[{_now()}] {best['question'][:45]:45} | "
          f"m5={m5:+.3f}% poly={poly_p:.3f} vs_ref={vs_ref_str} "
          f"OI={oi:+.3f} TFR={tfr:.2f} RSI={rsi:.0f} ptb={ptb_str} secs={seconds_remaining:.0f}")


def _now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastLoop Signal Research")
    parser.add_argument("--collect", action="store_true",
                        help="Collect signals continuously (Ctrl-C to stop)")
    parser.add_argument("--analyze", action="store_true",
                        help="Show correlation report")
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve pending outcomes")
    parser.add_argument("--export", metavar="FILE",
                        help="Export observations to CSV")
    parser.add_argument("--min-n", type=int, default=20,
                        help="Min observations to show in report")
    parser.add_argument("--interval", type=int, default=30,
                        help="Collection interval in seconds (default: 30)")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--window", default="5m")
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