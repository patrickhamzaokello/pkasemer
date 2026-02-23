#!/usr/bin/env python3
"""
FastLoop Composite Signal Engine

Drop-in replacement for the single-momentum signal in fast_trader.py.
Combines multiple signal factors into a scored, directional trade decision.

Usage (standalone test):
    python composite_signal.py --symbol BTCUSDT --condition-id <id>

Integration:
    from composite_signal import get_composite_signal
    signal = get_composite_signal(asset="BTC", condition_id="0x...")
    if signal["should_trade"]:
        side = signal["side"]
        confidence = signal["confidence"]
        position_pct = signal["position_pct"]  # 0.0 - 1.0, scale your max size by this

Signal architecture:
  The composite score is a weighted sum of normalized sub-signals.
  Weights start as equal, but can be overridden via config.json "signal_weights".
  After running signal_research.py --analyze, update weights based on actual
  correlation data to improve accuracy over time.

Weights calibrated from 362 resolved observations (2026-02-24):
  btc_vs_reference      0.40   (price_now - ref) / ref — corr +0.39, edge +0.26
  volume_ratio          0.15   elevated volume predicts direction — edge +0.22
  momentum_5m           0.15   5m price momentum — corr +0.17, edge +0.18
  vol_adjusted_momentum 0.10   momentum per unit vol — corr +0.19, edge +0.17
  cex_poly_lag          0.10   CEX vs Poly repricing gap — corr +0.17
  price_acceleration    0.05   momentum of momentum — corr +0.23
  momentum_consistency  0.05   candle direction agreement
  trade_flow_ratio      0.00   negative edge in data (-0.38) — excluded
  order_imbalance       0.00   no edge in data (+0.01) — excluded
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ─────────────────────────────────────────────
# Default signal weights
# Override via config.json: { "signal_weights": { "order_imbalance": 0.35, ... } }
# ─────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "btc_vs_reference":      0.40,  # corr +0.39, edge +0.26 — dominant signal
    "volume_ratio":          0.15,  # edge +0.22 — elevated volume predicts direction
    "momentum_5m":           0.15,  # corr +0.17, edge +0.18
    "vol_adjusted_momentum": 0.10,  # corr +0.19, edge +0.17
    "cex_poly_lag":          0.10,  # corr +0.17, edge +0.18
    "price_acceleration":    0.05,  # corr +0.23
    "momentum_consistency":  0.05,  # corr +0.06
    "trade_flow_ratio":      0.00,  # edge -0.38 in data — excluded
    "order_imbalance":       0.00,  # edge +0.01 in data — excluded
}

# Thresholds
COMPOSITE_ENTRY_THRESHOLD = 0.55   # composite score must exceed this to signal a trade
                                   # (0.5 = no edge, 1.0 = maximum conviction)
MIN_MOMENTUM_ABS = 0.05            # minimum |momentum_5m| to consider (filter noise)
MAX_VOLATILITY_5M = 2.0            # skip if volatility too high (chaotic market)
MIN_POLY_SPREAD = 0.10             # skip if Polymarket spread is very wide (illiquid)
RSI_OVERBOUGHT = 72                # avoid chasing already-overbought
RSI_OVERSOLD = 28                  # avoid chasing already-oversold

# ─────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────

def _sigmoid(x, scale=1.0):
    """Squash any value to (0, 1). scale controls steepness."""
    return 1.0 / (1.0 + math.exp(-x * scale))


def normalize_order_imbalance(oi):
    """
    OI is already in (-1, 1).
    Map to (0, 1) where 1 = all bids (bullish), 0 = all asks (bearish).
    """
    if oi is None:
        return None
    return (oi + 1) / 2


def normalize_trade_flow(tfr):
    """
    TFR is fraction of buy volume (0-1).
    Already in right range, 0.5 = neutral, > 0.5 = bullish.
    """
    if tfr is None:
        return None
    return tfr


def normalize_momentum(m5, clip_pct=1.0):
    """
    5m momentum in percent. Clip to ±clip_pct, then sigmoid to (0,1).
    Positive = bullish.
    """
    if m5 is None:
        return None
    clipped = max(-clip_pct, min(clip_pct, m5))
    return _sigmoid(clipped, scale=3.0)


def normalize_cex_poly_lag(lag):
    """
    CEX/poly lag: positive = CEX bullish but Poly hasn't repriced.
    Map via sigmoid.
    """
    if lag is None:
        return None
    return _sigmoid(lag, scale=2.0)


def normalize_consistency(c):
    """Already in (0,1). > 0.5 means more recent candles agree with direction."""
    return c


def normalize_vol_adjusted_momentum(vam):
    """Momentum per unit vol. Sigmoid-compress."""
    if vam is None:
        return None
    return _sigmoid(vam, scale=1.0)


def normalize_btc_vs_reference(v):
    """
    (current_btc_price - priceToBeat) / priceToBeat * 100
    Positive = Up is currently winning. Clip to ±0.30%, then sigmoid.
    At ±0.10% → ~0.73/0.27. At ±0.30% → ~0.95/0.05.
    """
    if v is None:
        return None
    clipped = max(-0.30, min(0.30, v))
    return _sigmoid(clipped, scale=10.0)


def normalize_volume_ratio(vr):
    """
    volume_ratio = latest_candle_vol / avg_vol. Centered at 1.0 (neutral).
    > 1.0 = above-average volume = bullish signal (edge +0.22 in data).
    At 1.5x → ~0.73. At 0.5x → ~0.27.
    """
    if vr is None:
        return None
    return _sigmoid(vr - 1.0, scale=2.0)


def normalize_price_acceleration(pa):
    """
    price_acceleration = recent_5m_momentum - prior_5m_momentum (in %).
    Positive = momentum increasing = bullish. Clip implicitly via sigmoid.
    """
    if pa is None:
        return None
    return _sigmoid(pa, scale=8.0)


NORMALIZERS = {
    "btc_vs_reference":      normalize_btc_vs_reference,
    "volume_ratio":          normalize_volume_ratio,
    "momentum_5m":           normalize_momentum,
    "vol_adjusted_momentum": normalize_vol_adjusted_momentum,
    "cex_poly_lag":          normalize_cex_poly_lag,
    "price_acceleration":    normalize_price_acceleration,
    "momentum_consistency":  normalize_consistency,
    "order_imbalance":       normalize_order_imbalance,
    "trade_flow_ratio":      normalize_trade_flow,
}


# ─────────────────────────────────────────────
# Pre-trade filters (hard gates before scoring)
# ─────────────────────────────────────────────

def apply_filters(cex, poly):
    """
    Returns (passed: bool, reason: str).
    Hard filters eliminate clearly bad conditions regardless of composite score.
    """
    m5 = cex.get("momentum_5m")
    vol5 = cex.get("volatility_5m")
    rsi = cex.get("rsi_14")
    vol_ratio = cex.get("volume_ratio", 1.0)
    poly_spread = poly.get("poly_spread")
    poly_oi = poly.get("poly_order_imbalance")

    # Momentum too weak — pure noise
    if m5 is not None and abs(m5) < MIN_MOMENTUM_ABS:
        return False, f"momentum too weak ({m5:+.3f}% < ±{MIN_MOMENTUM_ABS}%)"

    # Market too volatile — model unreliable
    if vol5 is not None and vol5 > MAX_VOLATILITY_5M:
        return False, f"volatility too high ({vol5:.3f} > {MAX_VOLATILITY_5M})"

    # RSI extremes — mean reversion likely, don't chase
    if rsi is not None:
        if m5 and m5 > 0 and rsi > RSI_OVERBOUGHT:
            return False, f"RSI overbought ({rsi:.0f} > {RSI_OVERBOUGHT}), skip long"
        if m5 and m5 < 0 and rsi < RSI_OVERSOLD:
            return False, f"RSI oversold ({rsi:.0f} < {RSI_OVERSOLD}), skip short"

    # Polymarket spread too wide — execution drag too high
    if poly_spread is not None and poly_spread > MIN_POLY_SPREAD:
        return False, f"Poly spread too wide ({poly_spread:.3f} > {MIN_POLY_SPREAD})"

    # Polymarket already priced in the move — data shows edge -0.29 once divergence > 0.08
    poly_divergence = poly.get("poly_divergence", 0)
    if m5 and m5 > 0 and poly_divergence > 0.08:
        return False, f"Poly already priced bullish ({poly_divergence:+.3f}), no edge left"
    if m5 and m5 < 0 and poly_divergence < -0.08:
        return False, f"Poly already priced bearish ({poly_divergence:+.3f}), no edge left"

    # Low volume — signal unreliable
    if vol_ratio is not None and vol_ratio < 0.3:
        return False, f"Volume too low ({vol_ratio:.2f}x avg)"

    return True, "ok"


# ─────────────────────────────────────────────
# Composite Scorer
# ─────────────────────────────────────────────

def compute_composite_score(cex, poly, weights=None):
    """
    Compute a composite bullishness score in (0, 1).
    > 0.5 = bullish signal, < 0.5 = bearish signal.
    Returns (score, breakdown dict).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    signals = {
        "btc_vs_reference":      cex.get("btc_vs_reference"),
        "volume_ratio":          cex.get("volume_ratio"),
        "momentum_5m":           cex.get("momentum_5m"),
        "vol_adjusted_momentum": cex.get("vol_adjusted_momentum"),
        "cex_poly_lag":          _calc_cex_poly_lag(cex, poly),
        "price_acceleration":    cex.get("price_acceleration"),
        "momentum_consistency":  cex.get("momentum_consistency"),
        "order_imbalance":       cex.get("order_imbalance"),
        "trade_flow_ratio":      cex.get("trade_flow_ratio"),
    }

    normalized = {}
    for name, val in signals.items():
        if val is None:
            continue
        fn = NORMALIZERS.get(name)
        if fn:
            n = fn(val)
            if n is not None:
                normalized[name] = n

    if not normalized:
        return 0.5, {}

    # Weighted average of available signals
    total_weight = sum(weights.get(k, 0) for k in normalized)
    if total_weight == 0:
        return 0.5, {}

    score = sum(normalized[k] * weights.get(k, 0) for k in normalized) / total_weight

    breakdown = {
        k: {"raw": signals.get(k), "normalized": normalized.get(k), "weight": weights.get(k, 0)}
        for k in set(list(signals.keys()) + list(normalized.keys()))
    }

    return score, breakdown


def _calc_cex_poly_lag(cex, poly):
    m5 = cex.get("momentum_5m")
    poly_div = poly.get("poly_divergence", 0)
    if m5 is None:
        return None
    # How much of the CEX signal is NOT yet reflected in Poly?
    # If m5=+1% and poly_div=+0.02 (barely moved), lag = large positive
    # If m5=+1% and poly_div=+0.15 (already priced in), lag = small
    return m5 * (0.15 - max(-0.15, min(0.15, poly_div))) / 0.15


# ─────────────────────────────────────────────
# Main Interface
# ─────────────────────────────────────────────

def get_composite_signal(cex_signals, poly_signals, config=None):
    """
    Main entry point. Takes pre-fetched CEX and Poly signal dicts.

    Returns dict:
        should_trade: bool
        side: 'yes' | 'no' | None
        score: float (0-1, > 0.5 = bullish)
        confidence: float (0-1, how far score is from 0.5)
        position_pct: float (0-1, suggested fraction of max position)
        filter_reason: str (why trade was blocked, if applicable)
        breakdown: dict (per-signal contributions)
    """
    weights = DEFAULT_WEIGHTS
    if config and "signal_weights" in config:
        weights = {**DEFAULT_WEIGHTS, **config["signal_weights"]}

    result = {
        "should_trade": False,
        "side": None,
        "score": 0.5,
        "confidence": 0.0,
        "position_pct": 0.0,
        "filter_reason": None,
        "breakdown": {},
    }

    # Hard filters first
    passed, reason = apply_filters(cex_signals, poly_signals)
    if not passed:
        result["filter_reason"] = reason
        return result

    # Compute composite score
    score, breakdown = compute_composite_score(cex_signals, poly_signals, weights)
    result["score"] = score
    result["breakdown"] = breakdown

    # Confidence = how far from neutral (0.5)
    confidence = abs(score - 0.5) * 2   # maps [0.5, 1.0] → [0, 1]
    result["confidence"] = confidence

    # Threshold check
    threshold = COMPOSITE_ENTRY_THRESHOLD
    if config:
        threshold = config.get("composite_threshold", threshold)

    if score > threshold:
        result["should_trade"] = True
        result["side"] = "yes"
    elif score < (1 - threshold):
        result["should_trade"] = True
        result["side"] = "no"
    else:
        result["filter_reason"] = f"score {score:.3f} within neutral band (threshold ±{threshold - 0.5:.3f})"

    # Position sizing: scale linearly with confidence, squared to be conservative
    # confidence=0.5 → 25% of max, confidence=1.0 → 100% of max
    result["position_pct"] = confidence ** 2

    return result


# ─────────────────────────────────────────────
# Standalone CLI for testing
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test composite signal")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--condition-id", default=None)
    args = parser.parse_args()

    # Import data fetchers from research module
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from signal_research import (
            extract_cex_signals, extract_poly_signals, fetch_poly_market
        )
    except ImportError:
        print("Error: signal_research.py not found in same directory")
        sys.exit(1)

    print(f"Fetching CEX signals for {args.symbol}...")
    cex = extract_cex_signals(args.symbol)

    poly = {}
    if args.condition_id:
        print(f"Fetching Poly signals for {args.condition_id}...")
        gamma_mkt = fetch_poly_market(args.condition_id)
        poly = extract_poly_signals(gamma_mkt) if gamma_mkt else {}
    else:
        print("No --condition-id provided, using empty Poly signals")

    signal = get_composite_signal(cex, poly)

    print(f"\n{'─'*55}")
    print(f"  COMPOSITE SIGNAL RESULT")
    print(f"{'─'*55}")
    print(f"  Score:        {signal['score']:.4f}  (0.5 = neutral)")
    print(f"  Confidence:   {signal['confidence']:.4f}  (0 = neutral, 1 = max)")
    print(f"  Should trade: {signal['should_trade']}")
    print(f"  Side:         {signal['side'] or 'n/a'}")
    print(f"  Position pct: {signal['position_pct']:.1%} of max size")
    if signal["filter_reason"]:
        print(f"  Blocked by:   {signal['filter_reason']}")

    print(f"\n  Signal breakdown:")
    for name, info in signal.get("breakdown", {}).items():
        raw = info.get("raw")
        norm = info.get("normalized")
        w = info.get("weight", 0)
        raw_str = f"{raw:+.4f}" if raw is not None else "   n/a"
        norm_str = f"{norm:.4f}" if norm is not None else "  n/a"
        print(f"    {name:<28} raw={raw_str}  norm={norm_str}  w={w:.2f}")

    print(f"\n  CEX signals:")
    for k in ["momentum_1m", "momentum_5m", "momentum_15m", "rsi_14",
              "volume_ratio", "order_imbalance", "trade_flow_ratio",
              "volatility_5m", "price_acceleration"]:
        v = cex.get(k)
        if v is not None:
            print(f"    {k:<28} {v:+.4f}")

    if poly:
        print(f"\n  Poly signals:")
        for k, v in poly.items():
            if v is not None:
                print(f"    {k:<28} {v:+.4f}")