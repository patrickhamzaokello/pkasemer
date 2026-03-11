#!/usr/bin/env python3
"""
Pknwitq Composite Signal Engine  —  v2.2

Composite signal engine for Polymarket BTC fast markets.

Key additions in v2.2:
- Dynamic price bands based on lag + volatility
- Dynamic momentum threshold based on volatility_5m
- Cleaner price gate messaging
- Missing constants added
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime, timezone


# =============================================================================
# Time-of-Day Signal Accuracy
# =============================================================================

_DEFAULT_HOUR_ACCURACY = {
    0:  0.719,
    2:  0.857,
    4:  0.773,
    5:  0.667,
    6:  0.446,
    7:  0.662,
    8:  0.459,
    9:  0.525,
    10: 0.738,
    11: 0.629,
    12: 0.672,
    13: 0.656,
    14: 0.649,
    15: 0.688,
    16: 0.258,
    17: 0.733,
    18: 0.679,
    19: 0.688,
    21: 0.636,
    22: 0.667,
}

_DEFAULT_BLOCKED_HOURS = [6, 8, 16]

_DEFAULT_BOOSTED_HOURS = {
    2:  -0.03,
    4:  -0.02,
    10: -0.02,
    17: -0.02,
    0:  -0.01,
}


def get_hour_accuracy(hour_utc, config=None):
    cfg = config or {}
    table = {**_DEFAULT_HOUR_ACCURACY, **cfg.get("hour_accuracy", {})}
    return table.get(hour_utc, 0.60)


def check_hour_gate(hour_utc, config=None):
    cfg = config or {}
    blocked = cfg.get("blocked_hours", _DEFAULT_BLOCKED_HOURS)
    raw_boosted = cfg.get("boosted_hours", _DEFAULT_BOOSTED_HOURS)
    boosted = {int(k): v for k, v in raw_boosted.items()}

    if hour_utc in blocked:
        acc = get_hour_accuracy(hour_utc, cfg)
        return False, 0.0, f"hour {hour_utc}h blocked (signal accuracy={acc:.1%})"

    return True, boosted.get(hour_utc, 0.0), "ok"


# =============================================================================
# Market Session Detection (UTC)
# =============================================================================

_SESSION_HOURS = {
    "overlap":   [13, 14, 15],
    "london":    [7, 8, 9, 10, 11, 12],
    "new_york":  [16, 17, 18, 19, 20],
    "asia":      [0, 1, 2, 3, 4, 5, 6],
    "off":       [21, 22, 23],
}

_DEFAULT_SESSION_DELTAS = {
    "overlap":  -0.03,
    "london":   -0.02,
    "new_york": -0.01,
    "asia":      0.00,
    "off":      +0.02,
}

_DEFAULT_SESSION_CAPS = {
    "overlap":  1.00,
    "london":   0.95,
    "new_york": 0.90,
    "asia":     0.85,
    "off":      0.70,
}


def get_market_session(hour_utc, config=None):
    cfg = config or {}
    sess_hours = {**_SESSION_HOURS, **cfg.get("session_hours", {})}
    for session in ("overlap", "london", "new_york", "asia", "off"):
        if hour_utc in sess_hours.get(session, []):
            return session
    return "off"


def get_session_threshold_delta(session, config=None):
    cfg = config or {}
    return cfg.get(f"{session}_threshold_delta", _DEFAULT_SESSION_DELTAS.get(session, 0.0))


def get_session_position_cap(session, config=None):
    cfg = config or {}
    return cfg.get(f"{session}_position_cap", _DEFAULT_SESSION_CAPS.get(session, 0.85))


# =============================================================================
# Signal Weights
# =============================================================================

DEFAULT_WEIGHTS = {
    "vol_adjusted_momentum": 0.205,
    "momentum_5m":           0.170,
    "cex_poly_lag":          0.160,
    "btc_vs_reference":      0.150,
    "momentum_1m":           0.105,
    "momentum_15m":          0.085,
    "price_acceleration":    0.080,
    "rsi_14":                0.040,
    "poly_spread":           0.005,
    "volume_ratio":          0.000,
    "momentum_consistency":  0.000,
    "order_imbalance":       0.000,
    "trade_flow_ratio":      0.000,
}

SLOW_MARKET_WEIGHTS = {
    "btc_vs_reference":      0.290,
    "cex_poly_lag":          0.250,
    "momentum_5m":           0.140,
    "momentum_1m":           0.100,
    "momentum_15m":          0.080,
    "vol_adjusted_momentum": 0.080,
    "price_acceleration":    0.060,
    "rsi_14":                0.000,
    "poly_spread":           0.000,
    "volume_ratio":          0.000,
    "momentum_consistency":  0.000,
    "order_imbalance":       0.000,
    "trade_flow_ratio":      0.000,
}

ACTIVE_MARKET_WEIGHTS = {
    "vol_adjusted_momentum": 0.240,
    "btc_vs_reference":      0.210,
    "momentum_5m":           0.180,
    "momentum_15m":          0.110,
    "cex_poly_lag":          0.080,
    "momentum_1m":           0.070,
    "price_acceleration":    0.060,
    "rsi_14":                0.030,
    "poly_spread":           0.020,
    "volume_ratio":          0.000,
    "momentum_consistency":  0.000,
    "order_imbalance":       0.000,
    "trade_flow_ratio":      0.000,
}


# =============================================================================
# Hard-filter thresholds
# =============================================================================

COMPOSITE_ENTRY_THRESHOLD = 0.55
MIN_MOMENTUM_ABS          = 0.05
MAX_VOLATILITY_5M         = 2.0
MIN_POLY_SPREAD           = 0.04
RSI_OVERBOUGHT            = 80
RSI_OVERSOLD              = 22

# Dynamic price-band defaults
USE_DYNAMIC_PRICE_BANDS = True
DYNAMIC_LAG_WEIGHT      = 0.40
DYNAMIC_VOL_K           = 0.01
GLOBAL_MAX_YES          = 0.49
GLOBAL_MIN_NO           = 0.51

# Dynamic momentum-threshold defaults
USE_DYNAMIC_MOMENTUM_THRESHOLD = True
DYNAMIC_MOMENTUM_MULTIPLIER    = 1.0
MIN_DYNAMIC_MOMENTUM           = 0.010
MAX_DYNAMIC_MOMENTUM           = 0.120


# =============================================================================
# Regime Detection
# =============================================================================

def detect_market_regime(cex_signals, config=None):
    cfg = config or {}
    m5 = abs(cex_signals.get("momentum_5m") or 0)
    vr = cex_signals.get("volume_ratio") or 1.0
    vr_valid = vr != 1.0

    slow_mom = cfg.get("slow_mom_threshold", 0.12)
    slow_vol = cfg.get("slow_vol_threshold", 0.80)
    active_mom = cfg.get("active_mom_threshold", 0.25)
    active_vol = cfg.get("active_vol_threshold", 1.30)

    if m5 < slow_mom and (not vr_valid or vr < slow_vol):
        return "slow"
    if m5 > active_mom and vr_valid and vr > active_vol:
        return "active"
    return "normal"


# =============================================================================
# Normalization helpers
# =============================================================================

def _sigmoid(x, scale=1.0):
    return 1.0 / (1.0 + math.exp(-x * scale))


def normalize_momentum_5m(m, clip_pct=1.0):
    if m is None:
        return None
    return _sigmoid(max(-clip_pct, min(clip_pct, m)), scale=3.0)


def normalize_momentum_1m(m, clip_pct=0.3):
    if m is None:
        return None
    return _sigmoid(max(-clip_pct, min(clip_pct, m)), scale=8.0)


def normalize_momentum_15m(m, clip_pct=2.0):
    if m is None:
        return None
    return _sigmoid(max(-clip_pct, min(clip_pct, m)), scale=1.5)


def normalize_cex_poly_lag(lag):
    if lag is None:
        return None
    return _sigmoid(lag, scale=2.0)


def normalize_vol_adjusted_momentum(vam):
    if vam is None:
        return None
    return _sigmoid(vam, scale=1.0)


def normalize_btc_vs_reference(v):
    if v is None:
        return None
    return _sigmoid(max(-0.30, min(0.30, v)), scale=10.0)


def normalize_volume_ratio(vr):
    if vr is None:
        return None
    return _sigmoid(vr - 1.0, scale=2.0)


def normalize_price_acceleration(pa):
    if pa is None:
        return None
    return _sigmoid(pa, scale=8.0)


def normalize_order_imbalance(oi):
    if oi is None:
        return None
    return (oi + 1) / 2


def normalize_trade_flow(tfr):
    return tfr


def normalize_consistency(c):
    return c


def normalize_rsi(rsi):
    if rsi is None:
        return None
    return _sigmoid(rsi - 50, scale=0.08)


def normalize_poly_spread(spread):
    if spread is None:
        return None
    return _sigmoid(min(spread, MIN_POLY_SPREAD) - 0.015, scale=80.0)


NORMALIZERS = {
    "vol_adjusted_momentum": normalize_vol_adjusted_momentum,
    "momentum_5m":           normalize_momentum_5m,
    "cex_poly_lag":          normalize_cex_poly_lag,
    "btc_vs_reference":      normalize_btc_vs_reference,
    "momentum_1m":           normalize_momentum_1m,
    "momentum_15m":          normalize_momentum_15m,
    "price_acceleration":    normalize_price_acceleration,
    "rsi_14":                normalize_rsi,
    "poly_spread":           normalize_poly_spread,
    "volume_ratio":          normalize_volume_ratio,
    "momentum_consistency":  normalize_consistency,
    "order_imbalance":       normalize_order_imbalance,
    "trade_flow_ratio":      normalize_trade_flow,
}


# =============================================================================
# CEX/Poly Lag
# =============================================================================

def _calc_cex_poly_lag(cex, poly):
    btc_ref = cex.get("btc_vs_reference")
    poly_div = poly.get("poly_divergence", 0) or 0
    if btc_ref is None:
        return None
    return btc_ref - poly_div


# =============================================================================
# Dynamic helpers
# =============================================================================

def compute_dynamic_bounds(poly_price, lag, volatility, config=None):
    cfg = config or {}

    lag_weight = cfg.get("dynamic_lag_weight", DYNAMIC_LAG_WEIGHT)
    vol_k = cfg.get("dynamic_vol_k", DYNAMIC_VOL_K)

    lag = 0.0 if lag is None else lag
    volatility = 0.0 if volatility is None else volatility

    fair = poly_price + lag * lag_weight
    edge = abs(volatility) * vol_k

    max_yes = fair - edge
    min_no = fair + edge

    max_yes = min(max_yes, cfg.get("global_max_yes", GLOBAL_MAX_YES))
    min_no = max(min_no, cfg.get("global_min_no", GLOBAL_MIN_NO))

    return max_yes, min_no, fair, edge


def compute_dynamic_momentum_threshold(volatility, config=None):
    cfg = config or {}

    base = cfg.get("min_momentum_pct", MIN_MOMENTUM_ABS)
    multiplier = cfg.get("dynamic_momentum_multiplier", DYNAMIC_MOMENTUM_MULTIPLIER)
    min_floor = cfg.get("min_dynamic_momentum", MIN_DYNAMIC_MOMENTUM)
    max_cap = cfg.get("max_dynamic_momentum", MAX_DYNAMIC_MOMENTUM)

    if volatility is None:
        volatility = 1.0

    threshold = base * abs(volatility) * multiplier
    threshold = max(min_floor, min(threshold, max_cap))
    return threshold


# =============================================================================
# Pre-trade Filters
# =============================================================================

def apply_filters(cex, poly, config=None):
    cfg = config or {}
    m5 = cex.get("momentum_5m")
    vol5 = cex.get("volatility_5m")
    rsi = cex.get("rsi_14")
    vol_ratio = cex.get("volume_ratio", 1.0)
    poly_spread = poly.get("poly_spread")

    use_dynamic_mom = cfg.get(
        "use_dynamic_momentum_threshold",
        USE_DYNAMIC_MOMENTUM_THRESHOLD,
    )
    if use_dynamic_mom:
        min_mom = compute_dynamic_momentum_threshold(vol5, cfg)
    else:
        min_mom = cfg.get("min_momentum_pct", MIN_MOMENTUM_ABS)

    # Minimum momentum gate
    if m5 is not None and abs(m5) < min_mom:
        lag_mom_override = cfg.get("lag_momentum_override", False)
        allowed = False

        if lag_mom_override:
            lag = _calc_cex_poly_lag(cex, poly) or 0.0
            ref = cex.get("btc_vs_reference") or 0.0
            vr = cex.get("volume_ratio") or 1.0
            min_lag = cfg.get("min_lag_override", 0.12)
            min_ref = cfg.get("min_vs_ref_override", 0.20)
            same_dir = (lag > 0) == (ref > 0)
            vol_ok = vr != 1.0 and vr >= 0.5

            allowed = (
                abs(ref) >= min_ref
                and abs(lag) >= min_lag
                and same_dir
                and vol_ok
            )

        if not allowed:
            return False, f"momentum too weak ({m5:+.3f}% < ±{min_mom:.3f}%)"

    # Volatility gate
    max_vol_5m = cfg.get("max_volatility_5m", MAX_VOLATILITY_5M)
    if vol5 is not None and vol5 > max_vol_5m:
        return False, f"volatility too high ({vol5:.3f} > {max_vol_5m})"

    # RSI hard blocks
    rsi_ob = cfg.get("rsi_overbought", RSI_OVERBOUGHT)
    rsi_os = cfg.get("rsi_oversold", RSI_OVERSOLD)

    if rsi is not None:
        rsi_override = cfg.get("active_rsi_override", False)

        if m5 is not None and m5 > 0 and rsi > rsi_ob:
            bypassed = False
            if rsi_override:
                regime = detect_market_regime(cex, cfg)
                lag = _calc_cex_poly_lag(cex, poly) or 0.0
                min_lag = cfg.get("min_lag_override", 0.12)
                session = get_market_session(datetime.now(timezone.utc).hour, cfg)
                bypassed = (
                    regime == "active"
                    and abs(lag) >= min_lag
                    and session in ("overlap", "london")
                )
            if not bypassed:
                return False, f"RSI overbought ({rsi:.0f} > {rsi_ob}), skip long"

        if m5 is not None and m5 < 0 and rsi < rsi_os:
            bypassed = False
            if rsi_override:
                regime = detect_market_regime(cex, cfg)
                lag = _calc_cex_poly_lag(cex, poly) or 0.0
                min_lag = cfg.get("min_lag_override", 0.12)
                session = get_market_session(datetime.now(timezone.utc).hour, cfg)
                bypassed = (
                    regime == "active"
                    and abs(lag) >= min_lag
                    and session in ("overlap", "london")
                )
            if not bypassed:
                return False, f"RSI oversold ({rsi:.0f} < {rsi_os}), skip short"

    # Poly spread gate
    min_spread_block = cfg.get("min_poly_spread", MIN_POLY_SPREAD)
    if poly_spread is not None and poly_spread > min_spread_block:
        return False, f"Poly spread too wide ({poly_spread:.3f} > {min_spread_block})"

    # Dynamic / fixed price gates
    poly_price = poly.get("poly_yes_price", 0.5) or 0.5
    cex_lag = _calc_cex_poly_lag(cex, poly) or 0.0

    use_dynamic_price = cfg.get("use_dynamic_price_bands", USE_DYNAMIC_PRICE_BANDS)

    if use_dynamic_price:
        max_yes, min_no, fair_price, edge = compute_dynamic_bounds(
            poly_price,
            cex_lag,
            vol5,
            cfg,
        )
        yes_label = "dynamic"
        no_label = "dynamic"
    else:
        max_yes = cfg.get("max_entry_yes", 0.476)
        min_no = cfg.get("min_entry_no", 0.53)
        fair_price = None
        edge = None
        yes_label = "fixed"
        no_label = "fixed"

    if m5 is not None and m5 > 0:
        if poly_price > max_yes:
            return False, f"YES overpriced ({poly_price:.3f} > {yes_label} {max_yes:.3f})"

    if m5 is not None and m5 < 0:
        if poly_price < min_no:
            return False, f"NO overpriced ({poly_price:.3f} < {no_label} {min_no:.3f})"

    # Optional volume confidence gate
    volume_confidence = cfg.get("volume_confidence", False)
    if (
        volume_confidence
        and vol_ratio is not None
        and vol_ratio != 1.0
        and vol_ratio < 0.3
    ):
        return False, f"Volume too low ({vol_ratio:.2f}x avg)"

    # 1m vs 5m agreement
    if cfg.get("momentum_agreement_filter", True):
        m1 = cex.get("momentum_1m")
        min_m1 = cfg.get("momentum_agreement_min_1m", 0.05)
        if (
            m1 is not None
            and m5 is not None
            and abs(m1) >= min_m1
            and abs(m5) >= min_mom
            and (m1 > 0) != (m5 > 0)
        ):
            return False, (
                f"timeframe disagreement: 1m={m1:+.3f}% opposes 5m={m5:+.3f}%"
                f" — reversal risk, skipping"
            )

    # 15m vs 5m agreement
    if cfg.get("momentum_15m_agreement_filter", True):
        m15 = cex.get("momentum_15m")
        min_m15 = cfg.get("momentum_agreement_min_15m", 0.05)
        if (
            m15 is not None
            and m5 is not None
            and abs(m15) >= min_m15
            and abs(m5) >= min_mom
            and (m15 > 0) != (m5 > 0)
        ):
            return False, (
                f"timeframe disagreement: 15m={m15:+.3f}% opposes 5m={m5:+.3f}%"
                f" — counter-trend spike, skipping"
            )

    return True, "ok"


# =============================================================================
# Composite Scorer
# =============================================================================

def compute_composite_score(cex, poly, weights=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS

    signals = {
        "vol_adjusted_momentum": cex.get("vol_adjusted_momentum"),
        "momentum_5m":           cex.get("momentum_5m"),
        "cex_poly_lag":          _calc_cex_poly_lag(cex, poly),
        "btc_vs_reference":      cex.get("btc_vs_reference"),
        "momentum_1m":           cex.get("momentum_1m"),
        "momentum_15m":          cex.get("momentum_15m"),
        "price_acceleration":    cex.get("price_acceleration"),
        "rsi_14":                cex.get("rsi_14"),
        "poly_spread":           poly.get("poly_spread"),
        "volume_ratio":          cex.get("volume_ratio"),
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

    total_weight = sum(weights.get(k, 0) for k in normalized)
    if total_weight == 0:
        return 0.5, {}

    score = sum(normalized[k] * weights.get(k, 0) for k in normalized) / total_weight

    breakdown = {
        k: {
            "raw": signals.get(k),
            "normalized": normalized.get(k),
            "weight": weights.get(k, 0),
            "contribution": (
                normalized[k] * weights.get(k, 0) / total_weight
                if k in normalized else 0.0
            ),
        }
        for k in set(list(signals.keys()) + list(normalized.keys()))
    }

    return score, breakdown


# =============================================================================
# Main Interface
# =============================================================================

def get_composite_signal(cex_signals, poly_signals, config=None):
    cfg = config or {}
    regime = detect_market_regime(cex_signals, cfg)

    if "signal_weights" in cfg:
        weights = DEFAULT_WEIGHTS.copy()
        weights.update(cfg["signal_weights"])
    elif regime == "slow":
        weights = SLOW_MARKET_WEIGHTS.copy()
        if "slow_signal_weights" in cfg:
            weights.update(cfg["slow_signal_weights"])
    elif regime == "active":
        weights = ACTIVE_MARKET_WEIGHTS.copy()
        if "active_signal_weights" in cfg:
            weights.update(cfg["active_signal_weights"])
    else:
        weights = DEFAULT_WEIGHTS.copy()

    hour_utc = datetime.now(timezone.utc).hour
    hour_ok, hour_delta, hour_reason = check_hour_gate(hour_utc, cfg)
    session = get_market_session(hour_utc, cfg)
    lag_value = _calc_cex_poly_lag(cex_signals, poly_signals)

    result = {
        "should_trade": False,
        "side": None,
        "score": 0.5,
        "confidence": 0.0,
        "position_pct": 0.0,
        "filter_reason": None,
        "breakdown": {},
        "regime": regime,
        "session": session,
        "hour_utc": hour_utc,
        "hour_accuracy": get_hour_accuracy(hour_utc, cfg),
        "threshold_delta": hour_delta,
        "threshold_used": None,
        "lag_value": lag_value,
    }

    if not hour_ok:
        result["filter_reason"] = hour_reason
        return result

    passed, reason = apply_filters(cex_signals, poly_signals, config=cfg)
    if not passed:
        result["filter_reason"] = reason
        return result

    score, breakdown = compute_composite_score(cex_signals, poly_signals, weights)
    result["score"] = score
    result["breakdown"] = breakdown
    result["confidence"] = abs(score - 0.5) * 2

    base_threshold = cfg.get("composite_threshold", COMPOSITE_ENTRY_THRESHOLD)
    if regime == "slow":
        threshold = cfg.get("slow_composite_threshold", max(base_threshold, 0.73))
    elif regime == "active":
        threshold = cfg.get("active_composite_threshold", max(base_threshold - 0.02, 0.60))
    else:
        threshold = base_threshold

    session_delta = get_session_threshold_delta(session, cfg)
    threshold = max(0.55, threshold + hour_delta + session_delta)
    result["threshold_used"] = threshold

    if score > threshold:
        result["should_trade"] = True
        result["side"] = "yes"
    elif score < (1 - threshold):
        result["should_trade"] = True
        result["side"] = "no"
    else:
        result["filter_reason"] = (
            f"[{regime}] score {score:.3f} within neutral band "
            f"(threshold ±{threshold - 0.5:.3f})"
        )
        return result

    MIN_POSITION_PCT = 0.55

    if regime == "slow":
        size_cap = cfg.get("slow_position_pct_cap", 0.70)
    elif regime == "active":
        size_cap = cfg.get("active_position_pct_cap", 1.0)
    else:
        size_cap = cfg.get("normal_position_pct_cap", 1.0)

    result["position_pct"] = min(
        max(result["confidence"], MIN_POSITION_PCT),
        size_cap,
    )

    vol5 = cex_signals.get("volatility_5m")
    vp_thresh = cfg.get("vol_penalty_threshold", 1.2)
    if vol5 is not None and vol5 > vp_thresh:
        v_max = cfg.get("max_volatility_5m", MAX_VOLATILITY_5M)
        v_range = max(v_max - vp_thresh, 0.1)
        penalty = min(0.50, (vol5 - vp_thresh) / v_range)
        result["position_pct"] = max(
            MIN_POSITION_PCT,
            result["position_pct"] * (1.0 - penalty),
        )

    if cfg.get("session_gating_enabled", True):
        sess_cap = get_session_position_cap(session, cfg)
        result["position_pct"] = min(result["position_pct"], sess_cap)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composite signal engine v2.2")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--condition-id", default=None)
    parser.add_argument("--config", default=None, help="Path to config.json")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as fh:
            cfg = json.load(fh)

    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from signal_research import (
            extract_cex_signals,
            extract_poly_signals,
            fetch_poly_market,
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
        print("No --condition-id provided; using empty Poly signals")

    signal = get_composite_signal(cex, poly, config=cfg)

    W = 64
    print(f"\n{'─' * W}")
    print("  COMPOSITE SIGNAL  v2.2")
    print(f"{'─' * W}")
    print(f"  Score:          {signal['score']:.4f}   (0.5 = neutral)")
    if signal["threshold_used"] is not None:
        print(f"  Threshold used: {signal['threshold_used']:.4f}")
    print(f"  Confidence:     {signal['confidence']:.4f}   (0=neutral, 1=max)")
    print(f"  Should trade:   {signal['should_trade']}")
    print(f"  Side:           {signal['side'] or 'n/a'}")
    print(f"  Position pct:   {signal['position_pct']:.1%} of max size")
    print(f"  Regime:         {signal['regime']}")
    print(
        f"  Session:        {signal['session']}  "
        f"({signal['hour_utc']}h UTC, accuracy={signal['hour_accuracy']:.1%})"
    )
    if signal["lag_value"] is not None:
        print(f"  CEX-Poly lag:   {signal['lag_value']:+.4f}%")
    print(
        f"  Overrides:      RSI={'ON' if cfg.get('active_rsi_override', False) else 'off'}  "
        f"LagMomentum={'ON' if cfg.get('lag_momentum_override', False) else 'off'}"
    )
    if signal["filter_reason"]:
        print(f"  Blocked by:     {signal['filter_reason']}")

    print(f"\n  {'Signal':<26} {'Raw':>9}  {'Norm':>6}  {'Weight':>6}  {'Contrib':>8}")
    print(f"  {'─'*26}  {'─'*9}  {'─'*6}  {'─'*6}  {'─'*8}")
    for name, info in sorted(
        signal.get("breakdown", {}).items(),
        key=lambda x: -abs(x[1].get("weight", 0)),
    ):
        w = info.get("weight", 0)
        raw = info.get("raw")
        if w == 0 and raw is None:
            continue
        norm = info.get("normalized")
        con = info.get("contribution", 0)
        raw_s = f"{raw:+.4f}" if raw is not None else "     n/a"
        norm_s = f"{norm:.4f}" if norm is not None else "   n/a"
        print(f"  {name:<26} {raw_s:>9}  {norm_s:>6}  {w:>6.3f}  {con:>+8.4f}")

    print("\n  CEX signals:")
    for k in [
        "momentum_1m", "momentum_5m", "momentum_15m",
        "vol_adjusted_momentum", "btc_vs_reference",
        "price_acceleration", "rsi_14",
        "volume_ratio", "volatility_5m",
        "order_imbalance", "trade_flow_ratio",
    ]:
        v = cex.get(k)
        if v is not None:
            print(f"    {k:<28} {v:+.4f}")

    if poly:
        print("\n  Poly signals:")
        for k, v in poly.items():
            if v is not None:
                print(f"    {k:<28} {v:+.4f}")
