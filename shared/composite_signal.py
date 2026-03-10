#!/usr/bin/env python3
"""
Pknwitq Composite Signal Engine

Drop-in replacement for the single-momentum signal in fast_trader.py.
Combines multiple signal factors into a scored, directional trade decision.

Usage (standalone test):
    python composite_signal.py --symbol BTCUSDT --condition-id <id>

Integration:
    from composite_signal import get_composite_signal
    signal = get_composite_signal(cex_signals, poly_signals, config=cfg)
    if signal["should_trade"]:
        side         = signal["side"]
        confidence   = signal["confidence"]
        position_pct = signal["position_pct"]  # 0.0-1.0, scale your max size by this

Signal architecture:
  The composite score is a weighted sum of normalized sub-signals.
  Weights start calibrated from observed data (see below).
  After running signal_research.py --analyze, update DEFAULT_WEIGHTS based on
  actual correlation data to improve accuracy over time.
  Override via config.json "signal_weights" key.

Weights calibrated from 362 resolved observations (2026-02-24):
  btc_vs_reference      0.45   (price_now - ref) / ref — corr +0.39, edge +0.26
  vol_adjusted_momentum 0.22   momentum per unit vol   — corr +0.19, edge +0.17
  price_acceleration    0.13   momentum of momentum    — corr +0.23
  momentum_15m          0.12   15m trend confirmation
  momentum_1m           0.08   very recent micro-move
  momentum_5m           0.00   redundant given btc_vs_reference
  cex_poly_lag          0.00   redundant with momentum_5m when poly_div≈0
  trade_flow_ratio      0.00   negative edge in data (-0.38) — excluded
  order_imbalance       0.00   no edge in data (+0.01) — excluded
  volume_ratio          0.00   moved into pre-filter gate rather than scorer

Changes from original:
  - apply_filters: `if m5 and ...` replaced with `if m5 is not None and ...`
    to prevent truthiness check failing on small-but-valid momentum values.
  - MIN_POLY_SPREAD corrected from 0.10 to 0.04 — the old value never fired
    since real fast-market spreads are 0.01-0.03; 0.04 actually catches
    illiquid windows.
  - normalize_momentum_1m / normalize_momentum_15m are now separate from
    normalize_momentum (5m). Each timeframe has its own clip range and sigmoid
    scale appropriate to typical move magnitudes at that horizon.
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime, timezone

# =============================================================================
# Time-of-Day Signal Accuracy (calibrated from 4,647 resolved observations)
#
# Signal accuracy = fraction of times the momentum direction correctly predicted
# the market outcome, filtered to |momentum_5m| >= 0.08%.
#
# Override via config.json "hour_accuracy" dict (keyed by int 0-23).
# "blocked_hours"   — list of UTC hours where trading is disabled entirely
# "boosted_hours"   — dict {hour: threshold_delta} to lower entry threshold
# =============================================================================

# Observed signal accuracy by UTC hour (n>=20 filter applied)
_DEFAULT_HOUR_ACCURACY = {
    0:  0.719,
    2:  0.857,   # best hour — 85.7% accuracy
    4:  0.773,
    5:  0.667,
    6:  0.446,   # anti-predictive — avoid
    7:  0.662,
    8:  0.459,   # anti-predictive — avoid
    9:  0.525,
    10: 0.738,
    11: 0.629,
    12: 0.672,
    13: 0.656,
    14: 0.649,
    15: 0.688,
    16: 0.258,   # STRONGLY anti-predictive — block
    17: 0.733,
    18: 0.679,
    19: 0.688,
    21: 0.636,
    22: 0.667,
}

# Default hour-based rules derived from the accuracy table
# Blocked: accuracy below 0.50 (signal is net-negative)
_DEFAULT_BLOCKED_HOURS = [6, 8, 16]

# Boosted: accuracy >= 0.73 → allow slightly lower composite threshold
# delta is subtracted from the threshold (negative = easier to trigger)
_DEFAULT_BOOSTED_HOURS = {
    2:  -0.03,   # 85.7% — lower threshold by 0.03
    4:  -0.02,   # 77.3%
    10: -0.02,   # 73.8%
    17: -0.02,   # 73.3%
    0:  -0.01,   # 71.9%
}


def get_hour_accuracy(hour_utc, config=None):
    """Return the observed signal accuracy for the given UTC hour (0-23)."""
    cfg = config or {}
    table = {**_DEFAULT_HOUR_ACCURACY, **cfg.get("hour_accuracy", {})}
    return table.get(hour_utc, 0.60)   # 0.60 = conservative default for unseen hours


def check_hour_gate(hour_utc, config=None):
    """
    Returns (allowed: bool, threshold_delta: float, reason: str).

    allowed=False  → skip this cycle entirely (bad hours)
    threshold_delta → added to composite_threshold (negative = easier to trade)
    """
    cfg = config or {}
    blocked = cfg.get("blocked_hours", _DEFAULT_BLOCKED_HOURS)
    # JSON loads dict keys as strings; normalise to int
    raw_boosted = cfg.get("boosted_hours", _DEFAULT_BOOSTED_HOURS)
    boosted = {int(k): v for k, v in raw_boosted.items()}

    if hour_utc in blocked:
        acc = get_hour_accuracy(hour_utc, cfg)
        return False, 0.0, f"hour {hour_utc}h blocked (signal accuracy={acc:.1%})"

    delta = boosted.get(hour_utc, 0.0)
    return True, delta, "ok"


# =============================================================================
# Market Session Detection (UTC)
#
# Polymarket BTC fast markets are directly influenced by traditional finance
# liquidity windows, even though crypto trades 24/7.  The three major sessions:
#
#   Asia      00:00-07:00 UTC  — Tokyo / Singapore
#   London    07:00-16:00 UTC  — European open (07:00-13:00 pre-overlap)
#   New York  13:00-21:00 UTC  — US open (13:00-16:00 overlaps London)
#   Off       21:00-00:00 UTC  — Post-NY quiet period
#
# The London-NY overlap (13:00-16:00 UTC) is the highest-liquidity window of
# the day: institutional order flow from both continents hits simultaneously.
# Momentum signals are most reliable here — lower the threshold, allow full size.
#
# Override via config.json "session_hours" dict (session_name → list of hours).
# =============================================================================

_SESSION_HOURS = {
    "overlap":   [13, 14, 15],              # London + NY simultaneously
    "london":    [7, 8, 9, 10, 11, 12],     # London only, pre-overlap
    "new_york":  [16, 17, 18, 19, 20],      # NY only, post London close
    "asia":      [0, 1, 2, 3, 4, 5, 6],     # Asia / Tokyo
    "off":       [21, 22, 23],              # Post-NY quiet period
}

# Composite threshold delta per session (negative = easier to trade)
_DEFAULT_SESSION_DELTAS = {
    "overlap":  -0.03,   # peak liquidity — most reliable signals
    "london":   -0.02,   # strong European institutional flow
    "new_york": -0.01,   # solid US session
    "asia":      0.00,   # mixed — per-hour accuracy table handles fine-tuning
    "off":      +0.02,   # low liquidity, be more selective
}

# Position-size cap per session (fraction of regime-adjusted max)
_DEFAULT_SESSION_CAPS = {
    "overlap":  1.00,   # full size during peak liquidity
    "london":   0.95,
    "new_york": 0.90,
    "asia":     0.85,
    "off":      0.70,   # protect capital in quiet hours
}


def get_market_session(hour_utc, config=None):
    """
    Return the primary market session label for a given UTC hour (0-23).

    Priority order: overlap > london > new_york > asia > off
    Override session hour lists via config.json "session_hours" dict.
    Returns one of: "overlap" | "london" | "new_york" | "asia" | "off"
    """
    cfg = config or {}
    sess_hours = {**_SESSION_HOURS, **{k: v for k, v in cfg.get("session_hours", {}).items()}}
    for session in ("overlap", "london", "new_york", "asia", "off"):
        if hour_utc in sess_hours.get(session, []):
            return session
    return "off"


def get_session_threshold_delta(session, config=None):
    """
    Return the composite threshold adjustment for the given session.
    Negative = lower threshold = easier to trigger a trade.
    Config key: <session>_threshold_delta  (e.g. "london_threshold_delta": -0.02)
    """
    cfg = config or {}
    return cfg.get(f"{session}_threshold_delta", _DEFAULT_SESSION_DELTAS.get(session, 0.0))


def get_session_position_cap(session, config=None):
    """
    Return the position-size cap (0.0-1.0) for the given session.
    Config key: <session>_position_cap  (e.g. "overlap_position_cap": 1.0)
    """
    cfg = config or {}
    return cfg.get(f"{session}_position_cap", _DEFAULT_SESSION_CAPS.get(session, 0.85))


# =============================================================================
# Default signal weights
# Override via config.json: { "signal_weights": { "btc_vs_reference": 0.50, ... } }
# =============================================================================

DEFAULT_WEIGHTS = {
    "btc_vs_reference":      0.315,   # dominant signal, corr=0.263
    "vol_adjusted_momentum": 0.194,   # corr=0.206
    "momentum_5m":           0.136,   # corr=0.173
    "cex_poly_lag":          0.136,   # corr=0.172 -- the arb signal
    "momentum_1m":           0.111,   # corr=0.156, was underweighted
    "price_acceleration":    0.074,   # corr=0.098
    "momentum_15m":          0.034,   # corr=0.071, trend confirmation only
    "volume_ratio":          0.000,   # handled as pre-filter, not scorer
    "trade_flow_ratio":      0.000,   # corr=0.051, noise
    "order_imbalance":       0.000,   # corr=0.034, near zero
    "momentum_consistency":  0.000,   # corr=-0.021, negative
}

# Slow-market weights: shift to latency-arb signals which remain reliable
# when momentum is weak. De-emphasise vol_adjusted_momentum (noisy at low vol).
SLOW_MARKET_WEIGHTS = {
    "cex_poly_lag":          0.340,   # pure latency arb — most reliable in quiet markets
    "btc_vs_reference":      0.290,   # structural reference-price edge
    "momentum_5m":           0.130,   # reduced — less reliable in consolidation
    "vol_adjusted_momentum": 0.100,   # de-weighted — noisy when vol is low
    "momentum_1m":           0.080,   # micro-move confirmation
    "price_acceleration":    0.060,   # acceleration still informative
    "momentum_15m":          0.000,   # 15m trend useless in a flat market
    "volume_ratio":          0.000,
    "trade_flow_ratio":      0.000,
    "order_imbalance":       0.000,
    "momentum_consistency":  0.000,
}

# Active-market weights: lean on strong multi-timeframe momentum alignment
ACTIVE_MARKET_WEIGHTS = {
    "btc_vs_reference":      0.280,
    "vol_adjusted_momentum": 0.250,   # momentum per unit vol very reliable in trends
    "momentum_5m":           0.180,
    "momentum_15m":          0.120,   # 15m confirmation matters in trending markets
    "cex_poly_lag":          0.080,   # lag signal less distinctive when poly already moved
    "momentum_1m":           0.050,
    "price_acceleration":    0.040,
    "volume_ratio":          0.000,
    "trade_flow_ratio":      0.000,
    "order_imbalance":       0.000,
    "momentum_consistency":  0.000,
}

# =============================================================================
# Thresholds
# =============================================================================

COMPOSITE_ENTRY_THRESHOLD = 0.55   # score must exceed this to signal a trade
                                   # (0.5 = no edge, 1.0 = maximum conviction)
MIN_MOMENTUM_ABS   = 0.05          # minimum |momentum_5m| to consider (filter noise)
MAX_VOLATILITY_5M  = 2.0           # skip if 5m volatility too high (chaotic market)
MIN_POLY_SPREAD    = 0.04          # FIX: was 0.10 — never fired since real spreads
                                   # are 0.01-0.03. Now correctly catches illiquid windows.
RSI_OVERBOUGHT     = 85            # only block extreme exhaustion spikes
RSI_OVERSOLD       = 15            # only block extreme capitulation


# =============================================================================
# Market Regime Detection
# =============================================================================

def detect_market_regime(cex_signals, config=None):
    """
    Classify the current market as 'slow', 'normal', or 'active'.

    Slow   — consolidating / low-momentum. Best edge: pure latency arb.
    Normal — typical conditions. Default strategy applies.
    Active — strong trending. Momentum signals most reliable.

    Thresholds are config-overridable:
        slow_mom_threshold   default 0.12  — |momentum_5m| below this = slow
        slow_vol_threshold   default 0.80  — volume_ratio below this = slow
        active_mom_threshold default 0.25  — |momentum_5m| above this = active
        active_vol_threshold default 1.30  — volume_ratio above this = active
    """
    cfg = config or {}
    m5  = abs(cex_signals.get("momentum_5m") or 0)
    vr  = cex_signals.get("volume_ratio") or 1.0
    # volume_ratio == 1.0 is the "no data yet" sentinel; treat as neutral
    vr_valid = vr != 1.0

    slow_mom   = cfg.get("slow_mom_threshold",   0.12)
    slow_vol   = cfg.get("slow_vol_threshold",   0.80)
    active_mom = cfg.get("active_mom_threshold", 0.25)
    active_vol = cfg.get("active_vol_threshold", 1.30)

    # Slow: weak momentum AND (below-average volume or no volume data)
    if m5 < slow_mom and (not vr_valid or vr < slow_vol):
        return "slow"

    # Active: strong momentum AND above-average volume
    if m5 > active_mom and vr_valid and vr > active_vol:
        return "active"

    return "normal"


# =============================================================================
# Normalization helpers
# =============================================================================

def _sigmoid(x, scale=1.0):
    """Squash any value to (0, 1). scale controls steepness."""
    return 1.0 / (1.0 + math.exp(-x * scale))


def normalize_order_imbalance(oi):
    """OI is already in (-1, 1). Map to (0, 1) where 1=all bids (bullish)."""
    if oi is None:
        return None
    return (oi + 1) / 2


def normalize_trade_flow(tfr):
    """TFR is fraction of buy volume (0-1). 0.5=neutral, >0.5=bullish."""
    if tfr is None:
        return None
    return tfr


def normalize_momentum_5m(m, clip_pct=1.0):
    """
    5m momentum in percent. Clip to ±1.0%, then sigmoid to (0,1).
    A ±1% move over 5m is large; clip prevents extreme outliers dominating.
    
    """
    if m is None:
        return None
    clipped = max(-clip_pct, min(clip_pct, m))
    return _sigmoid(clipped, scale=3.0)


def normalize_momentum_1m(m, clip_pct=0.3):
    """
    1m momentum in percent. Clip to ±0.3% (1m moves are much smaller).
    Steeper sigmoid (scale=8) to amplify signal from small moves.
    """
    if m is None:
        return None
    clipped = max(-clip_pct, min(clip_pct, m))
    return _sigmoid(clipped, scale=8.0)


def normalize_momentum_15m(m, clip_pct=2.0):
    """
    15m momentum in percent. Clip to ±2.0% (15m moves are much larger).
    Shallower sigmoid (scale=1.5) because even large 15m moves should not
    completely dominate — they are trend confirmation, not entry signal.
    """
    if m is None:
        return None
    clipped = max(-clip_pct, min(clip_pct, m))
    return _sigmoid(clipped, scale=1.5)


def normalize_cex_poly_lag(lag):
    """CEX/poly lag: positive = CEX bullish but Poly hasn't repriced."""
    if lag is None:
        return None
    return _sigmoid(lag, scale=2.0)


def normalize_consistency(c):
    """Already in (0,1). >0.5 means more recent candles agree with direction."""
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
    volume_ratio = latest_candle_vol / avg_vol. Centered at 1.0.
    >1.0 = above-average volume. At 1.5x → ~0.73. At 0.5x → ~0.27.
    """
    if vr is None:
        return None
    return _sigmoid(vr - 1.0, scale=2.0)


def normalize_price_acceleration(pa):
    """
    price_acceleration = recent_5m_momentum - prior_5m_momentum (in %).
    Positive = momentum increasing = bullish.
    """
    if pa is None:
        return None
    return _sigmoid(pa, scale=8.0)


NORMALIZERS = {
    "btc_vs_reference":      normalize_btc_vs_reference,
    "volume_ratio":          normalize_volume_ratio,
    "momentum_5m":           normalize_momentum_5m,
    "momentum_1m":           normalize_momentum_1m,    # FIX: own normalizer
    "momentum_15m":          normalize_momentum_15m,   # FIX: own normalizer
    "vol_adjusted_momentum": normalize_vol_adjusted_momentum,
    "cex_poly_lag":          normalize_cex_poly_lag,
    "price_acceleration":    normalize_price_acceleration,
    "momentum_consistency":  normalize_consistency,
    "order_imbalance":       normalize_order_imbalance,
    "trade_flow_ratio":      normalize_trade_flow,
}


# =============================================================================
# Pre-trade filters (hard gates before scoring)
# =============================================================================

def apply_filters(cex, poly, config=None):
    """
    Returns (passed: bool, reason: str).

    FIX: all momentum direction checks now use `m5 is not None and m5 > 0`
    instead of `if m5 and m5 > 0`. The old form treats m5=0.001 as truthy
    (correct) but also silently skips the check when m5=0 exactly (fine) —
    the real risk was that `if m5` is falsy for 0.0, which can cause the
    RSI/divergence gates to be skipped when momentum is exactly flat.
    Using `is not None` is unambiguous.
    """
    m5         = cex.get("momentum_5m")
    vol5       = cex.get("volatility_5m")
    rsi        = cex.get("rsi_14")
    vol_ratio  = cex.get("volume_ratio", 1.0)
    poly_spread = poly.get("poly_spread")

    # Minimum momentum filter -- read from config, fall back to module default
    min_mom = (config or {}).get("min_momentum_pct", MIN_MOMENTUM_ABS)
    if m5 is not None and abs(m5) < min_mom:
        return False, f"momentum too weak ({m5:+.3f}% < ±{min_mom}%)"

    # Volatility filter — threshold is config-driven so the optimizer can tune it
    max_vol_5m = (config or {}).get("max_volatility_5m", MAX_VOLATILITY_5M)
    if vol5 is not None and vol5 > max_vol_5m:
        return False, f"volatility too high ({vol5:.3f} > {max_vol_5m})"

    # RSI filters -- thresholds read from config if available, else module defaults
    rsi_ob = (config or {}).get("rsi_overbought", RSI_OVERBOUGHT)
    rsi_os = (config or {}).get("rsi_oversold",   RSI_OVERSOLD)
    if rsi is not None:
        if m5 is not None and m5 > 0 and rsi > rsi_ob:
            return False, f"RSI overbought ({rsi:.0f} > {rsi_ob}), skip long"
        if m5 is not None and m5 < 0 and rsi < rsi_os:
            return False, f"RSI oversold ({rsi:.0f} < {rsi_os}), skip short"

    # Polymarket spread filter — catches illiquid windows
    # FIX: threshold was 0.10 (never fired). Fast markets trade at 0.01-0.03 spread.
    if poly_spread is not None and poly_spread > MIN_POLY_SPREAD:
        return False, f"Poly spread too wide ({poly_spread:.3f} > {MIN_POLY_SPREAD})"

    # Polymarket directional gate -- block when market has already priced in the move.
    # Use poly_yes_price absolute value (not divergence) for reliable filtering.
    # Edge zone: poly near 0.45-0.55 while CEX has moved = latency arb opportunity.
    poly_price = poly.get("poly_yes_price", 0.5) or 0.5
    cex_lag    = _calc_cex_poly_lag(cex, poly) or 0.0  # compute live; not in cex dict yet

    MAX_ENTRY_YES    = (config or {}).get("max_entry_yes",    0.476)
    MIN_ENTRY_YES    = (config or {}).get("min_entry_yes",    0.35)
    MIN_ENTRY_NO     = (config or {}).get("min_entry_no",     0.28)
    MAX_ENTRY_NO     = (config or {}).get("max_entry_no",     0.65)
    MIN_LAG_OVERRIDE = (config or {}).get("min_lag_override", 0.15)

    if m5 is not None and m5 > 0:  # signal wants YES
        if poly_price > MAX_ENTRY_YES:
            return False, f"Market priced in YES ({poly_price:.3f} > {MAX_ENTRY_YES}) -- no edge"
        if poly_price < MIN_ENTRY_YES and abs(cex_lag) < MIN_LAG_OVERRIDE:
            return False, f"Poly disagrees ({poly_price:.3f}) and CEX lag too small ({cex_lag:.4f})"

    if m5 is not None and m5 < 0:  # signal wants NO
        if poly_price < MIN_ENTRY_NO:
            return False, f"Market priced in NO ({poly_price:.3f} < {MIN_ENTRY_NO}) -- no edge"
        if poly_price > MAX_ENTRY_NO and cex_lag > -MIN_LAG_OVERRIDE:
            return False, f"Poly says UP ({poly_price:.3f}), CEX lag insufficient ({cex_lag:.4f})"

    # Volume gate -- only when volume_confidence is enabled in config.
    # Skip when vol_ratio == 1.0 (fallback sentinel = no avg data yet).
    volume_confidence = config.get("volume_confidence", False) if config else False
    if volume_confidence and vol_ratio is not None and vol_ratio != 1.0 and vol_ratio < 0.3:
        return False, f"Volume too low ({vol_ratio:.2f}x avg)"

    # Cross-timeframe momentum agreement filter (1m vs 5m)
    # When 1m momentum opposes 5m momentum and both are meaningful, the market
    # is at a likely reversal point. Trading into disagreement is noise, not edge.
    if (config or {}).get("momentum_agreement_filter", True):
        m1 = cex.get("momentum_1m")
        min_m1 = (config or {}).get("momentum_agreement_min_1m", 0.05)
        if (m1 is not None and m5 is not None
                and abs(m1) >= min_m1 and abs(m5) >= min_mom
                and (m1 > 0) != (m5 > 0)):
            return False, (
                f"timeframe disagreement: 1m={m1:+.3f}% opposes 5m={m5:+.3f}% "
                f"— reversal risk, skipping"
            )

    # Multi-timeframe alignment: 15m vs 5m direction agreement
    # A 5m move that runs against the 15m trend is a counter-trend spike —
    # the kind that causes the composite score to flip direction mid-window.
    # Only fires when both timeframes are above their minimum thresholds so
    # near-flat readings don't generate false blocks.
    if (config or {}).get("momentum_15m_agreement_filter", True):
        m15 = cex.get("momentum_15m")
        min_m15 = (config or {}).get("momentum_agreement_min_15m", 0.05)
        if (m15 is not None and m5 is not None
                and abs(m15) >= min_m15 and abs(m5) >= min_mom
                and (m15 > 0) != (m5 > 0)):
            return False, (
                f"timeframe disagreement: 15m={m15:+.3f}% opposes 5m={m5:+.3f}% "
                f"— counter-trend spike, skipping"
            )

    return True, "ok"


# =============================================================================
# Composite Scorer
# =============================================================================

def compute_composite_score(cex, poly, weights=None):
    """
    Compute a composite bullishness score in (0, 1).
    >0.5 = bullish signal, <0.5 = bearish signal.
    Returns (score, breakdown dict).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    signals = {
        "btc_vs_reference":      cex.get("btc_vs_reference"),
        "volume_ratio":          cex.get("volume_ratio"),
        "momentum_5m":           cex.get("momentum_5m"),
        "momentum_1m":           cex.get("momentum_1m"),
        "momentum_15m":          cex.get("momentum_15m"),
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

    total_weight = sum(weights.get(k, 0) for k in normalized)
    if total_weight == 0:
        return 0.5, {}

    score = sum(normalized[k] * weights.get(k, 0) for k in normalized) / total_weight

    breakdown = {
        k: {
            "raw":        signals.get(k),
            "normalized": normalized.get(k),
            "weight":     weights.get(k, 0),
        }
        for k in set(list(signals.keys()) + list(normalized.keys()))
    }

    return score, breakdown


def _calc_cex_poly_lag(cex, poly):
    m5       = cex.get("momentum_5m")
    poly_div = poly.get("poly_divergence", 0) or 0
    if m5 is None:
        return None
    return m5 * (0.15 - max(-0.15, min(0.15, poly_div))) / 0.15


# NOTE: fee-adjusted threshold logic is handled live in fast_trader.py via the
# fee_rate_bps field returned by the Gamma API, not estimated here. The fee EV
# check (fast_trader.py lines ~1066-1078) computes the actual breakeven win-rate
# from the market's real fee_rate and blocks trades where edge < min_edge.

# =============================================================================
# Main Interface
# =============================================================================

def get_composite_signal(cex_signals, poly_signals, config=None):
    """
    Main entry point. Takes pre-fetched CEX and Poly signal dicts.

    Returns dict:
        should_trade:  bool
        side:          'yes' | 'no' | None
        score:         float (0-1, >0.5 = bullish)
        confidence:    float (0-1, how far score is from 0.5)
        position_pct:  float (0-1, suggested fraction of max position)
        filter_reason: str (why trade was blocked, if applicable)
        breakdown:     dict (per-signal contributions)
        regime:        'slow' | 'normal' | 'active'
    """
    cfg = config or {}

    # ── Market regime detection ───────────────────────────────────────────────
    regime = detect_market_regime(cex_signals, cfg)

    # ── Weight selection: config override > regime-specific > default ─────────
    if "signal_weights" in cfg:
        # Explicit config override always wins
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

    # ── Time-of-day gate ─────────────────────────────────────────────────────
    hour_utc = datetime.now(timezone.utc).hour
    hour_ok, threshold_delta, hour_reason = check_hour_gate(hour_utc, cfg)
    session = get_market_session(hour_utc, cfg)

    result = {
        "should_trade":    False,
        "side":            None,
        "score":           0.5,
        "confidence":      0.0,
        "position_pct":    0.0,
        "filter_reason":   None,
        "breakdown":       {},
        "regime":          regime,
        "hour_utc":        hour_utc,
        "hour_accuracy":   get_hour_accuracy(hour_utc, cfg),
        "threshold_delta": threshold_delta,
        "session":         session,
    }

    if not hour_ok:
        result["filter_reason"] = hour_reason
        return result

    # Hard filters first
    passed, reason = apply_filters(cex_signals, poly_signals, config=cfg)
    if not passed:
        result["filter_reason"] = reason
        return result

    # Compute composite score
    score, breakdown = compute_composite_score(cex_signals, poly_signals, weights)
    result["score"]     = score
    result["breakdown"] = breakdown

    # Confidence = how far from neutral (0.5)
    confidence = abs(score - 0.5) * 2   # maps [0.5, 1.0] → [0, 1]
    result["confidence"] = confidence

    # ── Regime-aware threshold ────────────────────────────────────────────────
    base_threshold = COMPOSITE_ENTRY_THRESHOLD
    if "composite_threshold" in cfg:
        base_threshold = cfg["composite_threshold"]

    if regime == "slow":
        # Require a higher conviction bar in slow markets — weak signals are
        # unreliable noise when momentum is near-zero.
        threshold = cfg.get("slow_composite_threshold", max(base_threshold, 0.73))
    elif regime == "active":
        # Slightly lower bar in active markets — signals align more clearly.
        threshold = cfg.get("active_composite_threshold", max(base_threshold - 0.02, 0.60))
    else:
        threshold = base_threshold

    # Apply hour-of-day boost + market session adjustment
    # Hour delta: per-hour accuracy (e.g. 2h=+85.7% → -0.03 boost)
    # Session delta: London/NY/overlap liquidity windows lower the bar;
    #                off-hours raise it.  Both compound intentionally.
    session_delta = get_session_threshold_delta(session, cfg)
    threshold = max(0.55, threshold + threshold_delta + session_delta)

    if score > threshold:
        result["should_trade"] = True
        result["side"]         = "yes"
    elif score < (1 - threshold):
        result["should_trade"] = True
        result["side"]         = "no"
    else:
        result["filter_reason"] = (
            f"[{regime}] score {score:.3f} within neutral band "
            f"(threshold ±{threshold - 0.5:.3f})"
        )
        return result

    # ── Regime-aware position sizing ──────────────────────────────────────────
    # Slow:   reduce exposure — signals less reliable, protect capital.
    # Normal: allow up to a configurable cap (default 1.0 = uncapped).
    # Active: full size allowed — signals most trustworthy in trending markets.
    # Floor at 0.55 to clear the 5-share minimum at $3.50 max_position.
    MIN_POSITION_PCT = 0.55
    if regime == "slow":
        slow_size_cap = cfg.get("slow_position_pct_cap", 0.70)
        result["position_pct"] = min(
            max(confidence, MIN_POSITION_PCT),
            slow_size_cap,
        )
    elif regime == "normal":
        normal_size_cap = cfg.get("normal_position_pct_cap", 1.0)
        result["position_pct"] = min(
            max(confidence, MIN_POSITION_PCT),
            normal_size_cap,
        )
    else:  # active
        active_size_cap = cfg.get("active_position_pct_cap", 1.0)
        result["position_pct"] = min(
            max(confidence, MIN_POSITION_PCT),
            active_size_cap,
        )

    # Volatility position penalty — scale down size when price volatility is
    # elevated. High volatility degrades signal reliability even in active markets.
    # Penalty starts above vol_penalty_threshold and ramps linearly to -50% at
    # max_volatility_5m. Uses volatility_5m (price swing), not volume_ratio.
    vol5 = cex_signals.get("volatility_5m")
    _vpthresh = cfg.get("vol_penalty_threshold", 1.2)
    if vol5 is not None and vol5 > _vpthresh:
        _vmax = cfg.get("max_volatility_5m", MAX_VOLATILITY_5M)
        _range = max(_vmax - _vpthresh, 0.1)
        _penalty = min(0.50, (vol5 - _vpthresh) / _range)
        result["position_pct"] = max(MIN_POSITION_PCT, result["position_pct"] * (1.0 - _penalty))

    # Session-aware position cap — applied last so it overrides regime sizing
    # when market liquidity is low (off-hours, Asia) and allows full size
    # during peak liquidity (London-NY overlap).
    if cfg.get("session_gating_enabled", True):
        sess_cap = get_session_position_cap(session, cfg)
        result["position_pct"] = min(result["position_pct"], sess_cap)

    return result


# =============================================================================
# Standalone CLI for testing
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test composite signal")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--condition-id", default=None)
    args = parser.parse_args()

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
        raw  = info.get("raw")
        norm = info.get("normalized")
        w    = info.get("weight", 0)
        raw_str  = f"{raw:+.4f}" if raw is not None else "   n/a"
        norm_str = f"{norm:.4f}" if norm is not None else "  n/a"
        print(f"    {name:<28} raw={raw_str}  norm={norm_str}  w={w:.2f}")

    print(f"\n  CEX signals:")
    for k in ["momentum_1m", "momentum_5m", "momentum_15m", "rsi_14",
              "volume_ratio", "order_imbalance", "trade_flow_ratio",
              "volatility_5m", "price_acceleration", "btc_vs_reference"]:
        v = cex.get(k)
        if v is not None:
            print(f"    {k:<28} {v:+.4f}")

    if poly:
        print(f"\n  Poly signals:")
        for k, v in poly.items():
            if v is not None:
                print(f"    {k:<28} {v:+.4f}")