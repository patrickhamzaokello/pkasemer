#!/usr/bin/env python3
"""
Pknwitq Composite Signal Engine  —  v2.1

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
        position_pct = signal["position_pct"]   # 0.0–1.0, scale your max size by this

─────────────────────────────────────────────────────────────────────────────
  CHANGELOG v2.1
─────────────────────────────────────────────────────────────────────────────

  10. Active-regime RSI override  (config: "active_rsi_override": true)
        In v2.0 the RSI hard block fired unconditionally at rsi_overbought /
        rsi_oversold regardless of market context. RSI correlates +0.103 with
        outcomes — high RSI in a genuine trend is bullish confirmation, not
        an exhaustion signal. During a validated active regime, a hard RSI
        block was inverting the signal's own meaning.

        New: when active_rsi_override is enabled and ALL of the following are
        true simultaneously, the RSI hard block is bypassed:
          • regime == "active"          (strong momentum + above-avg volume)
          • |cex_poly_lag| >= min_lag_override  (genuine arb gap confirmed)
          • session in ("overlap", "london")    (peak institutional liquidity)

        Only these three conditions together justify override. The RSI block
        remains fully active in slow and normal regimes, in off/asia sessions,
        and whenever lag is below the override threshold. Disabled by default
        (active_rsi_override: false) — must be explicitly enabled in config.

        Observed impact: 13:05–13:15 UTC session, active/overlap, RSI 76–92,
        lag +0.100 to +0.331. All 20 cycles blocked in v2.0. With override
        enabled, the early cycles (RSI 76–84, lag confirmed) would proceed
        to scoring and the composite score would determine entry. The extreme
        exhaustion cycles (RSI 89–92 at candle-end with momentum decelerating)
        would self-filter via the scoring step.

  11. Structural lag bypass for minimum momentum gate
        (config: "lag_momentum_override": true, "min_vs_ref_override": 0.15)

        Structural divergence pattern: after a large validated spike, the
        5m candle consolidates. m5 drops to near zero while btc_vs_reference
        remains strongly positive (BTC is still well above priceToBeat) and
        cex_poly_lag remains large (poly still hasn't fully repriced). The
        momentum gate (min_momentum_pct) blocks the trade because m5 is flat,
        but both the structural position and arb signals remain strongly active.

        Implementation note: checking regime=="active" here would be circular.
        The regime detector uses |momentum_5m| as its primary input — when m5
        is flat the regime returns "slow" regardless of what caused the flat.
        The correct discriminator is btc_vs_reference magnitude, which stays
        elevated for the entire candle after a spike, independently of m5.

        Override fires only when ALL four conditions are met:
          • |btc_vs_reference| >= min_vs_ref_override (default 0.20)
            — spike was real and substantial; excludes slow-market noise
          • |cex_poly_lag|     >= min_lag_override    (default 0.12)
            — arb gap still open; poly hasn't repriced
          • cex_lag and btc_vs_reference agree in direction (no conflict)
          • vol_ratio >= 0.5 and not the sentinel value (1.0)
            — some volume present; pure zero-vol consolidation is unreliable

        The 0.20 default for min_vs_ref_override is the key discriminator:
          Slow-market small spike:   vs_ref ≈ 0.10–0.14%  → BLOCKED (< 0.20)
          Post-spike consolidation:  vs_ref ≈ 0.20–0.35%  → ALLOWED (≥ 0.20)

        Disabled by default (lag_momentum_override: false) — must be
        explicitly enabled in config.

─────────────────────────────────────────────────────────────────────────────
  CHANGELOG v2.0  (calibrated from 10,090 resolved observations)
─────────────────────────────────────────────────────────────────────────────

  1. DEFAULT_WEIGHTS rebuilt from scratch against 10k-obs correlation table.
       • btc_vs_reference demoted 0.315 → 0.150  (was calibrated on 362 obs,
         correlation dropped from +0.39 → +0.191 at scale)
       • vol_adjusted_momentum promoted to #1 (corr +0.234)
       • momentum_15m raised 0.034 → 0.085  (was severely underweighted)
       • rsi_14 added at 0.040  (#8 by correlation +0.103, was zero-weight)
       • poly_spread added at 0.005  (61.4% WR(hi), highest in table)

  2. _calc_cex_poly_lag formula corrected.
       Old: m5 * modifier — mechanically derived from momentum_5m.
            When poly_div ≈ 0 this collapses to m5 * 1.0 — effectively
            double-counting momentum_5m. Both signals showed +0.194/+0.197
            correlation confirming the overlap.
       New: btc_vs_reference - poly_divergence — true residual: what CEX
            knows that poly has not yet priced in. Genuinely independent.

  3. normalize_rsi added.
       RSI now contributes graded score (scale=0.08, conservative) rather
       than binary block only. Hard-block thresholds loosened (oversold
       35→22, overbought 75→80) since the graded weight self-penalises.
       Hard blocks still fire at genuine capitulation/exhaustion levels.

  4. normalize_poly_spread added.
       poly_spread had the highest WR(hi) in the correlation table (61.4%)
       but was only used as a hard-block gate (too wide = illiquid).
       Now also contributes a small positive score when elevated, capturing
       the arb-opportunity signal it carries.

  5. SLOW_MARKET_WEIGHTS corrected.
       Old version set cex_poly_lag=0.340 with the broken lag formula,
       which was doubling momentum_5m's effective weight in slow markets —
       the worst possible regime for an already-weak signal. Now uses the
       corrected independent lag at 0.250.

  6. ACTIVE_MARKET_WEIGHTS: momentum_15m raised 0.000→0.110.
       15m trend confirmation is most valuable precisely when price is
       trending. Setting it to zero in active markets was the opposite of
       what the correlation data supports.

  7. apply_filters: MAX_ENTRY_YES gate now checks MIN_LAG_OVERRIDE first,
       allowing a strong cex_poly_lag to override the poly price ceiling.
       This was intended in v1 but the condition was never reached because
       the lag formula was wrong. Now actually fires when appropriate.

  8. breakdown dict extended with per-signal 'contribution' field
       (weight * normalised / total_weight) for easier diagnostics.

  9. CLI extended: --config flag to load config.json directly, breakdown
       table sorted by weight, lag_value and threshold_used exposed.

─────────────────────────────────────────────────────────────────────────────
  Signal correlation reference  (10,090 resolved observations)
─────────────────────────────────────────────────────────────────────────────
  Signal                   Corr    WR(hi)   WR(lo)    Edge
  vol_adjusted_momentum  +0.234    60.3%    39.6%    +0.207   ← #1
  momentum_5m            +0.197    60.3%    39.6%    +0.206   ← #2
  cex_poly_lag           +0.194    60.1%    39.8%    +0.203   ← #3
  btc_vs_reference       +0.191    58.4%    42.4%    +0.160   ← #4
  momentum_1m            +0.137    54.9%    44.6%    +0.103   ← #5
  momentum_15m           +0.115    57.4%    42.1%    +0.152   ← #6
  price_acceleration     +0.114    54.8%    44.4%    +0.104   ← #7
  rsi_14                 +0.103    55.4%    43.9%    +0.115   ← #8  (new)
  volatility_1m          +0.074    52.7%    46.5%    +0.062
  volatility_5m          +0.055    50.6%    48.6%    +0.019
  trade_flow_ratio       +0.039    51.0%    48.3%    +0.027
  poly_spread            +0.038    61.4%    49.2%    +0.122   ← WR(hi) champion
  order_imbalance        +0.031    51.2%    48.0%    +0.032
  volume_ratio           +0.012    49.9%    49.3%    +0.006   (not significant)
  momentum_consistency   -0.003    49.5%    49.6%    -0.001   (negative — kept 0)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime, timezone


# =============================================================================
# Time-of-Day Signal Accuracy  (calibrated from 4,647 resolved observations)
#
# Signal accuracy = fraction of times the momentum direction correctly
# predicted the market outcome, filtered to |momentum_5m| >= 0.08%.
# =============================================================================

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

_DEFAULT_BLOCKED_HOURS = [6, 8, 16]

_DEFAULT_BOOSTED_HOURS = {
    2:  -0.03,   # 85.7% — lower threshold by 0.03
    4:  -0.02,   # 77.3%
    10: -0.02,   # 73.8%
    17: -0.02,   # 73.3%
    0:  -0.01,   # 71.9%
}


def get_hour_accuracy(hour_utc, config=None):
    """Return the observed signal accuracy for the given UTC hour (0-23)."""
    cfg   = config or {}
    table = {**_DEFAULT_HOUR_ACCURACY, **cfg.get("hour_accuracy", {})}
    return table.get(hour_utc, 0.60)   # 0.60 = conservative default for unseen hours


def check_hour_gate(hour_utc, config=None):
    """
    Returns (allowed: bool, threshold_delta: float, reason: str).
    allowed=False  → skip this cycle (bad hour)
    threshold_delta → added to composite_threshold (negative = easier to trade)
    """
    cfg         = config or {}
    blocked     = cfg.get("blocked_hours", _DEFAULT_BLOCKED_HOURS)
    raw_boosted = cfg.get("boosted_hours", _DEFAULT_BOOSTED_HOURS)
    boosted     = {int(k): v for k, v in raw_boosted.items()}

    if hour_utc in blocked:
        acc = get_hour_accuracy(hour_utc, cfg)
        return False, 0.0, f"hour {hour_utc}h blocked (signal accuracy={acc:.1%})"

    delta = boosted.get(hour_utc, 0.0)
    return True, delta, "ok"


# =============================================================================
# Market Session Detection (UTC)
#
# Polymarket BTC fast markets are directly influenced by traditional finance
# liquidity windows.  Three major sessions:
#
#   Asia      00:00-07:00 UTC — Tokyo / Singapore
#   London    07:00-13:00 UTC — European open (pre-overlap)
#   Overlap   13:00-16:00 UTC — London + NY simultaneously (highest liquidity)
#   New York  16:00-21:00 UTC — US only, post London close
#   Off       21:00-00:00 UTC — Post-NY quiet period
# =============================================================================

_SESSION_HOURS = {
    "overlap":   [13, 14, 15],
    "london":    [7, 8, 9, 10, 11, 12],
    "new_york":  [16, 17, 18, 19, 20],
    "asia":      [0, 1, 2, 3, 4, 5, 6],
    "off":       [21, 22, 23],
}

_DEFAULT_SESSION_DELTAS = {
    "overlap":  -0.03,   # peak liquidity — most reliable signals
    "london":   -0.02,   # strong European institutional flow
    "new_york": -0.01,   # solid US session
    "asia":      0.00,   # per-hour accuracy table handles fine-tuning
    "off":      +0.02,   # low liquidity — be more selective
}

_DEFAULT_SESSION_CAPS = {
    "overlap":  1.00,
    "london":   0.95,
    "new_york": 0.90,
    "asia":     0.85,
    "off":      0.70,
}


def get_market_session(hour_utc, config=None):
    """Return the primary market session for a given UTC hour (0-23)."""
    cfg        = config or {}
    sess_hours = {**_SESSION_HOURS, **cfg.get("session_hours", {})}
    for session in ("overlap", "london", "new_york", "asia", "off"):
        if hour_utc in sess_hours.get(session, []):
            return session
    return "off"


def get_session_threshold_delta(session, config=None):
    cfg = config or {}
    return cfg.get(
        f"{session}_threshold_delta",
        _DEFAULT_SESSION_DELTAS.get(session, 0.0),
    )


def get_session_position_cap(session, config=None):
    cfg = config or {}
    return cfg.get(
        f"{session}_position_cap",
        _DEFAULT_SESSION_CAPS.get(session, 0.85),
    )


# =============================================================================
# Default Signal Weights  —  rebuilt from 10,090-observation correlation table
#
# IMPORTANT: if config.json contains "signal_weights", these tables are
# bypassed entirely and the config values are used instead.  Remove
# "signal_weights" from config.json to let regime-adaptive logic run.
#
# All tables sum to 1.000.
# =============================================================================

DEFAULT_WEIGHTS = {
    "vol_adjusted_momentum": 0.205,   # #1 corr +0.234
    "momentum_5m":           0.170,   # #2 corr +0.197
    "cex_poly_lag":          0.160,   # #3 corr +0.194  (formula fixed, now independent)
    "btc_vs_reference":      0.150,   # #4 corr +0.191  (demoted from legacy 0.315)
    "momentum_1m":           0.105,   # #5 corr +0.137
    "momentum_15m":          0.085,   # #6 corr +0.115  (raised from under-calibrated 0.034)
    "price_acceleration":    0.080,   # #7 corr +0.114
    "rsi_14":                0.040,   # #8 corr +0.103  (added — was zero-weight in v1)
    "poly_spread":           0.005,   # WR(hi) 61.4%    (added — was gate-only in v1)
    "volume_ratio":          0.000,   # corr +0.012 — not statistically significant
    "momentum_consistency":  0.000,   # corr −0.003 — negative edge, correctly excluded
    "order_imbalance":       0.000,   # corr +0.031 — too weak
    "trade_flow_ratio":      0.000,   # corr +0.039 — too weak
}
# Sum = 1.000

# -----------------------------------------------------------------------------
# Slow-market weights (consolidating / low-momentum)
#
# Best edge in quiet markets: structural reference-price arb + true lag arb.
# vol_adjusted_momentum de-weighted (noisy when volatility is near zero).
# rsi_14 removed (RSI unreliable during flat consolidation).
#
# Requires the corrected _calc_cex_poly_lag to be genuinely independent
# from momentum_5m — do NOT use this table with the v1 lag formula.
# -----------------------------------------------------------------------------
SLOW_MARKET_WEIGHTS = {
    "btc_vs_reference":      0.290,   # structural anchor — most stable in flat markets
    "cex_poly_lag":          0.250,   # true residual arb (corrected formula)
    "momentum_5m":           0.140,
    "momentum_1m":           0.100,
    "momentum_15m":          0.080,   # kept — +0.115 corr holds across regimes
    "vol_adjusted_momentum": 0.080,   # de-weighted: noisy when vol is low
    "price_acceleration":    0.060,
    "rsi_14":                0.000,
    "poly_spread":           0.000,
    "volume_ratio":          0.000,
    "momentum_consistency":  0.000,
    "order_imbalance":       0.000,
    "trade_flow_ratio":      0.000,
}
# Sum = 1.000

# -----------------------------------------------------------------------------
# Active-market weights (strong trending environment)
#
# Multi-timeframe alignment is the most reliable signal when price is trending.
# momentum_15m raised to 0.110 — was 0.000 in v1, directly contradicting the
# correlation data which shows this signal is most valuable in trends.
# cex_poly_lag reduced — poly reprices quickly in active markets, arb thin.
# -----------------------------------------------------------------------------
ACTIVE_MARKET_WEIGHTS = {
    "vol_adjusted_momentum": 0.240,   # momentum/vol: most reliable in trends
    "btc_vs_reference":      0.210,
    "momentum_5m":           0.180,
    "momentum_15m":          0.110,   # raised from 0.000 — trend confirmation is key
    "cex_poly_lag":          0.080,   # reduced — arb window thins when poly is fast
    "momentum_1m":           0.070,
    "price_acceleration":    0.060,
    "rsi_14":                0.030,   # light — overbought/sold still informative
    "poly_spread":           0.020,   # wider spread in active = more arb opportunity
    "volume_ratio":          0.000,
    "momentum_consistency":  0.000,
    "order_imbalance":       0.000,
    "trade_flow_ratio":      0.000,
}
# Sum = 1.000


# =============================================================================
# Hard-filter thresholds  (all overridable via config.json)
# =============================================================================

COMPOSITE_ENTRY_THRESHOLD = 0.55   # score must exceed this for a trade signal
MIN_MOMENTUM_ABS           = 0.05   # minimum |momentum_5m| to suppress flat noise
MAX_VOLATILITY_5M          = 2.0    # skip if 5m price volatility is too chaotic
MIN_POLY_SPREAD            = 0.04   # block illiquid windows (normal spreads 0.01-0.03)

# Loosened from v1 (was hardcoded 85/15 in script, configured at 75/35).
# RSI now provides graded scoring — weight self-penalises before the hard
# block fires.  These thresholds now target genuine capitulation/exhaustion
# rather than ordinary oversold/overbought conditions.
# The RSI=17–23 dip seen in the logs would still be hard-blocked here.
RSI_OVERBOUGHT = 80
RSI_OVERSOLD   = 22


# =============================================================================
# Market Regime Detection
# =============================================================================

def detect_market_regime(cex_signals, config=None):
    """
    Classify the current market as 'slow', 'normal', or 'active'.

    Slow   — consolidating / low-momentum. Best edge: latency arb + reference.
    Normal — typical conditions. Default strategy.
    Active — strong trending. Momentum signals most reliable.

    Config overrides:
        slow_mom_threshold    default 0.12   (|momentum_5m| below = slow)
        slow_vol_threshold    default 0.80   (volume_ratio below = slow)
        active_mom_threshold  default 0.25   (|momentum_5m| above = active)
        active_vol_threshold  default 1.30   (volume_ratio above = active)
    """
    cfg      = config or {}
    m5       = abs(cex_signals.get("momentum_5m") or 0)
    vr       = cex_signals.get("volume_ratio") or 1.0
    vr_valid = vr != 1.0   # 1.0 is the "no data yet" sentinel; treat as neutral

    slow_mom   = cfg.get("slow_mom_threshold",   0.12)
    slow_vol   = cfg.get("slow_vol_threshold",   0.80)
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
    """Squash any value to (0, 1). scale controls steepness."""
    return 1.0 / (1.0 + math.exp(-x * scale))


def normalize_momentum_5m(m, clip_pct=1.0):
    """
    5m momentum in percent. Clip to ±1.0%, sigmoid to (0,1).
    ±1% over 5m is large; clipping prevents outlier domination.
    """
    if m is None:
        return None
    return _sigmoid(max(-clip_pct, min(clip_pct, m)), scale=3.0)


def normalize_momentum_1m(m, clip_pct=0.3):
    """
    1m momentum in percent. Clip to ±0.3% (moves are smaller on 1m).
    Steeper sigmoid (scale=8) amplifies signal from small micro-moves.
    """
    if m is None:
        return None
    return _sigmoid(max(-clip_pct, min(clip_pct, m)), scale=8.0)


def normalize_momentum_15m(m, clip_pct=2.0):
    """
    15m momentum in percent. Clip to ±2.0% (larger moves expected).
    Shallower sigmoid (scale=1.5): trend confirmation, not entry trigger.
    """
    if m is None:
        return None
    return _sigmoid(max(-clip_pct, min(clip_pct, m)), scale=1.5)


def normalize_cex_poly_lag(lag):
    """
    True latency-arb signal: residual CEX edge that poly hasn't priced in.
    Positive = CEX moved up, poly is still cheap (arb: buy YES).
    Negative = CEX moved down, poly is still expensive (arb: buy NO).
    See _calc_cex_poly_lag for the v2.0 corrected computation.
    """
    if lag is None:
        return None
    return _sigmoid(lag, scale=2.0)


def normalize_vol_adjusted_momentum(vam):
    """Momentum per unit vol. Sigmoid scale=1.0 (gentle, avoids extremes)."""
    if vam is None:
        return None
    return _sigmoid(vam, scale=1.0)


def normalize_btc_vs_reference(v):
    """
    (current_btc_price - priceToBeat) / priceToBeat * 100.
    Positive = Up is currently winning vs the reference price.
    Clip to ±0.30%.  Steep sigmoid: ±0.10% → 0.73/0.27; ±0.30% → 0.95/0.05.
    """
    if v is None:
        return None
    return _sigmoid(max(-0.30, min(0.30, v)), scale=10.0)


def normalize_volume_ratio(vr):
    """
    volume_ratio = latest_candle_vol / avg_vol. Centred at 1.0.
    1.5x → ~0.73.  0.5x → ~0.27.
    """
    if vr is None:
        return None
    return _sigmoid(vr - 1.0, scale=2.0)


def normalize_price_acceleration(pa):
    """
    price_acceleration = recent_5m_mom - prior_5m_mom (in %).
    Positive = momentum accelerating (bullish confirmation).
    """
    if pa is None:
        return None
    return _sigmoid(pa, scale=8.0)


def normalize_order_imbalance(oi):
    """OI in (−1, 1). Map to (0, 1): 1.0 = all bids (bullish)."""
    if oi is None:
        return None
    return (oi + 1) / 2


def normalize_trade_flow(tfr):
    """TFR = fraction of buy volume (0-1). 0.5 = neutral."""
    return tfr


def normalize_consistency(c):
    """Already in (0,1). >0.5 = recent candles agree with direction."""
    return c


def normalize_rsi(rsi):
    """
    RSI 0-100, centred at 50 (neutral). Conservative scale=0.08.
    RSI acts as graded confirmation weight, not entry trigger.

    Hard blocks in apply_filters still fire at RSI_OVERSOLD / RSI_OVERBOUGHT
    for genuine capitulation/exhaustion; this normaliser handles the space
    in between, letting a healthy RSI add a small boost and an unhealthy
    RSI apply a small drag before the hard block is reached.

    Examples:
        RSI = 70  → sigmoid((70-50)*0.08) = sigmoid(1.60) ≈ 0.832  (bullish)
        RSI = 55  → sigmoid(0.40)          ≈ 0.599  (mild bullish)
        RSI = 50  → sigmoid(0.00)          = 0.500  (neutral)
        RSI = 35  → sigmoid(-1.20)         ≈ 0.231  (bearish drag)
        RSI = 25  → sigmoid(-2.00)         ≈ 0.119  (strong bearish drag)
    """
    if rsi is None:
        return None
    return _sigmoid(rsi - 50, scale=0.08)


def normalize_poly_spread(spread):
    """
    Poly bid-ask spread. Correlation table: WR(hi) 61.4% — highest in table.
    Elevated spread = more price uncertainty = more arb opportunity.

    Typical fast-market spread: 0.01-0.02. Elevated: 0.02-0.04.
    Centred at 0.015 (typical spread). Clipped at hard-block threshold (0.04).
    Above 0.04 apply_filters blocks the trade entirely.

    At spread=0.025 → sigmoid ≈ 0.73  (moderate arb signal)
    At spread=0.010 → sigmoid ≈ 0.27  (tight book, little edge)
    """
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
    "rsi_14":                normalize_rsi,           # v2.0: new
    "poly_spread":           normalize_poly_spread,   # v2.0: new
    "volume_ratio":          normalize_volume_ratio,
    "momentum_consistency":  normalize_consistency,
    "order_imbalance":       normalize_order_imbalance,
    "trade_flow_ratio":      normalize_trade_flow,
}


# =============================================================================
# CEX/Poly Lag Computation  —  v2.0 CORRECTED
#
# v1 (broken):
#     m5 * (0.15 - clamp(poly_div, ±0.15)) / 0.15
#   When poly_div ≈ 0, reduces to m5 * 1.0 — lag == momentum_5m.
#   This caused double-counting: the two signals had near-identical
#   correlations (+0.194 vs +0.197) because they were measuring the same
#   thing. The SLOW_MARKET_WEIGHTS amplified this further by setting
#   cex_poly_lag=0.340 alongside momentum_5m=0.130, giving momentum_5m
#   an effective combined weight of ~0.47 in slow markets.
#
# v2 (corrected):
#     btc_vs_reference - poly_divergence
#   btc_vs_reference = (CEX_price - priceToBeat) / priceToBeat * 100
#                      How much the CEX reference has moved vs the poly target.
#   poly_divergence  = how much poly has already repriced from its open.
#   Residual         = the fraction of CEX movement poly has NOT yet captured.
#                      This is the true latency-arb window.
#
# The two signals are now independent by construction:
#   momentum_5m       measures the speed of CEX price change over 5m
#   cex_poly_lag      measures the structural gap between CEX position and
#                     poly's current pricing — regardless of how fast it got there
# =============================================================================

def _calc_cex_poly_lag(cex, poly):
    """
    Compute the true CEX-to-Poly latency-arb residual.

    Positive: CEX is bullish vs its reference, but poly hasn't priced it in → buy YES.
    Negative: CEX is bearish vs its reference, poly still too high → buy NO.

    Returns None if btc_vs_reference is unavailable.
    """
    btc_ref  = cex.get("btc_vs_reference")
    poly_div = poly.get("poly_divergence", 0) or 0

    if btc_ref is None:
        return None

    return btc_ref - poly_div


def compute_dynamic_bounds(poly_price, lag, volatility, config=None):
    """
    Compute dynamic entry bounds based on fair value.

    fair_price = poly + lag_adjustment
    edge       = volatility buffer
    """

    cfg = config or {}

    lag_weight = cfg.get("dynamic_lag_weight", DYNAMIC_LAG_WEIGHT)
    vol_k      = cfg.get("dynamic_vol_k", DYNAMIC_VOL_K)

    if lag is None:
        lag = 0

    if volatility is None:
        volatility = 0

    fair = poly_price + lag * lag_weight
    edge = abs(volatility) * vol_k

    max_yes = fair - edge
    min_no  = fair + edge

    # Safety caps
    max_yes = min(max_yes, cfg.get("global_max_yes", GLOBAL_MAX_YES))
    min_no  = max(min_no,  cfg.get("global_min_no",  GLOBAL_MIN_NO))

    return max_yes, min_no, fair, edge


# =============================================================================
# Pre-trade Filters  (hard gates applied before scoring)
# =============================================================================

def apply_filters(cex, poly, config=None):
    """
    Returns (passed: bool, reason: str).

    All directional checks use `x is not None` guards to avoid truthiness
    ambiguity — `if m5` is falsy when m5=0.0 exactly, which would silently
    skip RSI / poly-price gates when momentum is precisely flat.
    `is not None` is unambiguous regardless of value.
    """
    cfg         = config or {}
    m5          = cex.get("momentum_5m")
    vol5        = cex.get("volatility_5m")
    rsi         = cex.get("rsi_14")
    vol_ratio   = cex.get("volume_ratio", 1.0)
    poly_spread = poly.get("poly_spread")
    min_mom     = cfg.get("min_momentum_pct", MIN_MOMENTUM_ABS)

    # ── Minimum momentum gate ────────────────────────────────────────────────
    # Bypass condition: "lag_momentum_override" (default: false)
    #
    # Structural divergence pattern: after a large validated spike, the 5m
    # candle consolidates — m5 drops to near zero while btc_vs_reference
    # remains strongly elevated and cex_poly_lag remains large.  The regime
    # detector (which uses m5 as its primary input) will itself return "slow"
    # at this point, so checking regime=="active" here would be circular and
    # would never fire. The correct discriminator is vs_ref magnitude: a large
    # vs_ref means a significant spike DID happen regardless of current m5.
    #
    # Override fires only when ALL four conditions are met:
    #   1. |btc_vs_reference| >= min_vs_ref_override (0.20)
    #      — large structural gap: spike was real and substantial
    #   2. |cex_poly_lag| >= min_lag_override (0.12)
    #      — arb gap still open: poly hasn't repriced yet
    #   3. cex_lag and btc_vs_reference agree in direction (no conflict)
    #      — both signals point the same way, no mixed evidence
    #   4. vol_ratio != 1.0 (not the sentinel) AND vol_ratio >= 0.5
    #      — some volume confirmation: pure zero-volume consolidation
    #        with strong vs_ref is unreliable; require at least 50% avg vol
    #
    # Default min_vs_ref_override = 0.20 differentiates the two patterns:
    #   Slow-market small spike:  vs_ref ≈ 0.10-0.14%  → BLOCKED (< 0.20)
    #   Post-active consolidation: vs_ref ≈ 0.20-0.35% → ALLOWED (>= 0.20)
    #
    # Does NOT require regime=="active" for the reason above. The vs_ref +
    # lag magnitude guards are sufficient to exclude slow-market noise.
    if m5 is not None and abs(m5) < min_mom:
        lag_mom_override = cfg.get("lag_momentum_override", False)
        _allowed = False
        if lag_mom_override:
            _lag      = _calc_cex_poly_lag(cex, poly) or 0.0
            _ref      = cex.get("btc_vs_reference") or 0.0
            _vr       = cex.get("volume_ratio") or 1.0
            _min_lag  = cfg.get("min_lag_override",    0.12)
            _min_ref  = cfg.get("min_vs_ref_override", 0.20)
            _same_dir = (_lag > 0) == (_ref > 0)
            _vol_ok   = _vr != 1.0 and _vr >= 0.5
            _allowed  = (
                abs(_ref) >= _min_ref
                and abs(_lag) >= _min_lag
                and _same_dir
                and _vol_ok
            )
        if not _allowed:
            return False, f"momentum too weak ({m5:+.3f}% < ±{min_mom}%)"
        # else: structural arb confirmed — vs_ref large, lag open, vol present

    # ── Volatility gate ──────────────────────────────────────────────────────
    max_vol_5m = cfg.get("max_volatility_5m", MAX_VOLATILITY_5M)
    if vol5 is not None and vol5 > max_vol_5m:
        return False, f"volatility too high ({vol5:.3f} > {max_vol_5m})"

    # ── RSI extreme-block gates ──────────────────────────────────────────────
    # Thresholds loosened vs v1 (overbought 75→80, oversold 35→22).
    # The graded normalize_rsi weight self-penalises before this fires.
    # Hard blocks now target genuine exhaustion / capitulation only.
    #
    # Bypass condition: "active_rsi_override" (default: false)
    #
    # RSI correlates +0.103 with outcomes — high RSI in a genuine trend
    # is bullish confirmation, not an exhaustion signal. In an active regime
    # with confirmed arb lag during peak liquidity, the hard block inverts
    # the signal's own meaning. The three-condition guard ensures the bypass
    # only fires during validated trends; it never fires in slow/normal
    # markets, off/asia sessions, or when lag is below the override threshold.
    #
    # Override fires only when ALL three conditions are met:
    #   1. regime == "active"                    — confirmed strong trend
    #   2. |cex_poly_lag| >= min_lag_override    — genuine arb gap open
    #   3. session in ("overlap", "london")      — peak institutional liquidity
    #      (RSI exhaustion near open or in Asia is a real reversal warning
    #       even in active conditions — session restriction keeps the guard tight)
    rsi_ob = cfg.get("rsi_overbought", RSI_OVERBOUGHT)
    rsi_os = cfg.get("rsi_oversold",   RSI_OVERSOLD)
    if rsi is not None:
        rsi_override = cfg.get("active_rsi_override", False)

        if m5 is not None and m5 > 0 and rsi > rsi_ob:
            _bypassed = False
            if rsi_override:
                _regime  = detect_market_regime(cex, cfg)
                _lag     = _calc_cex_poly_lag(cex, poly) or 0.0
                _min_lag = cfg.get("min_lag_override", 0.12)
                _session = get_market_session(datetime.now(timezone.utc).hour, cfg)
                _bypassed = (
                    _regime == "active"
                    and abs(_lag) >= _min_lag
                    and _session in ("overlap", "london")
                )
            if not _bypassed:
                return False, f"RSI overbought ({rsi:.0f} > {rsi_ob}), skip long"
            # else: active trend + confirmed lag + peak session — bypass RSI block

        if m5 is not None and m5 < 0 and rsi < rsi_os:
            _bypassed = False
            if rsi_override:
                _regime  = detect_market_regime(cex, cfg)
                _lag     = _calc_cex_poly_lag(cex, poly) or 0.0
                _min_lag = cfg.get("min_lag_override", 0.12)
                _session = get_market_session(datetime.now(timezone.utc).hour, cfg)
                _bypassed = (
                    _regime == "active"
                    and abs(_lag) >= _min_lag
                    and _session in ("overlap", "london")
                )
            if not _bypassed:
                return False, f"RSI oversold ({rsi:.0f} < {rsi_os}), skip short"
            # else: active trend + confirmed lag + peak session — bypass RSI block

    # ── Poly spread gate ─────────────────────────────────────────────────────
    # Blocks illiquid windows where execution risk exceeds arb opportunity.
    # Fast markets trade at 0.01-0.03; above 0.04 = abnormally thin book.
    # Below this threshold, spread contributes positively via normalize_poly_spread.
    min_spread_block = cfg.get("min_poly_spread", MIN_POLY_SPREAD)
    if poly_spread is not None and poly_spread > min_spread_block:
        return False, f"Poly spread too wide ({poly_spread:.3f} > {min_spread_block})"

    # ── Poly price directional gate ──────────────────────────────────────────
    # Block when poly has already priced in the move (no arb edge remaining).
    # poly_yes_price used as risk control ceiling/floor, not as a score weight,
    # because the correlation data shows it is better as a hard gate.
    #

    poly_price = poly.get("poly_yes_price", 0.5) or 0.5
    cex_lag    = _calc_cex_poly_lag(cex, poly) or 0.0
    vol5       = cex.get("volatility_5m")
    
    use_dynamic = cfg.get("use_dynamic_price_bands", USE_DYNAMIC_PRICE_BANDS)
    
    if use_dynamic:
        max_yes, min_no, fair_price, edge = compute_dynamic_bounds(
            poly_price,
            cex_lag,
            vol5,
            cfg
        )
    else:
        max_yes = cfg.get("max_entry_yes", 0.476)
        min_no  = cfg.get("min_entry_no", 0.52)
   

    if m5 is not None and m5 > 0:   # signal wants YES
    
        if poly_price > max_yes:
            return False, (
                f"YES overpriced ({poly_price:.3f} > dynamic {max_yes:.3f})"
            )

    if m5 is not None and m5 < 0:   # signal wants NO
    
        if poly_price < min_no:
            return False, (
                f"NO overpriced ({poly_price:.3f} < dynamic {min_no:.3f})"
            )

    # ── Volume confidence gate (optional) ────────────────────────────────────
    volume_confidence = cfg.get("volume_confidence", False)
    if (volume_confidence
            and vol_ratio is not None
            and vol_ratio != 1.0
            and vol_ratio < 0.3):
        return False, f"Volume too low ({vol_ratio:.2f}x avg)"

    # ── Cross-timeframe agreement: 1m vs 5m ──────────────────────────────────
    # When 1m opposes 5m and both are meaningful, the market is likely at a
    # short-term reversal point. Don't trade into disagreement.
    if cfg.get("momentum_agreement_filter", True):
        m1     = cex.get("momentum_1m")
        min_m1 = cfg.get("momentum_agreement_min_1m", 0.05)
        if (m1 is not None and m5 is not None
                and abs(m1) >= min_m1 and abs(m5) >= min_mom
                and (m1 > 0) != (m5 > 0)):
            return False, (
                f"timeframe disagreement: 1m={m1:+.3f}% opposes 5m={m5:+.3f}%"
                f" — reversal risk, skipping"
            )

    # ── Cross-timeframe agreement: 15m vs 5m ─────────────────────────────────
    # A 5m counter-trend spike against the 15m trend is the most common source
    # of false positives. Only fires when both timeframes are meaningfully moved.
    if cfg.get("momentum_15m_agreement_filter", True):
        m15     = cex.get("momentum_15m")
        min_m15 = cfg.get("momentum_agreement_min_15m", 0.05)
        if (m15 is not None and m5 is not None
                and abs(m15) >= min_m15 and abs(m5) >= min_mom
                and (m15 > 0) != (m5 > 0)):
            return False, (
                f"timeframe disagreement: 15m={m15:+.3f}% opposes 5m={m5:+.3f}%"
                f" — counter-trend spike, skipping"
            )

    return True, "ok"


# =============================================================================
# Composite Scorer
# =============================================================================

def compute_composite_score(cex, poly, weights=None):
    """
    Compute a composite bullishness score in (0, 1).
    >0.5 = bullish signal. <0.5 = bearish. Returns (score, breakdown).

    breakdown dict per signal:
        raw          — original value from CEX/poly dict
        normalized   — after normalizer function
        weight       — weight assigned from the active weight table
        contribution — weight * normalized / total_weight (score attribution)
    """
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
            "raw":          signals.get(k),
            "normalized":   normalized.get(k),
            "weight":       weights.get(k, 0),
            "contribution": (
                normalized[k] * weights.get(k, 0) / total_weight
                if k in normalized else 0.0
            ),
        }
        for k in set(list(signals.keys()) + list(normalized.keys()))
    }

    return score, breakdown


# NOTE: fee-adjusted threshold logic is handled in fast_trader.py via the
# fee_rate_bps field from the Gamma API.  The fee EV check computes the
# actual breakeven win-rate from the market's real fee_rate and blocks trades
# where edge < min_edge.  Do not duplicate that logic here.


# =============================================================================
# Main Interface
# =============================================================================

def get_composite_signal(cex_signals, poly_signals, config=None):
    """
    Main entry point. Takes pre-fetched CEX and Poly signal dicts.

    Returns dict:
        should_trade    bool
        side            'yes' | 'no' | None
        score           float (0-1, >0.5 = bullish)
        confidence      float (0-1, distance from 0.5 normalised to [0,1])
        position_pct    float (0-1, suggested fraction of max position)
        filter_reason   str   (why trade was blocked, if applicable)
        breakdown       dict  (per-signal raw/normalised/weight/contribution)
        regime          'slow' | 'normal' | 'active'
        session         'overlap' | 'london' | 'new_york' | 'asia' | 'off'
        hour_accuracy   float (observed accuracy for this UTC hour)
        threshold_used  float (final threshold after all adjustments)
        lag_value       float (corrected cex_poly_lag for diagnostics)
    """
    cfg = config or {}

    # ── Market regime ──────────────────────────────────────────────────────────
    regime = detect_market_regime(cex_signals, cfg)

    # ── Weight selection: config override > regime-specific > default ──────────
    # If config.json contains "signal_weights" that key always wins and the
    # regime-adaptive tables below are bypassed entirely.
    # To use regime-aware weights, remove "signal_weights" from config.json.
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

    # ── Time-of-day gate ───────────────────────────────────────────────────────
    hour_utc                  = datetime.now(timezone.utc).hour
    hour_ok, hour_delta, hour_reason = check_hour_gate(hour_utc, cfg)
    session                   = get_market_session(hour_utc, cfg)
    lag_value                 = _calc_cex_poly_lag(cex_signals, poly_signals)

    result = {
        "should_trade":   False,
        "side":           None,
        "score":          0.5,
        "confidence":     0.0,
        "position_pct":   0.0,
        "filter_reason":  None,
        "breakdown":      {},
        "regime":         regime,
        "session":        session,
        "hour_utc":       hour_utc,
        "hour_accuracy":  get_hour_accuracy(hour_utc, cfg),
        "threshold_delta": hour_delta,
        "threshold_used": None,
        "lag_value":      lag_value,
    }

    if not hour_ok:
        result["filter_reason"] = hour_reason
        return result

    # ── Hard filters ───────────────────────────────────────────────────────────
    passed, reason = apply_filters(cex_signals, poly_signals, config=cfg)
    if not passed:
        result["filter_reason"] = reason
        return result

    # ── Composite score ────────────────────────────────────────────────────────
    score, breakdown     = compute_composite_score(cex_signals, poly_signals, weights)
    result["score"]      = score
    result["breakdown"]  = breakdown
    result["confidence"] = abs(score - 0.5) * 2   # [0.5, 1.0] → [0, 1]

    # ── Regime-aware threshold ─────────────────────────────────────────────────
    base_threshold = cfg.get("composite_threshold", COMPOSITE_ENTRY_THRESHOLD)

    if regime == "slow":
        # Higher conviction bar — weak signals in flat markets are unreliable.
        threshold = cfg.get("slow_composite_threshold", max(base_threshold, 0.73))
    elif regime == "active":
        # Slightly lower bar — multi-signal alignment more reliable in trends.
        threshold = cfg.get("active_composite_threshold", max(base_threshold - 0.02, 0.60))
    else:
        threshold = base_threshold

    # Hour and session deltas compound intentionally.
    # Hour: accuracy-based boost (e.g. 2h=85.7% → −0.03)
    # Session: liquidity-based adjustment (overlap=−0.03, off=+0.02)
    session_delta            = get_session_threshold_delta(session, cfg)
    threshold                = max(0.55, threshold + hour_delta + session_delta)
    result["threshold_used"] = threshold

    # ── Entry decision ─────────────────────────────────────────────────────────
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

    # ── Position sizing ────────────────────────────────────────────────────────
    # Floor at 0.55 to clear the 5-share minimum at $3.50 max_position.
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

    # Volatility position penalty.
    # Penalty ramps from 0% at vol_penalty_threshold to -50% at max_volatility_5m.
    # Uses volatility_5m (price swing), not volume_ratio.
    vol5      = cex_signals.get("volatility_5m")
    vp_thresh = cfg.get("vol_penalty_threshold", 1.2)
    if vol5 is not None and vol5 > vp_thresh:
        v_max   = cfg.get("max_volatility_5m", MAX_VOLATILITY_5M)
        v_range = max(v_max - vp_thresh, 0.1)
        penalty = min(0.50, (vol5 - vp_thresh) / v_range)
        result["position_pct"] = max(
            MIN_POSITION_PCT,
            result["position_pct"] * (1.0 - penalty),
        )

    # Session-aware cap: applied last so it can override regime sizing
    # during off-hours or Asia, and allows full size in the London-NY overlap.
    if cfg.get("session_gating_enabled", True):
        sess_cap             = get_session_position_cap(session, cfg)
        result["position_pct"] = min(result["position_pct"], sess_cap)

    return result


# =============================================================================
# Standalone CLI  —  quick smoke-test / live signal preview
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composite signal engine v2.1")
    parser.add_argument("--symbol",       default="BTCUSDT")
    parser.add_argument("--condition-id", default=None)
    parser.add_argument("--config",       default=None,
                        help="Path to config.json (optional, for live config test)")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as fh:
            cfg = json.load(fh)

    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from signal_research import (
            extract_cex_signals, extract_poly_signals, fetch_poly_market,
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
    print(f"\n{'─'*W}")
    print(f"  COMPOSITE SIGNAL  v2.1")
    print(f"{'─'*W}")
    print(f"  Score:          {signal['score']:.4f}   (0.5 = neutral)")
    if signal["threshold_used"] is not None:
        print(f"  Threshold used: {signal['threshold_used']:.4f}")
    print(f"  Confidence:     {signal['confidence']:.4f}   (0=neutral, 1=max)")
    print(f"  Should trade:   {signal['should_trade']}")
    print(f"  Side:           {signal['side'] or 'n/a'}")
    print(f"  Position pct:   {signal['position_pct']:.1%} of max size")
    print(f"  Regime:         {signal['regime']}")
    print(f"  Session:        {signal['session']}  "
          f"({signal['hour_utc']}h UTC, accuracy={signal['hour_accuracy']:.1%})")
    if signal["lag_value"] is not None:
        print(f"  CEX-Poly lag:   {signal['lag_value']:+.4f}%")
    rsi_ov  = cfg.get("active_rsi_override",   False)
    lag_ov  = cfg.get("lag_momentum_override",  False)
    print(f"  Overrides:      RSI={'ON' if rsi_ov else 'off'}  "
          f"LagMomentum={'ON' if lag_ov else 'off'}")
    if signal["filter_reason"]:
        print(f"  Blocked by:     {signal['filter_reason']}")

    print(f"\n  {'Signal':<26} {'Raw':>9}  {'Norm':>6}  {'Weight':>6}  {'Contrib':>8}")
    print(f"  {'─'*26}  {'─'*9}  {'─'*6}  {'─'*6}  {'─'*8}")
    for name, info in sorted(
        signal.get("breakdown", {}).items(),
        key=lambda x: -abs(x[1].get("weight", 0)),
    ):
        w   = info.get("weight", 0)
        raw = info.get("raw")
        if w == 0 and raw is None:
            continue
        norm = info.get("normalized")
        con  = info.get("contribution", 0)
        raw_s  = f"{raw:+.4f}" if raw is not None else "     n/a"
        norm_s = f"{norm:.4f}" if norm is not None else "   n/a"
        print(f"  {name:<26} {raw_s:>9}  {norm_s:>6}  {w:>6.3f}  {con:>+8.4f}")

    print(f"\n  CEX signals:")
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
        print(f"\n  Poly signals:")
        for k, v in poly.items():
            if v is not None:
                print(f"    {k:<28} {v:+.4f}")
