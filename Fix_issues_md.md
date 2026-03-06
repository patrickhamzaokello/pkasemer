# Pknwitq — Exact Code Changes (Mapped to Actual Files)

> Based on actual source: `fast_trader.py`, `composite_signal.py`, `signal_research.py`, `scheduler.py`

---

## CHANGE 1 — `composite_signal.py` — RSI Thresholds

**Problem:** RSI 72/28 blocks entire trending markets. At RSI=86 during the March 5 uptrend, every long signal was killed for 30+ minutes. RSI oversold at 23–27 blocked good shorts on March 6.

**Find** (near line 53):
```python
RSI_OVERBOUGHT     = 72            # avoid chasing already-overbought
RSI_OVERSOLD       = 28            # avoid chasing already-oversold
```

**Replace with:**
```python
RSI_OVERBOUGHT     = 85            # only block extreme exhaustion spikes
RSI_OVERSOLD       = 15            # only block extreme capitulation
```

---

## CHANGE 2 — `composite_signal.py` — DEFAULT_WEIGHTS

**Problem:** Current defaults have `cex_poly_lag=0.00`, `momentum_5m=0.00`, and no `momentum_1m`. These don't match the research. The config overrides them, but the defaults should also be correct as a fallback.

**Find** (line ~30):
```python
DEFAULT_WEIGHTS = {
    "btc_vs_reference":      0.45,
    "vol_adjusted_momentum": 0.22,
    "price_acceleration":    0.13,
    "momentum_15m":          0.12,
    "momentum_1m":           0.08,
    "momentum_5m":           0.00,
    "cex_poly_lag":          0.00,
    "volume_ratio":          0.00,
    "trade_flow_ratio":      0.00,
    "order_imbalance":       0.00,
    "momentum_consistency":  0.00,
}
```

**Replace with:**
```python
DEFAULT_WEIGHTS = {
    "btc_vs_reference":      0.315,   # dominant signal, corr=0.263
    "vol_adjusted_momentum": 0.194,   # corr=0.206
    "momentum_5m":           0.136,   # corr=0.173
    "cex_poly_lag":          0.136,   # corr=0.172 — the arb signal
    "momentum_1m":           0.111,   # corr=0.156, was underweighted
    "price_acceleration":    0.074,   # corr=0.098
    "momentum_15m":          0.034,   # corr=0.071, trend confirmation only
    "volume_ratio":          0.000,   # handled as pre-filter, not scorer
    "trade_flow_ratio":      0.000,   # corr=0.051, noise
    "order_imbalance":       0.000,   # corr=0.034, near zero
    "momentum_consistency":  0.000,   # corr=-0.021, negative
}
```

---

## CHANGE 3 — `composite_signal.py` — Fix Poly Divergence Filter

**Problem:** The current filter uses `poly_divergence > 0.05` (poly=0.55 threshold). But 95% of observations have poly=0.50–0.51, giving divergence=0.01. This filter **never fires**. Meanwhile when you weighted `poly_yes_price` in the composite it dragged scores to 0.500.

The correct approach: use `poly_yes_price` absolute value as a hard directional gate — don't include it in the composite score at all.

**Find** (in `apply_filters()`, near line 198):
```python
    # Polymarket already-priced-in filter
    poly_divergence = poly.get("poly_divergence", 0) or 0
    # Change from 0.08 to 0.05 — block earlier when Poly has already priced the move
    if m5 and m5 > 0 and poly_divergence > 0.05:
        return False, f"Poly already priced bullish ({poly_divergence:+.3f}), no edge left"
    if m5 and m5 < 0 and poly_divergence < -0.05:
        return False, f"Poly already priced bearish ({poly_divergence:+.3f}), no edge left"
```

**Replace with:**
```python
    # Polymarket directional gate — block when market has already priced in the move.
    # Use poly_yes_price absolute value (not divergence) for reliable filtering.
    # Edge zone: poly near 0.45–0.55 while CEX has moved = latency arb opportunity.
    poly_price = poly.get("poly_yes_price", 0.5) or 0.5
    cex_lag    = cex.get("cex_poly_lag", 0.0) or 0.0

    MAX_ENTRY_YES    = (config or {}).get("max_entry_yes",    0.72)
    MIN_ENTRY_YES    = (config or {}).get("min_entry_yes",    0.35)
    MIN_ENTRY_NO     = (config or {}).get("min_entry_no",     0.28)
    MAX_ENTRY_NO     = (config or {}).get("max_entry_no",     0.65)
    MIN_LAG_OVERRIDE = (config or {}).get("min_lag_override", 0.15)

    if m5 is not None and m5 > 0:  # signal wants YES
        if poly_price > MAX_ENTRY_YES:
            return False, f"Market priced in YES ({poly_price:.3f} > {MAX_ENTRY_YES}) — no edge"
        if poly_price < MIN_ENTRY_YES and abs(cex_lag) < MIN_LAG_OVERRIDE:
            return False, f"Poly disagrees ({poly_price:.3f}) and CEX lag too small ({cex_lag:.4f})"

    if m5 is not None and m5 < 0:  # signal wants NO
        if poly_price < MIN_ENTRY_NO:
            return False, f"Market priced in NO ({poly_price:.3f} < {MIN_ENTRY_NO}) — no edge"
        if poly_price > MAX_ENTRY_NO and cex_lag > -MIN_LAG_OVERRIDE:
            return False, f"Poly says UP ({poly_price:.3f}), CEX lag insufficient ({cex_lag:.4f})"
```

---

## CHANGE 4 — `composite_signal.py` — Volume Filter (0.00x avg bug)

**Problem:** Logs show `vol=1.00x` in display but `BLOCK: Volume too low (0.00x avg)`. 

The bug: `extract_cex_signals()` in `signal_research.py` already guards `avg_vol > 0` correctly. But `apply_filters()` in `composite_signal.py` receives the `vol_ratio` value directly — if the klines fetch returns only 1 candle (new window, cold start), `avg_vol` is computed over an empty slice and returns the fallback `1.0`, but a separate code path can produce `0.0`. The safe fix: don't block when `vol_ratio` is exactly `1.0` (the fallback sentinel) AND when `volume_confidence` is disabled.

**Find** (in `apply_filters()`, near line 193):
```python
    # Volume gate — only when volume_confidence is enabled in config
    volume_confidence = config.get("volume_confidence", True) if config else True
    if volume_confidence and vol_ratio is not None and vol_ratio < 0.3:
        return False, f"Volume too low ({vol_ratio:.2f}x avg)"
```

**Replace with:**
```python
    # Volume gate — only when volume_confidence is enabled in config.
    # Skip the check entirely when vol_ratio == 1.0 (fallback sentinel = no avg data yet).
    # This prevents false blocks on new market windows where avg_vol hasn't accumulated.
    volume_confidence = config.get("volume_confidence", False) if config else False
    if volume_confidence and vol_ratio is not None and vol_ratio != 1.0 and vol_ratio < 0.3:
        return False, f"Volume too low ({vol_ratio:.2f}x avg)"
```

**Note:** The default `volume_confidence` is also changed from `True` to `False` here, matching the config change. The `volume_confidence: false` in `config.json` (Change 8) is what controls this in practice.

---

## CHANGE 5 — `fast_trader.py` — Min Shares Constant

**Problem:** `MIN_SHARES_PER_ORDER = 5.1` allows orders of 5.15 shares which are below Polymarket CLOB's enforced minimum of 6 shares. The order submits (HTTP 200) but fills 0 shares. This has occurred in both observed sessions.

**Find** (near line 50):
```python
MIN_SHARES_PER_ORDER = 5.1
```

**Replace with:**
```python
MIN_SHARES_PER_ORDER = 6.0
```

That's it. The rest of the position sizing logic already uses `MIN_SHARES_PER_ORDER` correctly — it computes `min_order_usdc = MIN_SHARES_PER_ORDER * ask_price` and bumps `position_size` up to meet it. Changing `5.1` → `6.0` fixes the fill bug without any other changes needed in that block.

At `entry_price=0.495`: `6.0 * 0.495 = $2.97` — just under the $3 cap. ✓  
At `entry_price=0.510`: `6.0 * 0.510 = $3.06` — marginally over, acceptable. ✓

---

## CHANGE 6 — `fast_trader.py` — btc_vs_reference Seeding Bug (ROOT CAUSE)

**Problem:** `vs_ref=+0.0000%` appeared in 47% of observations on March 6. The seeding logic:

```python
window_just_opened = bool(
    event_start and (now_utc - event_start).total_seconds() < 60
)
```

**`event_start` is `None` in the Gamma API response for most fast markets.** When `event_start is None`, the expression `event_start and ...` short-circuits to `False` immediately — `window_just_opened` is always `False` — the reference price never seeds.

**Find** (in `run_fast_market_strategy()`, near line 195):
```python
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
```

**Replace with:**
```python
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
        # No event_start from API — treat as newly opened if not yet cached
        from signal_research import _load_ref_cache
        window_just_opened = market_slug not in _load_ref_cache()

    price_to_beat = get_window_reference_price(
        market_slug,
        cex_price_now=cex_signals.get("price_now"),
        window_open=window_just_opened,
    )
```

**Also apply the same fix in `signal_research.py` `collect_one()`** (near line 490):

**Find:**
```python
    event_start = best.get("event_start")
    window_just_opened = bool(
        event_start and (now - event_start).total_seconds() < 60
    )
```

**Replace with:**
```python
    event_start = best.get("event_start")
    if event_start is not None:
        window_just_opened = (now - event_start).total_seconds() < 60
    else:
        window_just_opened = best.get("slug", "") not in _load_ref_cache()
```

---

## CHANGE 7 — `fast_trader.py` — Add btc_vs_reference=0 Trade Block

**Problem:** Even after fixing seeding, there will be cases (new slug not yet in cache at cycle start) where `btc_vs_reference=0`. A composite score built without the 31.5%-weight dominant signal is unreliable.

**Find** (in `run_fast_market_strategy()`, after the `cex_signals["btc_vs_reference"]` assignment, before `get_composite_signal()` call):

```python
    vs_ref     = cex_signals.get("btc_vs_reference")
    vs_ref_str = f"{vs_ref:+.4f}%" if vs_ref is not None else "n/a"
    m5         = cex_signals.get("momentum_5m", 0) or 0
    vol_r      = cex_signals.get("volume_ratio", 1.0) or 1.0
    poly_p     = poly_signals.get("poly_yes_price", 0.5) or 0.5

    # ── Step 4: Composite signal ──────────────────────────────────────────────
    signal     = get_composite_signal(cex_signals, poly_signals, config=cfg)
```

**Add between those two blocks:**
```python
    # Block trade if dominant signal (btc_vs_reference, weight=0.315) is missing.
    # A score built without it is unreliable — better to skip than trade blind.
    if vs_ref is None or vs_ref == 0.0:
        log(
            f"{mode_tag} {now_str} | {slug_short} {remaining:4.0f}s | "
            f"m5={m5:+.3f}% vs_ref=MISSING poly={poly_p:.3f} vol={vol_r:.2f}x | "
            f"score=n/a BLOCK: btc_vs_reference not seeded"
        )
        return

    # ── Step 4: Composite signal ──────────────────────────────────────────────
    signal     = get_composite_signal(cex_signals, poly_signals, config=cfg)
```

**Remove this block once seeding fix (Change 6) has been running reliably for 24h.**

---

## CHANGE 8 — `config.json` — Full Replacement

**Problems in current config (from logs):**
- `btc_vs_reference: 0.20` — reduced from correct 0.315
- `poly_yes_price: 0.10` — caused 95% of scores to be exactly 0.500
- `poly_divergence: 0.10` — noise signal with 0.063 correlation
- `rsi_14: 0.05` — already a hard filter, double-penalizing
- `min_momentum_pct: 0.10` — blocks 84% of observations in flat markets
- `volume_confidence: true` — triggers the 0.00x avg bug
- `composite_threshold: 0.73` — only 1 obs out of 43 crossed it

**Replace entire `config.json` with:**
```json
{
  "entry_threshold": 0.05,
  "min_momentum_pct": 0.03,
  "max_position": 3.0,
  "min_time_remaining": 120,
  "max_time_remaining": 420,
  "asset": "BTC",
  "window": "5m",
  "signal_source": "binance",
  "volume_confidence": false,
  "daily_budget": 1000.0,
  "daily_import_limit": 1000.0,
  "composite_threshold": 0.70,
  "rsi_overbought": 85,
  "rsi_oversold": 15,
  "max_entry_yes": 0.72,
  "min_entry_yes": 0.35,
  "min_entry_no": 0.28,
  "max_entry_no": 0.65,
  "min_lag_override": 0.15,
  "webhook_url": "",
  "telegramlink": "configure bot",
  "signal_weights": {
    "btc_vs_reference":      0.315,
    "vol_adjusted_momentum": 0.194,
    "momentum_5m":           0.136,
    "cex_poly_lag":          0.136,
    "momentum_1m":           0.111,
    "price_acceleration":    0.074,
    "momentum_15m":          0.034,
    "rsi_14":                0.0,
    "trade_flow_ratio":      0.0,
    "volume_ratio":          0.0,
    "order_imbalance":       0.0,
    "momentum_consistency":  0.0,
    "poly_divergence":       0.0,
    "poly_yes_price":        0.0
  }
}
```

**New fields used by Changes 3 and the RSI fix:**
- `rsi_overbought / rsi_oversold` — read in `composite_signal.apply_filters()` after Change 9 below
- `max_entry_yes / min_entry_yes / min_entry_no / max_entry_no / min_lag_override` — read by the poly gate in Change 3

---

## CHANGE 9 — `composite_signal.py` — Read RSI Thresholds from Config

**Problem:** `RSI_OVERBOUGHT = 72` is a module-level constant — changing it requires a code deploy. Should be config-driven like `composite_threshold` is.

**Find** (in `apply_filters()`, the RSI check block near line 175):
```python
    # RSI filters — only apply when we have both RSI and a clear direction
    if rsi is not None:
        if m5 is not None and m5 > 0 and rsi > RSI_OVERBOUGHT:
            return False, f"RSI overbought ({rsi:.0f} > {RSI_OVERBOUGHT}), skip long"
        if m5 is not None and m5 < 0 and rsi < RSI_OVERSOLD:
            return False, f"RSI oversold ({rsi:.0f} < {RSI_OVERSOLD}), skip short"
```

**Replace with:**
```python
    # RSI filters — thresholds read from config if available, else module defaults
    rsi_ob = (config or {}).get("rsi_overbought", RSI_OVERBOUGHT)
    rsi_os = (config or {}).get("rsi_oversold",   RSI_OVERSOLD)
    if rsi is not None:
        if m5 is not None and m5 > 0 and rsi > rsi_ob:
            return False, f"RSI overbought ({rsi:.0f} > {rsi_ob}), skip long"
        if m5 is not None and m5 < 0 and rsi < rsi_os:
            return False, f"RSI oversold ({rsi:.0f} < {rsi_os}), skip short"
```

---

## CHANGE 10 — `composite_signal.py` — Read MIN_MOMENTUM from Config

**Problem:** `MIN_MOMENTUM_ABS = 0.05` is hardcoded. The last config had `min_momentum_pct: 0.10` but the code never read it — the filter stayed at 0.05. Conversely when you set `min_momentum_pct: 0.03` in config it also had no effect. These need to be wired together.

**Find** (in `apply_filters()`, near line 162):
```python
    # Minimum momentum filter
    if m5 is not None and abs(m5) < MIN_MOMENTUM_ABS:
        return False, f"momentum too weak ({m5:+.3f}% < ±{MIN_MOMENTUM_ABS}%)"
```

**Replace with:**
```python
    # Minimum momentum filter — read from config, fall back to module default
    min_mom = (config or {}).get("min_momentum_pct", MIN_MOMENTUM_ABS)
    if m5 is not None and abs(m5) < min_mom:
        return False, f"momentum too weak ({m5:+.3f}% < ±{min_mom}%)"
```

---

## CHANGE 11 — `fast_trader.py` — Add max_time_remaining Filter

**Problem:** `CONFIG_SCHEMA` has `min_time_remaining` but no `max_time_remaining`. From the data analysis, observations beyond 420s have no edge. Markets at 380–420s remaining are new windows where vs_ref hasn't seeded — blocking these early reduces bad trades.

**Find** (in `CONFIG_SCHEMA` dict, near line 85):
```python
    "min_time_remaining": {"default": 30,        "env": "SIMMER_SPRINT_MIN_TIME",     "type": int},
```

**Add the line below it:**
```python
    "min_time_remaining": {"default": 120,       "env": "SIMMER_SPRINT_MIN_TIME",     "type": int},
    "max_time_remaining": {"default": 420,       "env": "SIMMER_SPRINT_MAX_TIME",     "type": int},
```

**Also update** the `cfg` variable reads near line 120:
```python
MIN_TIME_REMAINING = cfg["min_time_remaining"]
```
**Add below it:**
```python
MAX_TIME_REMAINING = cfg.get("max_time_remaining", 420)
```

**And add the check** in `run_fast_market_strategy()` where `remaining` is computed and `find_best_fast_market` filters by min time. Find the block:
```python
    best = find_best_fast_market(markets, min_time_remaining=MIN_TIME_REMAINING)
    if not best:
        log(f"{mode_tag} {now_str} | no market with >{MIN_TIME_REMAINING}s left")
        return
```

**Replace with:**
```python
    best = find_best_fast_market(markets, min_time_remaining=MIN_TIME_REMAINING)
    if not best:
        log(f"{mode_tag} {now_str} | no market with >{MIN_TIME_REMAINING}s left")
        return

    end_time  = best.get("end_time")
    remaining = (end_time - datetime.now(timezone.utc)).total_seconds() if end_time else 0

    if remaining > MAX_TIME_REMAINING:
        log(
            f"{mode_tag} {now_str} | {best.get('slug','')[-24:]} {remaining:.0f}s | "
            f"SKIP: too early ({remaining:.0f}s > {MAX_TIME_REMAINING}s max) — vs_ref not seeded yet"
        )
        return
```

*Remove the duplicate `end_time` / `remaining` computation that appears a few lines later — it will now be a duplicate.*

---

## CHANGE 12 — `scheduler.py` — Confirm Interval (Already Correct)

**The scheduler already defaults to 20s** (`INTERVAL = int(os.environ.get("PKNWITQ_INTERVAL", "20"))`). **No code change needed.**

What you need to check is your Docker run command or `docker-compose.yml`. If it passes `-e PKNWITQ_INTERVAL=60`, that overrides the default. Remove that env var or set it to 20:

```bash
# Check current running container env
docker inspect pknwitq-collector | grep -A5 INTERVAL

# If it shows 60, redeploy without that env var, or add:
# -e PKNWITQ_INTERVAL=20
```

---

## Deployment Order

```bash
# 1. Edit config.json (Change 8) — no restart needed if hot-reload works
#    Otherwise it's read at module load, so still need a restart

# 2. Edit composite_signal.py (Changes 1, 2, 3, 4, 9, 10)
#    All in one file, do all 6 changes together

# 3. Edit fast_trader.py (Changes 5, 6, 7, 11)
#    All in one file, do all 4 changes together

# 4. Edit signal_research.py (Change 6 — collect_one seeding fix)

# 5. Rebuild and redeploy
cd ~/poly_trader
docker build -t fastloop .
docker stop pknwitq-collector
docker run -d --name pknwitq-collector \
  --restart unless-stopped \
  -v /data:/data \
  -v $(pwd)/config.json:/app/config.json \
  fastloop

# 6. Verify fixes are working — watch for these in logs:
docker logs -f pknwitq-collector | grep -E "vs_ref|BLOCK|TRADED|score"

# Good signs:
#   vs_ref=+0.0XXX%    (non-zero — seeding fixed)
#   score=0.7XX        (not stuck at 0.500 — weight fix working)
#   BLOCK: RSI overbought (XX > 85)   (only extreme RSI blocked now)
#   TRADED X.X YES/NO shares          (non-zero — min shares fixed)

# Bad signs still present:
#   vs_ref=+0.0000%    (seeding still broken)
#   score=0.500        (weight bug still present)
#   TRADED 0.0 shares  (min shares still 5.1)
```

---

## Summary Table

| # | File | Lines | What Changes | Fixes |
|---|------|-------|-------------|-------|
| 1 | `composite_signal.py` | ~53–54 | RSI 72/28 → 85/15 constants | RSI blocking whole uptrends |
| 2 | `composite_signal.py` | ~30–41 | `DEFAULT_WEIGHTS` full replace | Wrong signal weights as fallback |
| 3 | `composite_signal.py` | ~198–205 | Poly divergence → poly gate | `poly_yes_price` dragging scores to 0.500 |
| 4 | `composite_signal.py` | ~193–196 | Volume filter sentinel guard | `Volume too low (0.00x avg)` false blocks |
| 5 | `fast_trader.py` | ~50 | `MIN_SHARES_PER_ORDER` 5.1 → 6.0 | `TRADED 0.0 shares` 0-fill bug |
| 6 | `fast_trader.py` + `signal_research.py` | ~195, ~490 | `event_start` None handling | `vs_ref=0.0000` in 47% of obs |
| 7 | `fast_trader.py` | ~210 | Block when `vs_ref==0.0` | Trades on unreliable scores |
| 8 | `config.json` | all | Full replacement | All bad config values |
| 9 | `composite_signal.py` | ~175 | RSI thresholds read from config | Can tune RSI without deploy |
| 10 | `composite_signal.py` | ~162 | `min_momentum_pct` from config | Config value was silently ignored |
| 11 | `fast_trader.py` | ~87, ~120, ~230 | Add `max_time_remaining` | Bot trading at 380s+ before vs_ref seeds |
| 12 | `scheduler.py` | — | No change needed | Already defaults to 20s |