# Pknwitq — Polymarket BTC 5-Minute Trading System
## Complete Operations Manual

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites](#3-prerequisites)
4. [First-Time Setup](#4-first-time-setup)
5. [Operating Modes](#5-operating-modes)
6. [Three-Phase Workflow](#6-three-phase-workflow)
7. [Docker Commands Reference](#7-docker-commands-reference)
8. [Dashboard Guide](#8-dashboard-guide)
9. [Log Format](#9-log-format)
10. [Configuration Reference](#10-configuration-reference)
11. [Manually Changing Config](#11-manually-changing-config)
12. [Signal Engine](#12-signal-engine)
13. [Market Regime Detection](#13-market-regime-detection)
14. [Market Session Awareness](#14-market-session-awareness)
15. [Noise Filters and Risk Controls](#15-noise-filters-and-risk-controls)
16. [Optimizer](#16-optimizer)
17. [Signal Recalibration](#17-signal-recalibration)
18. [Monitoring and Alerts](#18-monitoring-and-alerts)
19. [Troubleshooting](#19-troubleshooting)
20. [File Structure Reference](#20-file-structure-reference)

---

## 1. What This System Does

Every 20 seconds, the system:

1. **Discovers** the active BTC 5-minute market on Polymarket (e.g. `btc-updown-5m-1740441600`)
2. **Fetches** real-time data from Binance (BTCUSDT candles, order book, trades) and Polymarket (prices, order flow)
3. **Scores** the opportunity using a multi-factor weighted composite signal (0 = strongly bearish, 0.5 = neutral, 1 = strongly bullish)
4. **Filters** the trade through regime detection, session gates, time gates, noise filters, and risk controls
5. **Executes** a BUY YES or BUY NO order if all checks pass, sizing the position by confidence
6. **Tracks** outcomes 10 minutes after each market closes and feeds results back into the optimizer
7. **Auto-optimizes** key parameters every 30 minutes based on what is actually winning

The goal is a sustained **70%+ win rate** on executed trades.

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                      Docker Compose                                │
│                                                                    │
│  ┌──────────────────────────────┐   ┌───────────────────────────┐  │
│  │     COLLECTOR SERVICE        │   │     MONITOR SERVICE       │  │
│  │  (container: pknwitq-collector)│ │  (container: pknwitq-monitor)│ │
│  │                              │   │                           │  │
│  │  scheduler.py  ← entry point │   │  monitor_api.py ← Flask   │  │
│  │    ↓ every 20s               │   │  Serves:                  │  │
│  │  fast_trader.py              │   │   /live.html  (P&L)       │  │
│  │    ↓ signal scoring          │   │   /dry-run.html (lab)     │  │
│  │  composite_signal.py         │   │   /index.html (monitor)   │  │
│  │    ↓ data collection         │   │   /api/* (REST API)       │  │
│  │  signal_research.py          │   │                           │  │
│  │    ↓ trade execution         │   │  Port 5000 → host         │  │
│  │  polymarket_sdk/             │   │                           │  │
│  │    ↓ every 30 min            │   │                           │  │
│  │  optimizer.py                │   │                           │  │
│  └──────────────────────────────┘   └───────────────────────────┘  │
│                    ↕ shared volume                  ↕               │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  /data/  (persistent volume → ./data/)         │ │
│  │  config.json       ← live optimizer-tuned config               │ │
│  │  trade_log.json    ← all executed trades + outcomes            │ │
│  │  signal_research.db← SQLite: signals, observations, outcomes   │ │
│  │  optimizer.log     ← timestamped parameter change history      │ │
│  │  market_id_cache.json ← slug→condition_id cache               │ │
│  │  daily_spend.json  ← budget + quota tracking                   │ │
│  │  status.json       ← current cycle status (dashboard feed)     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                    ↕ bind mount                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                  ./logs/  (bind mount)                         │ │
│  │  collector.log   ← unified stdout+stderr                       │ │
│  │  scheduler.log   ← scheduler-specific log                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

**Config Priority (highest to lowest):**
1. `/data/config.json` — live, optimizer-writable, survives restarts and rebuilds
2. `/app/config.json` — image default, baked into Docker image (reset point only)
3. Environment variables in `.env`

---

## 3. Prerequisites

- **Docker Desktop** (or Docker Engine + Compose plugin)
- **Polymarket account** with:
  - Private key (`POLY_PRIVATE_KEY`)
  - API key + secret + passphrase
  - Funded wallet (`POLY_FUNDER`) with USDC on Polygon
- **Internet access** for Binance and Polymarket APIs

---

## 4. First-Time Setup

### Step 1 — Clone and configure environment

```bash
# Copy the example environment file
cp .env.poly.example .env
```

Edit `.env` and fill in your credentials:

```env
# Polymarket authentication
POLY_PRIVATE_KEY=0x...          # Your wallet private key
POLY_API_KEY=...                # From Polymarket API settings
POLY_API_SECRET=...
POLY_API_PASSPHRASE=...
POLY_FUNDER=0x...               # Wallet address that holds USDC

# Trading mode (start with dry)
PKNWITQ_MODE=dry
PKNWITQ_INTERVAL=20             # Seconds between cycles
PKNWITQ_ASSET=BTC
PKNWITQ_WINDOW=5m

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Dashboard authentication
SECRET_KEY=...                  # Random string for session security
```

### Step 2 — Create required directories

```bash
mkdir -p data logs
```

### Step 3 — Build and start

```bash
docker compose up -d
```

First boot automatically seeds `/data/config.json` from the image default. You do not need to create it manually.

### Step 4 — Verify it is running

```bash
docker compose logs -f collector
```

You should see lines like:
```
[DRY] 14:32:05 | m-1740441600 287s | m5=-0.025% ... | score=0.282 BLOCK: momentum too weak
```

### Step 5 — Open dashboard

```
http://localhost:5000/live.html
```

---

## 5. Operating Modes

Set `PKNWITQ_MODE` in `.env` (then `docker compose up -d collector` to apply):

| Mode      | Collects Signals | Executes Trades | Use For                                  |
|-----------|:----------------:|:---------------:|------------------------------------------|
| `collect` | Yes              | No              | Building the initial signal dataset      |
| `dry`     | Yes              | No (simulated)  | Validating signals before spending money |
| `both`    | Yes              | Yes (live)      | Full live trading                        |
| `trade`   | No               | Yes (live)      | Trade-only (rare, no data collection)    |

> Start with `dry`. Only switch to `both` after you have seen 55%+ win rate in dry mode over at least 50 resolved predictions.

---

## 6. Three-Phase Workflow

### Phase 1 — Data Collection (2–5 days)

```env
PKNWITQ_MODE=collect
```

Goal: Build 200+ resolved signal observations so the optimizer has data to work with.

```bash
# Check collection progress
docker exec pknwitq-collector python signal_research.py --analyze --min-n 30
```

Watch for:
- Total resolved > 100
- At least some variation in outcomes (not all "up" or all "down")

### Phase 2 — Dry Run Validation

```env
PKNWITQ_MODE=dry
```

Goal: Confirm signals predict outcomes correctly before spending real money.

Monitor at: `http://localhost:5000/dry-run.html`

Minimum bar to proceed to live:
- **Win rate > 55%** on resolved predictions (52.1% breaks even after fees)
- At least **50 resolved** predictions
- Optimizer has run at least once (check `/data/optimizer.log`)

### Phase 3 — Live Trading

```env
PKNWITQ_MODE=both
```

```bash
docker compose up -d collector    # Apply the mode change, no rebuild needed
```

Monitor at: `http://localhost:5000/live.html`

Key targets:
- Win rate > 70% (system target)
- Daily P&L positive after fees
- No single day losing more than your `daily_budget`

---

## 7. Docker Commands Reference

### Start / Stop

```bash
# Start everything
docker compose up -d

# Stop everything (data preserved)
docker compose down

# Restart just the collector (picks up .env changes)
docker compose up -d collector

# Restart just the monitor (picks up monitor_api.py changes)
docker compose restart monitor
```

### Rebuilding After Code Changes

```bash
# Rebuild collector image after changing trader/*.py or shared/*.py
docker compose up -d --build collector

# Rebuild monitor image after changing monitor/*.py (not needed for monitor_api.py — it is volume-mounted)
docker compose up -d --build monitor

# Full rebuild, both services
docker compose build --no-cache && docker compose up -d
```

> **Config is safe during rebuilds.** The optimizer writes to `/data/config.json` (persistent volume), not the image. `docker compose up -d --build` will NOT reset your optimized config.

### Viewing Logs

```bash
# Follow collector logs
docker compose logs -f collector

# Follow monitor logs
docker compose logs -f monitor

# Last 100 lines
docker compose logs --tail=100 collector
```

### Utility Commands

```bash
# Run signal correlation analysis
docker exec pknwitq-collector python signal_research.py --analyze --min-n 30

# Force resolve pending outcomes now
docker exec pknwitq-collector python signal_research.py --resolve

# Export all observations to CSV
docker exec pknwitq-collector python signal_research.py --export /data/signals.csv
# File appears at ./data/signals.csv on host

# Pre-warm market ID cache (if getting rate-limit errors at startup)
docker exec pknwitq-collector python seed_cache.py

# Reset optimizer config to factory defaults
docker exec pknwitq-collector cp /app/config.json /data/config.json

# Quick DB stats
docker exec pknwitq-collector python -c "
import sqlite3
c = sqlite3.connect('/data/signal_research.db')
total    = c.execute('SELECT COUNT(*) FROM signal_observations').fetchone()[0]
resolved = c.execute('SELECT COUNT(*) FROM signal_observations WHERE resolved=1').fetchone()[0]
up       = c.execute(\"SELECT COUNT(*) FROM signal_observations WHERE outcome='up'\").fetchone()[0]
down     = c.execute(\"SELECT COUNT(*) FROM signal_observations WHERE outcome='down'\").fetchone()[0]
print(f'Total: {total}  Resolved: {resolved}  Up: {up}  Down: {down}')
"

# Full wipe and fresh start
docker compose down && rm -rf ./data && mkdir data && docker compose up -d
```

---

## 8. Dashboard Guide

Open `http://localhost:5000` in your browser.

### Monitor (`/index.html`)

Overview of signal collection health.

| Panel | What It Shows |
|-------|---------------|
| Status strip | Cycle count, DB total, resolved count, up/down outcome split |
| Current Window | Active market slug, seconds remaining, YES price, BTC vs reference |
| Signal Bars | Live values for each sub-signal |
| Recent Windows | Last 20 five-minute windows with avg signals and final outcome |

### Signal Lab (`/dry-run.html`)

Real-time composite score with full breakdown.

| Panel | What It Shows |
|-------|---------------|
| Live Composite Score | Current score (0–1), gauge, BUY/SELL decision, per-signal breakdown |
| Score History | 100-cycle SVG chart with threshold lines and trade markers |
| Would-Trade Log | Every cycle where score crossed threshold: side, score, outcome |
| Prediction vs Outcome | All resolved predictions with win rate, YES/NO accuracy |
| Sidebar — Weights | Active signal weights |
| Sidebar — Session Stats | Cycles scored, trade rate, avg score, top block reason |
| Sidebar — Prediction Accuracy | Win rate, YES accuracy, NO accuracy, avg score on correct vs wrong |

### Live Trading (`/live.html`)

P&L and trade history for live mode.

| Panel | What It Shows |
|-------|---------------|
| P&L Strip | Today P&L, Total P&L, Win Rate, Budget used, Open Positions, Avg per trade |
| Cumulative P&L Chart | Running P&L over all trades |
| Trade History | Every executed trade: side, amount, entry price, outcome, P&L |
| Open Positions | Trades with unresolved outcomes |

---

## 9. Log Format

Each 20-second cycle produces a single summary line:

```
[LIVE/slow/asia/--] 14:32:05 | m-1740441600 287s | m5=-0.025% poly=0.485 vol=1.12x | score=0.282 BLOCK: momentum too weak
[LIVE/normal/london/ok] 14:35:08 | m-1740441600 104s | m5=-0.085% poly=0.482 vol=1.44x | score=0.208 conf=0.58 → NO $2.90
  [LIVE] Bought NO $2.90 (~5.9 shares @ $0.490)
```

| Field | Meaning |
|-------|---------|
| `[LIVE]` / `[DRY]` | Operating mode |
| `/slow/` `/normal/` `/active/` | Market regime (volatility/momentum level) |
| `/asia/` `/london/` `/overlap/` `/new_york/` `/off/` | UTC market session |
| `/ok/` / `/--/` | Hour accuracy gate (ok = not blocked) |
| `m-1740441600` | Last 10 chars of market slug |
| `287s` | Seconds remaining in this window |
| `m5=` | 5-minute Binance momentum % |
| `poly=` | Current Polymarket YES price |
| `vol=` | Volume ratio (current ÷ average) |
| `score=` | Composite signal score |
| `BLOCK:` | Why the trade was skipped |
| `conf=` | Confidence multiplier for position sizing |
| `→ YES/NO $X.XX` | Trade direction and dollar size |

### Common Block Reasons

| Block Reason | Meaning | Action |
|---|---|---|
| `momentum too weak` | `|momentum_5m|` below threshold | Normal in quiet markets. Optimizer will adjust threshold. |
| `too early` | `seconds_remaining` > `max_time_remaining` | Market just opened. Wait. |
| `too late` | `seconds_remaining` < `min_time_remaining` | Too close to close. Skip. |
| `score in neutral band` | Score between thresholds | Signal is ambiguous. |
| `timeframe disagreement` | 1m and 5m momentum point opposite directions | Likely reversal forming. Noise filter active. |
| `daily budget exhausted` | Hit `daily_budget` for today | Resets at midnight UTC. |
| `hour blocked` | Trading blocked at this UTC hour (historically anti-predictive) | Hours 6, 8, 16 blocked by default. |
| `COOLDOWN` | Last N resolved trades were all losses | Waiting one cycle before resuming. |
| `kill switch` | Hard stop triggered | Check kill switch setting in config. |

---

## 10. Configuration Reference

The live config lives at `./data/config.json` and is auto-optimized every 30 minutes. The full set of parameters:

### Core Trading

| Key | Default | Description |
|-----|---------|-------------|
| `min_momentum_pct` | 0.08 | Minimum `|momentum_5m|` % required to trade |
| `max_position` | 3.5 | Maximum USD per trade (before confidence scaling) |
| `max_position_per_window` | 4.0 | Maximum USD invested in any single 5-minute window |
| `max_no_score` | 0.25 | Maximum composite score to enter a NO trade |
| `min_time_remaining` | 200 | Don't trade if < N seconds remain in window |
| `max_time_remaining` | 350 | Don't trade if > N seconds remain (market too young) |
| `daily_budget` | 1000.0 | Total USD budget per day |
| `composite_threshold` | 0.70 | Score threshold for BUY YES / BUY NO |

### Price Gates

| Key | Default | Description |
|-----|---------|-------------|
| `min_entry_yes` | 0.35 | Minimum YES price to buy YES |
| `max_entry_yes` | 0.60 | Maximum YES price to buy YES (above = priced in) |
| `min_entry_no` | 0.28 | Minimum NO price to buy NO |
| `max_entry_no` | 0.65 | Maximum NO price to buy NO |
| `min_entry_price` | 0.35 | Hard minimum entry price for any side |
| `min_payout_ratio` | 0.90 | Minimum `(1/price)` required for normal regime |
| `slow_min_payout_ratio` | 1.10 | Minimum payout ratio for slow regime (higher bar) |

### Signal Weights

These must sum to approximately 1.0. Set a signal to `0.0` to exclude it.

| Signal | Default Weight | What It Measures |
|--------|:--------------:|------------------|
| `vol_adjusted_momentum` | 0.28 | Momentum per unit of volatility |
| `momentum_5m` | 0.22 | 5-minute price trend % |
| `cex_poly_lag` | 0.20 | CEX-to-Polymarket price divergence |
| `momentum_15m` | 0.12 | 15-minute confirmation trend |
| `btc_vs_reference` | 0.08 | % change since this window opened |
| `price_acceleration` | 0.05 | Rate of change of momentum |
| `momentum_1m` | 0.03 | 1-minute micro-move |
| `rsi_14` | 0.02 | RSI overbought/oversold |
| `trade_flow_ratio` | 0.00 | (excluded — negative edge) |
| `volume_ratio` | 0.00 | (excluded) |
| `order_imbalance` | 0.00 | (excluded) |

### Market Regime Thresholds

| Key | Default | Description |
|-----|---------|-------------|
| `slow_mom_threshold` | 0.12 | Below this momentum = slow regime |
| `slow_vol_threshold` | 0.80 | Below this volume ratio = slow regime |
| `active_mom_threshold` | 0.25 | Above this momentum = active regime |
| `active_vol_threshold` | 1.30 | Above this volume ratio = active regime |
| `slow_composite_threshold` | 0.73 | Higher bar in slow (quiet) markets |
| `active_composite_threshold` | 0.68 | Lower bar in active (liquid) markets |
| `slow_max_entry_yes` | 0.54 | Tighter price gate in slow regime |
| `slow_min_entry_no` | 0.46 | Tighter price gate in slow regime |
| `slow_position_pct_cap` | 0.70 | Max position size in slow regime (70%) |
| `normal_position_pct_cap` | 0.90 | Max position size in normal regime (90%) |
| `active_position_pct_cap` | 1.00 | Max position size in active regime (100%) |

### Volatility Controls

| Key | Default | Description |
|-----|---------|-------------|
| `max_volatility_5m` | 2.0 | Above this 5m volatility, trade is blocked entirely |
| `vol_penalty_threshold` | 1.2 | Above this, position size starts scaling down (up to 50% reduction) |

### Noise Filters

| Key | Default | Description |
|-----|---------|-------------|
| `momentum_agreement_filter` | true | Block trades when 1m and 5m momentum disagree |
| `momentum_agreement_min_1m` | 0.05 | Minimum `|momentum_1m|` needed to trigger the disagreement filter |

### Risk Controls

| Key | Default | Description |
|-----|---------|-------------|
| `loss_cooldown_enabled` | true | Skip one cycle after N consecutive losses |
| `loss_cooldown_streak` | 3 | Number of consecutive losses before cooldown triggers |

### Hour and Session Gating

| Key | Default | Description |
|-----|---------|-------------|
| `blocked_hours` | [6, 8, 16] | UTC hours where trading is fully blocked |
| `boosted_hours` | `{"2": -0.03, "4": -0.02, ...}` | UTC hours where threshold is lowered (easier to trade) |
| `session_gating_enabled` | true | Enable per-session threshold and cap adjustments |
| `overlap_threshold_delta` | -0.03 | Threshold adjustment during London-NY overlap (13–16h UTC) |
| `london_threshold_delta` | -0.02 | Threshold adjustment during London (7–13h UTC) |
| `new_york_threshold_delta` | -0.01 | Threshold adjustment during New York (16–21h UTC) |
| `asia_threshold_delta` | 0.00 | No adjustment during Asia session (0–7h UTC) |
| `off_threshold_delta` | +0.02 | Harder to trade during off-hours (21–24h UTC) |
| `overlap_position_cap` | 1.00 | Max position size during London-NY overlap |
| `london_position_cap` | 0.95 | Max position size during London |
| `new_york_position_cap` | 0.90 | Max position size during New York |
| `asia_position_cap` | 0.85 | Max position size during Asia |
| `off_position_cap` | 0.70 | Max position size during off-hours |

### RSI Gates

| Key | Default | Description |
|-----|---------|-------------|
| `rsi_overbought` | 85 | RSI above this blocks YES buys |
| `rsi_oversold` | 15 | RSI below this blocks NO buys |

---

## 11. Manually Changing Config

You can change any config value at any time without restarting the container. The trader reloads config on every cycle.

### Method 1 — Edit `./data/config.json` directly (easiest)

```bash
# On Windows — open in notepad
notepad ./data/config.json

# Or any editor
code ./data/config.json
```

Change any value, save the file. The next cycle picks it up automatically.

### Method 2 — Via the REST API

```bash
# Change one value
curl -X POST http://localhost:5000/api/config \
     -H "Content-Type: application/json" \
     -d '{"max_position": 5.0}'

# Change multiple values at once
curl -X POST http://localhost:5000/api/config \
     -H "Content-Type: application/json" \
     -d '{"max_position": 5.0, "min_momentum_pct": 0.10, "daily_budget": 50.0}'
```

Only the keys you send are changed. All other values are preserved.

### Method 3 — Via docker exec

```bash
docker exec -it pknwitq-collector python -c "
import json
p = '/data/config.json'
c = json.load(open(p))
c['max_position'] = 5.0
json.dump(c, open(p, 'w'), indent=2)
print('done')
"
```

### Reset to Factory Defaults

```bash
docker exec pknwitq-collector cp /app/config.json /data/config.json
```

This overwrites `/data/config.json` with the original image defaults. All optimizer changes are lost. The next cycle immediately loads the reset config.

---

## 12. Signal Engine

The composite signal is a weighted sum of normalized sub-signals, each mapped to [0, 1] where 0.5 = neutral.

### How the Score Maps to Decisions

| Score | Decision |
|-------|----------|
| > `composite_threshold` (default 0.70) | BUY YES |
| < `1 - composite_threshold` (default 0.30) | BUY NO |
| Between 0.30 and 0.70 | No trade (neutral band) |

**Confidence** = `|score - 0.5| × 2` — ranges 0 (neutral) to 1 (maximum conviction).

**Position size** = `confidence × max_position`, further adjusted by:
- Regime position cap (slow/normal/active)
- Session position cap (Asia/London/overlap/New York/off)
- Volatility penalty (reduces size when 5m volatility is high)

### Sub-Signals Explained

| Signal | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| `vol_adjusted_momentum` | `momentum_5m ÷ volatility_5m` | Momentum normalized for turbulence — avoids overweighting noisy moves |
| `momentum_5m` | % price change over last 5 minutes | Direct trend signal |
| `cex_poly_lag` | CEX price vs Polymarket implied price | Measures how much Polymarket lags Binance — a lag is an edge |
| `momentum_15m` | % price change over last 15 minutes | Confirms whether the 5m move is part of a larger trend |
| `btc_vs_reference` | % change from window-open price | Directly answers whether BTC is up/down from the start of this market |
| `price_acceleration` | Rate of change of `momentum_5m` | Catches trend acceleration before it fully shows in momentum |
| `momentum_1m` | 1-minute price change | Micro-move confirmation |
| `rsi_14` | 14-period RSI | Overbought/oversold filter |

### Hour Accuracy (UTC)

Calibrated from empirical observations:

| Hours | Historical Accuracy | Effect |
|-------|:-------------------:|--------|
| 2h, 4h | 77–86% | Threshold lowered by 0.02–0.03 (boosted) |
| 10h, 17h | 73% | Threshold lowered by 0.02 (boosted) |
| 0h | ~71% | Threshold lowered by 0.01 |
| 6h, 8h, 16h | 25–46% | **Blocked entirely** |
| Other hours | ~65% | No adjustment |

---

## 13. Market Regime Detection

The system detects the current market regime each cycle based on momentum and volume:

| Regime | Condition | Behavior |
|--------|-----------|----------|
| **slow** | Low momentum AND low volume | Higher composite threshold, tighter price gates, smaller max position |
| **normal** | Neither slow nor active | Standard config values |
| **active** | High momentum OR high volume | Lower threshold (easier to trade), larger position allowed |

Regime transitions happen automatically each cycle — no configuration needed.

**Slow regime example:** BTC is flat, low volume. The system requires a stronger signal before trading and limits position to 70% of max.

**Active regime example:** BTC is breaking out with 2x+ volume. The system requires a weaker signal (more opportunities) and allows full position size.

---

## 14. Market Session Awareness

The system tracks UTC market sessions and adjusts the composite threshold and position cap accordingly:

| Session | UTC Hours | Strategy |
|---------|-----------|----------|
| **overlap** | 13:00–16:00 | London + NY both open → peak liquidity → threshold lowered most, full position cap |
| **london** | 07:00–13:00 | London open → good liquidity → threshold lowered moderately |
| **new_york** | 16:00–21:00 | NY afternoon → moderate liquidity → threshold lowered slightly |
| **asia** | 00:00–07:00 | Lower liquidity → no threshold adjustment, reduced cap |
| **off** | 21:00–00:00 | Very low liquidity → threshold raised (harder to enter), lowest cap |

Session adjustments stack with regime adjustments. For example, during the London-NY overlap in an active regime, the system uses the lowest threshold and the highest position cap.

The current session is shown in every log line: `[LIVE/normal/overlap/ok]`.

---

## 15. Noise Filters and Risk Controls

### Cross-Timeframe Agreement Filter

Blocks trades where 1-minute and 5-minute momentum are pointing in **opposite directions** — a sign of a likely price reversal.

- Enabled when `momentum_agreement_filter: true`
- Only triggers when `|momentum_1m|` ≥ `momentum_agreement_min_1m` (default 0.05%)
- Log shows: `BLOCK: timeframe disagreement: 1m=+0.12% opposes 5m=-0.08% — reversal risk`

### Volatility Position Penalty

When 5-minute volatility exceeds `vol_penalty_threshold` (default 1.2), position size is scaled down linearly. At twice the threshold, position is reduced by 50%.

At `max_volatility_5m` (default 2.0), the trade is blocked entirely.

### Consecutive Loss Cooldown

After `loss_cooldown_streak` consecutive resolved losses, the system skips one trading cycle. This prevents the system from aggressively adding losses during a signal breakdown.

- Log shows: `COOLDOWN: last 3 resolved trades all losses — sitting out this cycle`
- Re-enables automatically on the next cycle

### Time Gate

- `min_time_remaining` — do not enter a market with fewer than N seconds remaining (default 200s ≈ 3:20). Avoids last-minute entries with high adverse selection.
- `max_time_remaining` — do not enter a market with more than N seconds remaining (default 350s). Avoids entering at the very start when the outcome is most uncertain.

### Hour Block

UTC hours 6, 8, and 16 are fully blocked based on historical anti-predictive performance (as low as 25% accuracy).

---

## 16. Optimizer

The optimizer runs automatically every 30 minutes inside the collector container. It reads `trade_log.json` and `signal_research.db`, analyzes win rates by parameter, and updates `/data/config.json` with bounded adjustments.

### What the Optimizer Tunes

| Parameter Group | What Gets Tuned |
|----------------|----------------|
| Composite threshold | Raised if win rate is low, lowered if win rate is high |
| Momentum threshold | Swept from 0.04–0.25 in bands to find the cutoff that maximises win rate |
| Time window | Jointly sweeps all `(min_time, max_time)` pairs (120–460s, 15s steps) to find the best window |
| Position caps | Slow/normal/active regime caps adjusted based on regime-specific win rates |
| Session deltas | Per-session threshold adjustments tuned if enough session-specific trades exist |
| Session caps | Per-session position caps adjusted based on session-specific win rates |
| RSI gates | Overbought/oversold levels adjusted |
| Noise filter | `momentum_agreement_min_1m` threshold adjusted |

### Safety Bounds

The optimizer never makes a change that could destabilize the system:
- Composite threshold: 0.62–0.82
- Momentum threshold: 0.04–0.25
- Min time remaining: 120–300 seconds
- Max time remaining: 240–460 seconds
- Max change per run: 0.01 per numeric parameter
- Minimum sample size before any change: 5 resolved trades

### Viewing Optimizer History

```bash
# See all changes the optimizer has made
cat ./data/optimizer.log

# Or inside the container
docker exec pknwitq-collector cat /data/optimizer.log
```

### Running the Optimizer Manually

```bash
# Dry run — shows what it would change
docker exec pknwitq-collector python optimizer.py --report

# Apply changes now (same as auto-run)
docker exec pknwitq-collector python optimizer.py --apply

# Continuous watch mode (apply every 30 minutes)
docker exec pknwitq-collector python optimizer.py --watch
```

---

## 17. Signal Recalibration

After collecting 200+ resolved observations, you can check if signal weights should be updated:

```bash
docker exec pknwitq-collector python signal_research.py --analyze --min-n 30
```

Key columns to read:

| Column | Meaning |
|--------|---------|
| `Corr` | Point-biserial correlation with outcome. Higher absolute value = more predictive. |
| `WR(+)` | Win rate when signal is positive (bullish reading) |
| `WR(-)` | Win rate when signal is negative (bearish reading) |
| `Edge` | `WR(+) - WR(-)`. Positive = signal works as expected. Negative = inverted or noise. |

**If a signal shows negative edge:** set its weight to `0.0` in `./data/config.json`.

**If a new signal shows strong positive edge:** increase its weight and reduce weight elsewhere.

Weights update live on the next cycle — no restart needed.

---

## 18. Monitoring and Alerts

### Telegram Notifications

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.

Notifications are sent for:
- Every executed trade (side, size, price, market)
- Optimizer parameter changes (what changed, old vs new value)
- Consecutive loss cooldown triggered

### Webhook Notifications

Set `webhook_url` in `config.json` to receive trade notifications at any HTTP endpoint (e.g. Discord, Slack, custom service):

```json
{ "webhook_url": "https://hooks.slack.com/services/..." }
```

### Log Files

| File | Location | What It Contains |
|------|----------|-----------------|
| `collector.log` | `./logs/collector.log` | All stdout from the collector container |
| `scheduler.log` | `./logs/scheduler.log` | Scheduler-specific log |
| `optimizer.log` | `./data/optimizer.log` | Timestamped parameter change history |

---

## 19. Troubleshooting

### Container won't start

```bash
docker compose logs collector
```

Common causes:
- Missing `.env` file or missing required keys (`POLY_PRIVATE_KEY`, `POLY_API_KEY`)
- Port 5000 already in use (another service)
- `./data/` directory does not exist

### "Rate limited — will retry next cycle"

The Polymarket import endpoint is rate-limited. The system pre-warms the market ID cache at startup. If it persists:

```bash
docker exec pknwitq-collector python seed_cache.py
docker compose restart collector
```

### "KeyError" in logs

The config may have stale keys from an old version. Reset to defaults:

```bash
docker exec pknwitq-collector cp /app/config.json /data/config.json
docker compose restart collector
```

### Every cycle shows "momentum too weak"

BTC is in a flat period. Options:
1. Wait — this is normal during low-volatility hours
2. Lower `min_momentum_pct` in `./data/config.json` (the optimizer will find the right floor over time)
3. Check if blocked hours are covering too much of the day

### Every cycle shows "too early" or "too late"

The time window needs adjusting. The optimizer will tune this, but to fix immediately:

```bash
# Edit directly
notepad ./data/config.json
# Change min_time_remaining (lower to allow earlier entries)
# Change max_time_remaining (higher to allow entries right after open)
```

### Win rate below 55% after 50+ trades

1. Run signal analysis: `docker exec pknwitq-collector python signal_research.py --analyze --min-n 30`
2. Look for signals with negative `Edge` — zero them out in config
3. Check if specific hours are dragging down win rate — add them to `blocked_hours`
4. Let the optimizer run for several cycles — it adjusts threshold automatically

### Config changes not taking effect

The config is reloaded every cycle automatically. If changes aren't appearing after 20–30 seconds:

```bash
# Confirm the file was written
cat ./data/config.json | grep "your_key"

# Check container can read it
docker exec pknwitq-collector cat /data/config.json | grep "your_key"
```

---

## 20. File Structure Reference

```
polymarket-trader-1.0.12/
│
├── .env                          ← Your credentials and mode settings (never commit)
├── .env.poly.example             ← Template for .env
├── docker-compose.yml            ← Service orchestration
├── Dockerfile.trader             ← Collector container image
├── Dockerfile.monitor            ← Monitor container image
├── requirements.txt              ← Python deps for collector
├── requirements.monitor.txt      ← Python deps for monitor
│
├── trader/
│   ├── scheduler.py              ← Main loop: collect → trade → optimize (every 20s)
│   ├── fast_trader.py            ← Trade execution engine
│   ├── optimizer.py              ← Auto-parameter optimization (every 30 min)
│   ├── signal_research.py        ← Signal collection, DB writes, outcome resolution
│   ├── market_utils.py           ← Market discovery and slug resolution
│   ├── config.json               ← Image-default config (reset point only)
│   └── seed_cache.py             ← Pre-warm market ID cache utility
│
├── shared/
│   └── composite_signal.py       ← Multi-factor signal scoring engine
│
├── monitor/
│   └── monitor_api.py            ← Flask REST API + dashboard server
│
├── dashboard/
│   ├── index.html                ← Monitor page (signal health)
│   ├── dry-run.html              ← Signal Lab (composite score + predictions)
│   └── live.html                 ← Live Trading (P&L + trade history)
│
├── polymarket_sdk/               ← Polymarket API wrappers (CLOB, Gamma)
│
├── data/                         ← Persistent volume (bind-mounted to Docker)
│   ├── config.json               ← LIVE config (optimizer writes here)
│   ├── trade_log.json            ← All executed trades and outcomes
│   ├── signal_research.db        ← SQLite: signals, observations, outcomes
│   ├── optimizer.log             ← Timestamped parameter change history
│   ├── market_id_cache.json      ← Slug → condition_id cache
│   ├── daily_spend.json          ← Budget + API quota tracking
│   └── status.json               ← Current cycle status (dashboard feed)
│
└── logs/                         ← Bind-mounted log directory
    ├── collector.log             ← Full collector stdout+stderr
    └── scheduler.log             ← Scheduler log
```

---

## Quick Reference Card

```
START:          docker compose up -d
STOP:           docker compose down
LOGS:           docker compose logs -f collector
DASHBOARD:      http://localhost:5000/live.html

CHANGE MODE:    Edit .env → PKNWITQ_MODE=dry|both|collect
                docker compose up -d collector

CHANGE CONFIG:  Edit ./data/config.json  (takes effect next cycle, ~20s)

RESET CONFIG:   docker exec pknwitq-collector cp /app/config.json /data/config.json

REBUILD CODE:   docker compose up -d --build collector

ANALYZE SIGNALS: docker exec pknwitq-collector python signal_research.py --analyze --min-n 30

VIEW OPTIMIZER LOG: cat ./data/optimizer.log

FULL RESET:     docker compose down && rm -rf ./data && mkdir data && docker compose up -d
```
