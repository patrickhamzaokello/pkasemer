# FastLoop — Polymarket BTC 5-Minute Trading System

Automated signal collection and trading system for Polymarket's BTC Up/Down 5-minute fast markets. Fetches real-time Binance price data, computes a multi-factor composite signal score, and places trades when the signal exceeds a confidence threshold.

---

## How It Works

Every 20 seconds the system:

1. **Discovers** the current active BTC 5-minute market on Polymarket (e.g. `btc-updown-5m-1740441600`)
2. **Fetches** Binance BTCUSDT 1m candles + order book + recent trades
3. **Computes** a composite signal score from 6 weighted sub-signals (0 = strongly bearish, 0.5 = neutral, 1 = strongly bullish)
4. **Decides** — if score > 0.57 buy YES (BUY UP), if score < 0.43 buy NO (BUY DOWN), else hold
5. **Sizes** the position: `confidence × max_position` dollars, subject to fee EV check and daily budget
6. **Records** the observation, signal score, and prediction to the database
7. **Resolves** outcomes 10 minutes after each window closes and tracks accuracy

---

## Quick Start

```bash
# 1. Copy environment file
cp .env.example .env
# Add your SIMMER_API_KEY to .env

# 2. Create data/logs directories
mkdir -p data logs

# 3. Build and start (dry-run by default)
docker compose up -d

# 4. Open dashboard
# http://localhost:5000

# 5. Watch live logs
docker compose logs -f collector
```

---

## Operating Modes

Set `FASTLOOP_MODE` in `docker-compose.yml`:

| Mode      | What Runs                                    | Use When                              |
|-----------|----------------------------------------------|---------------------------------------|
| `collect` | Signal collection only, no trades            | Building initial dataset (Phase 1)    |
| `dry`     | Collection + dry-run trader (no real trades) | Validating signals before going live  |
| `both`    | Collection + live trading simultaneously     | Live trading (Phase 3)                |
| `trade`   | Live trading only (no collection)            | Rarely used                           |

> **Note:** `dry` mode runs both the collector and the trader. The collector writes observations and signal scores to the database. The trader logs what it would have done. This is the recommended validation phase before live trading.

---

## Three-Phase Workflow

```
Phase 1 — Data Collection (2–5 days)
  FASTLOOP_MODE=collect
  Goal: Build 200+ resolved observations
  Monitor: http://localhost:5000 (Monitor tab)
  Check: docker exec fastloop-collector python signal_research.py --analyze --min-n 30

Phase 2 — Dry Run Validation
  FASTLOOP_MODE=dry
  Goal: Verify predictions are accurate before spending real money
  Monitor: http://localhost:5000/dry-run.html (Signal Lab tab)
  Check: Prediction vs Outcome table — win rate should exceed 55%

Phase 3 — Live Trading
  FASTLOOP_MODE=both (collect + trade simultaneously)
  Ensure SIMMER_API_KEY is set in .env
  Monitor: http://localhost:5000/live.html (Live Trading tab)
```

---

## Dashboard Pages

### Monitor (`/` or `/index.html`)
Overview of signal collection health and raw signal data.

| Panel | What It Shows |
|-------|--------------|
| **Status strip** | Cycle count, DB total, resolved count, outcome distribution (up/down) |
| **Current Window** | Active market slug, seconds remaining, YES price, BTC vs reference price |
| **Signal Bars** | Live values for each signal (btc_vs_reference, momentum_5m, vol_ratio, RSI, etc.) |
| **Recent Windows** | Last 20 five-minute windows with avg signals and final outcome |

### Signal Lab (`/dry-run.html`)
Real-time composite score display with full signal breakdown.

| Panel | What It Shows |
|-------|--------------|
| **Live Composite Score** | Current score (0–1), gauge bar, BUY/SELL decision, per-signal breakdown |
| **Score History** | 100-cycle SVG chart with BUY/SELL threshold lines and trade markers |
| **Would-Trade Log** | Every cycle where score crossed the threshold — side, score, vs_ref, outcome |
| **Prediction vs Outcome** | All resolved predictions: was the call correct? Win rate, YES/NO accuracy |
| **Sidebar — Weights** | Active signal weights (amber bars) |
| **Sidebar — Session Stats** | Cycles scored, would-trade count, trade rate, avg score, top block reason |
| **Sidebar — Prediction Accuracy** | Overall win rate, YES accuracy, NO accuracy, avg score on correct vs wrong calls |

### Live Trading (`/live.html`)
P&L and trade history for live mode.

| Panel | What It Shows |
|-------|--------------|
| **P&L Strip** | Today P&L, Total P&L, Win Rate, Budget used, Open Positions, Avg per trade |
| **Cumulative P&L Chart** | SVG area chart of running P&L over all trades |
| **Trade History** | Every executed trade: side, amount, entry price, outcome, P&L |
| **Open Positions** | Trades with unresolved outcomes |

---

## Signals Explained

### Primary Signals (used in scoring)

| Signal | Weight | What It Measures | Interpretation |
|--------|--------|-----------------|----------------|
| `btc_vs_reference` | 40% | `(price_now - window_start_price) / window_start_price × 100` | Most predictive. Directly answers "is BTC up or down since this window started?" Positive = UP is winning. |
| `volume_ratio` | 15% | Current 1m volume ÷ average 1m volume | >1.5x surge confirms directional momentum. Low volume = weak signal. |
| `momentum_5m` | 15% | % price change over last 5 minutes on Binance | Direct BTC price trend. Positive = bullish. |
| `price_acceleration` | 15% | Rate of change of momentum (momentum of momentum) | Catches trend acceleration independent of the current momentum level. |
| `vol_adjusted_momentum` | 10% | `momentum_5m ÷ volatility_5m` | Momentum normalized for market turbulence — high vol periods produce noisier signals. |
| `momentum_consistency` | 5% | Fraction of last N candles agreeing on direction | 1.0 = all candles pointing same way, 0.0 = mixed/choppy. |

### Excluded Signals (weight = 0)

| Signal | Why Excluded |
|--------|-------------|
| `cex_poly_lag` | Redundant with `momentum_5m` when `poly_divergence ≈ 0` (common case) |
| `trade_flow_ratio` | Showed negative predictive edge (−0.38) in 362-obs empirical data |
| `order_imbalance` | Near-zero edge (+0.01) — no predictive value found |

---

## Key Metrics to Watch

### Signal Health
- **`btc_vs_reference`** — should be non-zero once the window has been open for >20s. If always `n/a`, the reference price cache (`window_refs.json`) may be missing or the window hasn't opened yet.
- **`momentum_5m`** — typical range ±0.05% to ±0.3% during active markets. Consistently near zero = flat market, expect many "momentum too weak" blocks.
- **`volume_ratio`** — >1.3x is notable, >2x is strong. Flat 1.0 = no volume spike.

### Score Thresholds
- **Score > 0.57** → BUY YES (signal says BTC will be UP at window close)
- **Score < 0.43** → BUY NO (signal says BTC will be DOWN at window close)
- **0.43–0.57** → Neutral band, no trade
- **`confidence`** — `|score - 0.5| × 2` — scales from 0 (completely neutral) to 1 (maximum conviction). Position size = `confidence × max_position`

### Filter Reasons (why trades are blocked)
| Reason | Meaning | Action |
|--------|---------|--------|
| `momentum too weak` | `|momentum_5m| < 0.05%` | Normal during quiet markets. No action needed. |
| `poly_divergence too large` | Polymarket has already priced in the move | Edge has been captured. Skip. |
| `score within neutral band` | Score between 0.43–0.57 | Signal is ambiguous. |
| `fee EV negative` | Win rate implied by score < fee breakeven | Market priced efficiently, no edge. |
| `daily budget exhausted` | Spent `daily_budget` USD today | Resets at midnight UTC. |
| `position too small` | Sized position < $0.50 | Increase `max_position` in config.json |

### Prediction Accuracy (Signal Lab → Prediction vs Outcome)
- **Win rate > 55%** — good, system has edge above the 52.1% fee breakeven
- **Win rate 50–55%** — marginal, may not beat fees long-term
- **Win rate < 50%** — signal is degraded; re-run `--analyze` and recalibrate weights
- **YES accuracy vs NO accuracy** — check if system is better at calling one direction than the other. Imbalance may suggest threshold asymmetry.
- **Avg score (correct) vs Avg score (wrong)** — correct calls should have higher scores. If wrong calls have similar/higher scores, signal is noisy at the margin.

### P&L Health (Live Trading)
- **Win rate** should exceed 52.1% to beat 10% fees at typical entry prices (~$0.49)
- **Avg per trade** should be positive after fees
- **Daily budget** tracks total USD spent today vs configured `daily_budget`

---

## Configuration (`config.json`)

```json
{
  "entry_threshold":    0.05,    // Minimum |score - 0.5| to consider a trade
  "min_momentum_pct":   0.05,    // Minimum |momentum_5m| % to proceed
  "max_position":       5.00,    // Max USD per trade (before confidence scaling)
  "lookback_minutes":   5,       // Binance candle lookback window
  "min_time_remaining": 30,      // Don't enter a market with < 30s left
  "max_time_remaining": 900,     // Ignore markets > 15min away
  "daily_budget":       5.00,    // Total USD to spend per day across all trades
  "composite_threshold": 0.57,   // Score threshold for BUY YES / BUY NO
  "signal_weights": {
    "btc_vs_reference":      0.40,
    "volume_ratio":          0.15,
    "momentum_5m":           0.15,
    "price_acceleration":    0.15,
    "vol_adjusted_momentum": 0.10,
    "momentum_consistency":  0.05,
    "cex_poly_lag":          0.00,
    "trade_flow_ratio":      0.00,
    "order_imbalance":       0.00
  }
}
```

> **Important:** `signal_weights` in `config.json` completely overrides the defaults in `composite_signal.py`. Always update both together, or only edit `config.json`.

After any config change:
```bash
docker compose restart collector
```

---

## Signal Recalibration

After collecting 100+ resolved observations, run the correlation analysis to check if weights need updating:

```bash
docker exec fastloop-collector python signal_research.py --analyze --min-n 30
```

Columns to focus on:
- **`Corr`** — point-biserial correlation with outcome (up=1, down=0). Higher absolute value = more predictive.
- **`WR(+)`** — win rate when signal is positive (bullish)
- **`WR(−)`** — win rate when signal is negative (bearish)
- **`Edge`** — `WR(+) - WR(−)`. Positive = signal works in expected direction. Negative = signal is inverted or noise.

If a signal shows negative edge, set its weight to 0.00 in `config.json`. Redistribute weight to signals with the highest positive edge.

---

## Useful Commands

```bash
# View live logs
docker compose logs -f collector
docker compose logs -f monitor

# Run signal correlation analysis
docker exec fastloop-collector python signal_research.py --analyze --min-n 30

# Manually trigger outcome resolution
docker exec fastloop-collector python signal_research.py --resolve

# Export all observations to CSV
docker exec fastloop-collector python signal_research.py --export /data/signals.csv
# File appears at ./data/signals.csv on host

# Quick DB stats
docker exec fastloop-collector python -c "
import sqlite3
c = sqlite3.connect('/data/signal_research.db')
total    = c.execute('SELECT COUNT(*) FROM signal_observations').fetchone()[0]
resolved = c.execute('SELECT COUNT(*) FROM signal_observations WHERE resolved=1').fetchone()[0]
preds    = c.execute('SELECT COUNT(*) FROM signal_observations WHERE would_trade=1').fetchone()[0]
correct  = c.execute('''SELECT COUNT(*) FROM signal_observations
    WHERE would_trade=1 AND resolved=1 AND (
        (signal_side=\"yes\" AND outcome=\"up\") OR
        (signal_side=\"no\"  AND outcome=\"down\"))''').fetchone()[0]
print(f'Total: {total}  Resolved: {resolved}  Predictions: {preds}  Correct: {correct}')
if preds > 0: print(f'Win rate (unresolved included): {correct/preds:.1%}')
"

# Restart collector (e.g. after changing config.json)
docker compose restart collector

# Full rebuild after code changes
docker compose down
docker compose build --no-cache
docker compose up -d

# Full reset (wipe all data and start fresh)
docker compose down && rm -rf ./data && mkdir data && docker compose up -d
```

---

## Log Format

Each 20-second cycle produces a single line:

```
[DRY] 14:32:05 | m-1740441600 287s | m5=-0.025% vs_ref=-0.061% poly=0.485 vol=1.12x | score=0.282 BLOCK: momentum too weak
[DRY] 14:35:08 | m-1740441600 104s | m5=-0.085% vs_ref=-0.195% poly=0.482 vol=1.44x | score=0.208 conf=0.58 → NO $2.90
  [DRY RUN] Would buy NO $2.90 (~5.9 shares @ $0.490)
```

| Field | Meaning |
|-------|---------|
| `[DRY]` / `[LIVE]` | Operating mode |
| `m-1740441600` | Last 10 chars of market slug (Unix timestamp) |
| `287s` | Seconds remaining in this window |
| `m5=` | 5m Binance momentum % |
| `vs_ref=` | BTC % change since window opened |
| `poly=` | Current Polymarket YES price |
| `vol=` | Volume ratio (current ÷ avg) |
| `score=` | Composite signal score |
| `BLOCK:` | Reason trade was skipped |
| `conf=` | Confidence (position sizing multiplier) |
| `→ YES/NO $X.XX` | Trade direction and position size |

---

## File Structure

```
polymarket-fast-loop-1.0.12/
├── fast_trader.py          Main trading logic and market discovery
├── signal_research.py      Signal collection, DB write, outcome resolution
├── composite_signal.py     Weighted signal scoring engine
├── scheduler.py            24/7 run loop (collect + trade)
├── monitor_api.py          Flask API serving all dashboard endpoints
├── config.json             Runtime configuration (overrides defaults)
├── docker-compose.yml      Container orchestration
├── Dockerfile
├── requirements.txt
├── .env.example            Copy to .env, add SIMMER_API_KEY
├── dashboard/
│   ├── index.html          Monitor page (signal collection status)
│   ├── dry-run.html        Signal Lab (composite score + predictions)
│   └── live.html           Live Trading (P&L + trade history)
├── data/                   Auto-created; holds DB + window_refs.json
└── logs/                   Auto-created; holds scheduler.log
```

---

## Architecture Notes

- **`window_refs.json`** — Cached Binance price at the first observation of each 5-minute window. Used to compute `btc_vs_reference`. Polymarket's API never returns `priceToBeat` so we maintain our own. Stored in `/data/` (Docker) or `./` (local).
- **Signal scoring** is done twice per cycle: once in `collect_one` (stored to DB) and once in `run_fast_market_strategy` (for the trading decision). Both use fresh Binance data fetched at that moment.
- **Outcome resolution** runs every 10 minutes — `resolve_outcomes()` queries Polymarket's Gamma API for closed markets and fills in `outcome = 'up'/'down'` on matching DB rows.
- **`config.json` takes priority** over all code defaults. After any weight change in `config.json`, restart the collector — no rebuild required.
