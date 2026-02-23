# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polymarket FastLoop is a real-time trading system for Polymarket's 5-minute crypto fast markets. It uses CEX (Binance) momentum signals to predict outcomes, scores them via a composite signal engine, and executes trades through the Simmer SDK.

## Running the System

**One-shot trade/dry-run:**
```bash
python fast_trader.py              # Dry-run (default, no real trades)
python fast_trader.py --live       # Live trading
python fast_trader.py --positions  # Show open positions
python fast_trader.py --config     # Show current config
python fast_trader.py --set KEY=VALUE  # Update config (e.g. --set asset=ETH)
```

**Signal research and analysis:**
```bash
python signal_research.py --collect     # Collect signals and log to DB
python signal_research.py --analyze     # Print correlation report
python signal_research.py --analyze --min-n 30  # With minimum observation count
python signal_research.py --live        # Trade using composite signal
python signal_research.py --export FILE # Export to CSV
python signal_research.py --resolve     # Resolve outstanding outcomes
```

**Continuous scheduler (Docker):**
```bash
docker compose up -d                    # Start collector + monitor services
docker compose logs -f collector        # Watch scheduler output
docker compose restart collector        # Restart after config changes
docker compose down && rm -rf ./data && mkdir data  # Full reset
```

**Docker exec shortcuts:**
```bash
docker exec fastloop-collector python signal_research.py --analyze --min-n 30
docker exec fastloop-collector python signal_research.py --export /data/signals.csv
docker exec fastloop-collector sqlite3 /data/signal_research.db "SELECT COUNT(*) FROM signal_observations"
```

**Monitor API (Flask on port 5000):**
```bash
python monitor_api.py
# Endpoints: /api/status, /api/observations, /api/correlations
```

## Architecture

### Data Flow

```
Binance (OHLCV klines) ──→ CEX signals ─┐
                                          ├──→ composite_signal.py ──→ Trade via Simmer SDK
Polymarket (CLOB API) ──→ Poly signals ──┘
                                          └──→ signal_research.db (SQLite)
```

### Core Modules

- **`fast_trader.py`** — Main execution entry point. Discovers active fast markets, collects signals, computes composite score, executes trades, enforces daily budget.
- **`composite_signal.py`** — Signal scoring engine. Applies hard pre-trade filters (momentum, volatility, RSI, spread), normalizes 6 signals to (0,1), computes weighted average score. Score >0.60 → BUY YES, <0.40 → BUY NO.
- **`signal_research.py`** — Multi-factor signal logger. Collects 9+ dimensions (momentum, RSI, volume ratio, order imbalance, trade flow) and logs against outcomes to SQLite. Supports correlation analysis to refine signal weights.
- **`scheduler.py`** — Market-hours enforcer. Runs only Mon-Fri 14:30–21:00 UTC. Orchestrates collection/trading cycles and writes health status to `/data/status.json`.
- **`monitor_api.py`** — Flask REST API serving the dashboard at `dashboard/index.html`.

### Configuration (`config.json`)

Key parameters:
- `composite_threshold` (default 0.60): Score cutoff for trade entry
- `signal_weights`: Dict of 6 signal weights summing to 1.0
- `daily_budget`: Total USD allocated per day
- `max_position`: Max USD per individual trade
- `asset`: BTC | ETH | SOL | XRP
- `window`: `5m` | `15m` (fast market duration)
- `entry_threshold`: Min price divergence from $0.50

### Signal Weights (in `config.json`)

```json
{
  "order_imbalance": 0.25,
  "trade_flow_ratio": 0.20,
  "momentum_5m": 0.20,
  "cex_poly_lag": 0.15,
  "momentum_consistency": 0.10,
  "vol_adjusted_momentum": 0.10
}
```

Tune these weights based on `signal_research.py --analyze` correlation output after collecting ≥30 resolved observations.

### Docker Services

Two services defined in `docker-compose.yml`:
- `collector`: runs `scheduler.py`, mounts `./data` to `/data`
- `monitor`: runs `monitor_api.py`, exposes port 5000

Both share the `fastloop-data` volume. The collector writes `/data/status.json` which the monitor reads and serves.

## Environment Setup

Requires a `.env` file with:
```
SIMMER_API_KEY=sk_live_...
WALLET_PRIVATE_KEY=0x...
```

The Simmer API key is obtained from simmer.markets/dashboard. The wallet private key signs Polygon USDC.e transactions.

## Recommended Workflow (from Readme.md)

1. **Phase 1** — `FASTLOOP_MODE=collect` for 2–5 days to build signal dataset
2. **Phase 2** — `FASTLOOP_MODE=dry` to validate signal → outcome correlations
3. **Phase 3** — `FASTLOOP_MODE=both` for live trading with simultaneous collection

## Key External Dependencies

- **Simmer SDK** (`simmer-sdk==0.8.27`): Polymarket market discovery, import, and trade execution
- **Binance API**: CEX OHLCV klines and order book data (no auth required for public endpoints)
- **Polymarket CLOB API**: Real-time odds and order book
- **SQLite** (`signal_research.db`): Stores signal observations and resolved outcomes
