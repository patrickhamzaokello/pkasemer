# FastLoop — Docker Deployment

## Directory structure

```
polymarket-fast-loop-1.0.12/
├── fast_trader.py          ← original (unchanged)
├── signal_research.py      ← updated (ghost market fix, resolve fix)
├── composite_signal.py     ← composite signal engine
├── scheduler.py            ← market-hours enforcer + run loop
├── monitor_api.py          ← Flask API serving dashboard data
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .env                    ← create this (copy .env.example)
├── config.json             ← updated config
├── dashboard/
│   └── index.html          ← monitoring dashboard
├── data/                   ← auto-created, holds signal_research.db
└── logs/                   ← auto-created, holds scheduler.log
```

## Quick start

```bash
# 1. Create your .env file
cp .env.example .env
# Edit .env and add your SIMMER_API_KEY if trading

# 2. Create data/logs directories
mkdir -p data logs

# 3. Build and start
docker compose up -d

# 4. Open dashboard
# http://localhost:5000

# 5. Watch logs
docker compose logs -f collector
```

## Modes

Set `FASTLOOP_MODE` in `docker-compose.yml`:

| Mode      | What it does                                      |
|-----------|---------------------------------------------------|
| `collect` | Collect signal data only, no trades (default)     |
| `dry`     | Dry-run fast_trader (shows signals, no trades)    |
| `trade`   | Live trading only                                 |
| `both`    | Collect signals AND trade simultaneously          |

## Market hours

The scheduler **only runs cycles Mon–Fri 14:30–21:00 UTC** (9:30am–4pm ET).
Outside these hours it sleeps and logs the time until next open.

## Workflow

```
Phase 1 — Research (run for 2–5 trading days):
  FASTLOOP_MODE=collect
  docker compose up -d

  Monitor at http://localhost:5000
  Check correlations: docker exec fastloop-collector python signal_research.py --analyze --min-n 30

Phase 2 — Validate (dry-run trading):
  FASTLOOP_MODE=dry
  docker compose up -d collector

Phase 3 — Live trading:
  FASTLOOP_MODE=both
  Ensure SIMMER_API_KEY is set in .env
  docker compose up -d collector
```

## Useful commands

```bash
# Run signal analysis
docker exec fastloop-collector python signal_research.py --analyze --min-n 30

# Export data to CSV
docker exec fastloop-collector python signal_research.py --export /data/signals.csv
# File will be at ./data/signals.csv on host

# Manually resolve outcomes
docker exec fastloop-collector python signal_research.py --resolve

# View DB stats
docker exec fastloop-collector python -c "
import sqlite3
conn = sqlite3.connect('/data/signal_research.db')
total = conn.execute('SELECT COUNT(*) FROM signal_observations').fetchone()[0]
resolved = conn.execute('SELECT COUNT(*) FROM signal_observations WHERE resolved=1').fetchone()[0]
outcomes = conn.execute('SELECT outcome, COUNT(*) FROM signal_observations WHERE resolved=1 GROUP BY outcome').fetchall()
print(f'Total: {total}  Resolved: {resolved}  Outcomes: {outcomes}')
conn.close()
"

# Restart just the collector (e.g. after config change)
docker compose restart collector

# Stop everything
docker compose down

# Full reset (delete all data)
docker compose down && rm -rf ./data && mkdir data
```

## Updating signal weights

After running analysis, update `config.json`:

```json
{
  "signal_weights": {
    "order_imbalance":      0.30,
    "trade_flow_ratio":     0.25,
    "momentum_5m":          0.20,
    "cex_poly_lag":         0.15,
    "momentum_consistency": 0.05,
    "vol_adjusted_momentum": 0.05
  }
}
```

Then restart: `docker compose restart collector`