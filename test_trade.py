#!/usr/bin/env python3
"""
test_trade.py — Test $2 trade on the current BTC 5m up/down market.

Usage:
  python test_trade.py              # dry run (no real trade, just shows market + price)
  python test_trade.py --live       # executes real $2 YES trade
  python test_trade.py --live --no  # executes real $2 NO trade
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Make both trader/ and project root importable
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "trader"))
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from market_utils import discover_fast_market_markets, find_best_fast_market
from polymarket_sdk.poly_market import resolve_market
from polymarket_sdk.poly_trade  import execute_trade

# ── Config ────────────────────────────────────────────────────────────────────

AMOUNT  = 2.0
SIDE    = "no" if "--no" in sys.argv else "yes"
DRY_RUN = "--live" not in sys.argv

# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"TEST TRADE  |  BTC 5m up/down  |  {'DRY RUN' if DRY_RUN else 'LIVE'}")
    print("=" * 60)

    # ── Step 1: Discover live market ──────────────────────────────────────────
    print("\n[1] Discovering BTC 5m markets via Gamma API...")
    markets = discover_fast_market_markets("BTC", "5m")

    if not markets:
        print("  ERROR: No active BTC 5m markets found right now.")
        print("  Markets may be between windows. Try again in a few seconds.")
        sys.exit(1)

    best = find_best_fast_market(markets, min_time_remaining=30)
    if not best:
        print("  ERROR: No market has enough time remaining (< 30s).")
        sys.exit(1)

    now       = datetime.now(timezone.utc)
    remaining = (best["end_time"] - now).total_seconds()

    try:
        prices    = json.loads(best.get("outcome_prices", "[0.5,0.5]"))
        yes_price = float(prices[0]) if prices else 0.5
    except (json.JSONDecodeError, IndexError, ValueError):
        yes_price = 0.5

    no_price    = 1.0 - yes_price
    entry_price = yes_price if SIDE == "yes" else no_price
    est_shares  = AMOUNT / entry_price if entry_price > 0 else 0

    print(f"  Question : {best['question']}")
    print(f"  Slug     : {best['slug']}")
    print(f"  Ends in  : {remaining:.0f}s  ({best['end_time'].strftime('%H:%M:%S UTC')})")
    print(f"  YES      : {yes_price:.4f}   NO: {no_price:.4f}")
    print(f"  Bid/Ask  : {best.get('bestBid', 'n/a')} / {best.get('bestAsk', 'n/a')}")

    # ── Step 2: Resolve slug → condition_id + token_ids ───────────────────────
    print(f"\n[2] Resolving slug → condition_id...")
    cid, err = resolve_market(best["slug"])
    if err:
        print(f"  ERROR: {err}")
        sys.exit(1)

    print(f"  condition_id : {cid}")

    # ── Step 3: Trade ─────────────────────────────────────────────────────────
    print(f"\n[3] Order: {SIDE.upper()} ${AMOUNT:.2f}  (~{est_shares:.2f} shares @ ${entry_price:.4f})")

    if DRY_RUN:
        print("\n  [DRY RUN] No order placed.")
        print("  Run with --live to execute for real.")
        return

    print("  Submitting FOK market order...", flush=True)
    result = execute_trade(cid, SIDE, AMOUNT)

    print()
    if result.get("success"):
        shares = result.get("shares_bought", 0)
        price  = result.get("average_price", 0)
        tid    = result.get("trade_id") or "n/a"
        print(f"  ✓ FILLED")
        print(f"    shares    : {shares:.4f}")
        print(f"    avg price : {price:.4f}")
        print(f"    trade_id  : {tid}")
    else:
        error = result.get("error", "unknown error")
        print(f"  ✗ FAILED: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
