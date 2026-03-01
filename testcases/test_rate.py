#!/usr/bin/env python3
"""
Test Simmer agent info endpoint.
Usage: python test_agent.py
"""

import os
import json
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from dotenv import load_dotenv

load_dotenv()

def get_agent_info():
    api_key = os.environ.get("SIMMER_API_KEY")
    if not api_key:
        print("ERROR: SIMMER_API_KEY not set in .env")
        return

    req = Request(
        "https://api.simmer.markets/api/sdk/agents/me",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "simmer-pknwitq/1.0",
        }
    )

    try:
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        return
    except URLError as e:
        print(f"Connection error: {e.reason}")
        return

    # ── Identity ──────────────────────────────────────────────────────────────
    print("=" * 50)
    print("  AGENT INFO")
    print("=" * 50)
    print(f"  Name:          {data.get('name')}")
    print(f"  Agent ID:      {data.get('agent_id')}")
    print(f"  Status:        {data.get('status')}")
    print(f"  Claimed:       {data.get('claimed')}")
    print(f"  Real trading:  {data.get('real_trading_enabled')}")
    print(f"  Auto redeem:   {data.get('auto_redeem_enabled')}")
    print(f"  Created:       {data.get('created_at')}")
    print(f"  Last trade:    {data.get('last_trade_at')}")

    # ── Performance ───────────────────────────────────────────────────────────
    print()
    print("  PERFORMANCE")
    print("-" * 50)
    print(f"  Sim balance:   ${data.get('balance', 0):,.2f}")
    print(f"  Sim PnL:       ${data.get('sim_pnl', 0):+,.2f}")
    print(f"  Total PnL:     ${data.get('total_pnl', 0):+,.2f}  ({data.get('total_pnl_percent', 0):+.2f}%)")
    print(f"  Poly PnL:      ${data.get('polymarket_pnl') or 0:+,.2f}")
    print()
    print(f"  Trades:        {data.get('trades_count', 0)}")
    print(f"  Wins:          {data.get('win_count', 0)}")
    print(f"  Losses:        {data.get('loss_count', 0)}")
    win_rate = data.get('win_rate')
    print(f"  Win rate:      {f'{win_rate:.1%}' if win_rate else 'n/a'}")

    # ── Rate limits ───────────────────────────────────────────────────────────
    print()
    print("  RATE LIMITS")
    print("-" * 50)
    rate_limits = data.get("rate_limits", {})
    print(f"  Tier:          {rate_limits.get('tier', 'unknown')}")
    print(f"  Window:        {rate_limits.get('window_seconds', 60)}s")
    print(f"  Default:       {rate_limits.get('default_requests_per_minute')} req/min")
    print()
    endpoints = rate_limits.get("endpoints", {})
    key_endpoints = [
        "/api/sdk/markets/import",
        "/api/sdk/trade",
        "/api/sdk/markets",
        "/api/sdk/portfolio",
        "/api/sdk/positions",
    ]
    for ep in key_endpoints:
        if ep in endpoints:
            rpm = endpoints[ep].get("requests_per_minute", "?")
            print(f"  {ep:<35} {rpm} req/min")

    print("=" * 50)

    # ── Warnings ──────────────────────────────────────────────────────────────
    warnings = []
    if not data.get("claimed"):
        warnings.append("⚠ Agent not claimed — go to claim_url to activate")
    if not data.get("real_trading_enabled"):
        warnings.append("⚠ Real trading disabled — trades will be simulated only")
    if data.get("polymarket_pnl") is None:
        warnings.append("⚠ No wallet linked — polymarket_pnl is null")

    if warnings:
        print()
        for w in warnings:
            print(f"  {w}")
        print()

if __name__ == "__main__":
    get_agent_info()