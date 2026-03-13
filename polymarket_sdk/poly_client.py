#!/usr/bin/env python3
"""
poly_client.py — Unified PolyClient (drop-in SimmerClient replacement)

Provides a single PolyClient class with the same method signatures as
SimmerClient, so fast_trader.py can be updated with minimal code changes.

SimmerClient method → PolyClient equivalent:
  .import_market(url)              → .import_market(url)    [returns {market_id: condition_id}]
  .trade(market_id, side, amount)  → .trade(market_id, side, amount)
  .get_positions()                 → .get_positions()
  .get_portfolio()                 → .get_portfolio()
  .redeem(market_id, side)         → .redeem(market_id, side)

Usage (in fast_trader.py):
    # Before:
    from simmer_sdk import SimmerClient
    _client = SimmerClient(api_key=api_key, venue="polymarket")

    # After:
    from polymarket_sdk import PolyClient
    _client = PolyClient()

The PolyClient singleton is managed by get_poly_client() below, mirroring
fast_trader.py's existing get_client() pattern.
"""

from __future__ import annotations

if __package__ is None:
    import sys as _sys; from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent)); __package__ = "polymarket_sdk"

import os
import re
from dataclasses import dataclass, field
from typing import Any

from .poly_auth    import get_clob_client, get_wallet_address
from .poly_market  import resolve_market, get_token_id
from .poly_trade   import execute_trade, sell_shares as _sell_shares
from .poly_positions import get_positions as _get_positions
from .poly_portfolio import get_portfolio as _get_portfolio
from .poly_history   import build_trade_index
from .poly_redeem    import redeem_position


@dataclass
class TradeResult:
    """Mirrors the SimmerClient trade result object shape."""
    success:       bool  = False
    trade_id:      str   = ""
    shares_bought: float = 0.0
    error:         str   = ""


@dataclass
class Position:
    """Mirrors the SimmerClient position dataclass shape."""
    market_id:     str   = ""
    question:      str   = ""
    side:          str   = ""
    shares_yes:    float = 0.0
    shares_no:     float = 0.0
    entry_price:   float = 0.0
    current_price: float = 0.0
    redeemable:    bool  = False
    pnl:           float = 0.0


class PolyClient:
    """
    Direct Polymarket CLOB client with same API surface as SimmerClient.

    All methods match the signatures and return shapes that fast_trader.py
    expects from SimmerClient.
    """

    def __init__(self):
        # Verify credentials are available at construction time
        _ = get_clob_client()
        self._address = get_wallet_address()

    # ------------------------------------------------------------------
    # market import (replaces SimmerClient.import_market)
    # ------------------------------------------------------------------

    def import_market(self, url: str) -> Any:
        """
        Resolve a Polymarket URL or slug to a market entry.

        Args:
            url: Either a full Polymarket URL or a bare slug string.
                 e.g. "https://polymarket.com/event/btc-updown-5m-1748000000"
                 or   "btc-updown-5m-1748000000"

        Returns:
            Object with .market_id = condition_id (str), or dict on failure.
        """
        slug = _extract_slug(url)
        condition_id, error = resolve_market(slug)

        if error:
            return {"error": error, "market_id": None}

        # Return an object with .market_id for SimmerClient compatibility
        return _MarketImportResult(market_id=condition_id, slug=slug)

    # ------------------------------------------------------------------
    # trade execution (replaces SimmerClient.trade)
    # ------------------------------------------------------------------

    def trade(
        self,
        market_id: str,
        side: str,
        amount: float,
        source: str = "",
    ) -> TradeResult:
        """
        Execute a market-buy order.

        Args:
            market_id: condition_id (returned by import_market)
            side:      "yes" or "no"
            amount:    USDC amount to spend
            source:    Ignored (kept for API compatibility)

        Returns:
            TradeResult with .success, .trade_id, .shares_bought, .error
        """
        result = execute_trade(market_id, side, amount)
        return TradeResult(
            success=result.get("success", False),
            trade_id=result.get("trade_id") or "",
            shares_bought=result.get("shares_bought", 0.0),
            error=result.get("error") or "",
        )

    # ------------------------------------------------------------------
    # positions (replaces SimmerClient.get_positions)
    # ------------------------------------------------------------------

    def get_positions(self) -> list[Position]:
        """
        Fetch all open positions.

        Returns:
            List of Position dataclass instances.
        """
        raw_positions = _get_positions(self._address)
        positions = []
        for p in raw_positions:
            positions.append(Position(
                market_id=p.get("market_id", ""),
                question=p.get("question", ""),
                side=p.get("side", ""),
                shares_yes=p.get("shares_yes", 0.0),
                shares_no=p.get("shares_no", 0.0),
                entry_price=p.get("entry_price", 0.0),
                current_price=p.get("current_price", 0.0),
                redeemable=p.get("redeemable", False),
                pnl=p.get("pnl", 0.0),
            ))
        return positions

    # ------------------------------------------------------------------
    # portfolio (replaces SimmerClient.get_portfolio)
    # ------------------------------------------------------------------

    def get_portfolio(self) -> dict:
        """
        Fetch portfolio balance and exposure.

        Returns:
            {
              "balance_usdc":    float,
              "total_exposure":  float,
              "positions_count": int,
            }
        """
        return _get_portfolio(self._address)

    # ------------------------------------------------------------------
    # sell (early exit — sell existing shares before market resolves)
    # ------------------------------------------------------------------

    def sell(self, market_id: str, side: str, shares: float) -> dict:
        """
        Sell existing conditional token shares via a SELL market order.

        Args:
            market_id: condition_id of the open market
            side:      "yes" or "no" — the side you hold
            shares:    number of tokens to sell

        Returns:
            {"success": bool, "trade_id": str|None, "error": str|None}
        """
        return _sell_shares(market_id, side, shares)

    # ------------------------------------------------------------------
    # redeem (replaces SimmerClient.redeem)
    # ------------------------------------------------------------------

    def redeem(self, market_id: str, side: str) -> dict:
        """
        Redeem a winning position by submitting an on-chain transaction.

        Args:
            market_id: condition_id of the resolved market
            side:      "yes" or "no" (which tokens you hold)

        Returns:
            {"success": bool, "tx_hash": str | None, "error": str | None}
        """
        return redeem_position(market_id, side)

    # ------------------------------------------------------------------
    # trade history (used by backfill_trade_outcomes)
    # ------------------------------------------------------------------

    def get_trade_history_index(self, limit: int = 200) -> dict:
        """
        Return a dict of {order_id: trade_record} for P&L backfill.
        """
        return build_trade_index(self._address, limit)


@dataclass
class _MarketImportResult:
    """Returned by import_market() — mimics the SimmerClient market object."""
    market_id: str
    slug:      str = ""

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


def _extract_slug(url_or_slug: str) -> str:
    """
    Extract slug from a Polymarket URL or return the input directly if it's
    already a bare slug.

    Examples:
      "https://polymarket.com/event/btc-updown-5m-1748000000" → "btc-updown-5m-1748000000"
      "btc-updown-5m-1748000000"                              → "btc-updown-5m-1748000000"
    """
    url_or_slug = url_or_slug.strip()

    if "polymarket.com" in url_or_slug:
        # Extract the path segment after /event/
        match = re.search(r"/event/([^/?#]+)", url_or_slug)
        if match:
            return match.group(1)
        # Fallback: last path segment
        return url_or_slug.rstrip("/").split("/")[-1]

    return url_or_slug


# ---------------------------------------------------------------------------
# Singleton pattern (mirrors fast_trader.py's get_client())
# ---------------------------------------------------------------------------

_poly_client: PolyClient | None = None


def get_poly_client() -> PolyClient:
    """
    Return singleton PolyClient. Creates on first call.

    Drop-in replacement for fast_trader.py's get_client() function.
    Replace:
        _client = SimmerClient(...)
    With:
        from polymarket_sdk import get_poly_client
        _client = get_poly_client()
    """
    global _poly_client
    if _poly_client is None:
        _poly_client = PolyClient()
    return _poly_client


# ---------------------------------------------------------------------------
# CLI: python poly_client.py [portfolio|positions|history]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    cmd = sys.argv[1] if len(sys.argv) > 1 else "portfolio"
    client = get_poly_client()
    print(f"Wallet: {client._address}")

    if cmd == "portfolio":
        p = client.get_portfolio()
        print(f"Balance:  ${p['balance_usdc']:.2f}")
        print(f"Exposure: ${p['total_exposure']:.2f}")
        print(f"Positions: {p['positions_count']}")

    elif cmd == "positions":
        positions = client.get_positions()
        if not positions:
            print("No open positions.")
        for pos in positions:
            from dataclasses import asdict
            d = asdict(pos)
            print(f"  {d['question'][:50]:50}  YES={d['shares_yes']:.1f}  NO={d['shares_no']:.1f}  P&L=${d['pnl']:.2f}")

    elif cmd == "history":
        idx = client.get_trade_history_index(limit=10)
        if not idx:
            print("No trade history.")
        for oid, t in list(idx.items())[:10]:
            print(f"  {oid[:12]}...  {t['side'].upper():3}  {t['shares']:.2f} shares  paid=${t['amount_paid']:.2f}")

    else:
        print(f"Unknown command: {cmd}. Use: portfolio | positions | history")
