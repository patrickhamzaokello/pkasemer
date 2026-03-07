#!/usr/bin/env python3
"""
poly_trade.py — Component 3: Order Execution

Replaces: client.trade(market_id, side, amount, source=...)

Executes Fill-Or-Kill (FOK) market orders directly on Polymarket's CLOB.
FOK = the market-order equivalent: fills immediately at best price or cancels.

Depends on: poly_auth.get_clob_client(), poly_market.get_token_id()

Flow:
  1. Look up yes_token_id / no_token_id for the given condition_id
  2. Build a signed MarketOrderArgs (USDC amount — CLOB prices it automatically)
  3. POST with OrderType.FOK
  4. Return normalised result matching fast_trader.py's expected shape
"""

from __future__ import annotations

if __package__ is None:
    import sys as _sys; from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent)); __package__ = "polymarket_sdk"

from typing import Any

from py_clob_client.clob_types import MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

from .poly_auth import get_clob_client
from .poly_market import get_token_id


def execute_trade(
    market_id: str,
    side: str,
    amount_usdc: float,
) -> dict:
    """
    Execute a market-buy order on Polymarket CLOB.

    Uses MarketOrderArgs where `amount` is the USDC amount to spend.
    The CLOB auto-calculates the best available price and fills shares accordingly.

    Args:
        market_id:   condition_id (returned by resolve_market)
        side:        "yes" or "no"
        amount_usdc: USDC amount to spend (e.g. 5.0 = $5)

    Returns:
        {
            "success":       bool,
            "trade_id":      str | None,
            "shares_bought": float,
            "shares":        float,        # alias for shares_bought
            "average_price": float,
            "error":         str | None,
        }
    """
    client = get_clob_client()

    token_id = get_token_id(market_id, side)
    if not token_id:
        return {
            "success": False,
            "trade_id": None,
            "shares_bought": 0.0,
            "shares": 0.0,
            "average_price": 0.0,
            "error": (
                f"No token_id found for market_id={market_id} side={side}. "
                "Call resolve_market(slug) first to populate the cache."
            ),
        }

    # MarketOrderArgs: amount is USDC, price auto-calculated by CLOB
    order_args = MarketOrderArgs(
        token_id=token_id,
        amount=amount_usdc,
        side=BUY,
    )

    try:
        signed_order = client.create_market_order(order_args)
    except Exception as e:
        return {
            "success": False,
            "trade_id": None,
            "shares_bought": 0.0,
            "shares": 0.0,
            "average_price": 0.0,
            "error": f"Order signing failed: {e}",
        }

    try:
        resp = client.post_order(signed_order, OrderType.FOK)
    except Exception as e:
        return {
            "success": False,
            "trade_id": None,
            "shares_bought": 0.0,
            "shares": 0.0,
            "average_price": 0.0,
            "error": f"Order post failed: {e}",
        }

    # Get mid price for average_price in response (informational only)
    entry_price = _get_mid_price(client, token_id) or 0.0
    return _parse_order_response(resp, entry_price)


def _get_mid_price(client: Any, token_id: str) -> float | None:
    """Fetch current mid price from the CLOB for informational use."""
    try:
        mid = client.get_midpoint(token_id)
        if mid:
            val = mid.get("mid") if isinstance(mid, dict) else getattr(mid, "mid", None)
            if val:
                return round(float(val), 4)
        # Fallback: order book mid
        book = client.get_order_book(token_id)
        if not book:
            return None
        asks = getattr(book, "asks", None) or book.get("asks", [])
        bids = getattr(book, "bids", None) or book.get("bids", [])
        if asks and bids:
            ask_p = float(getattr(asks[0], "price", None) or asks[0].get("price", 0))
            bid_p = float(getattr(bids[0], "price", None) or bids[0].get("price", 0))
            return round((ask_p + bid_p) / 2, 4)
        return None
    except Exception:
        return None


def _parse_order_response(resp: Any, entry_price: float) -> dict:
    """Normalise the CLOB post_order response to match fast_trader.py's expected shape."""
    if not resp:
        return {
            "success": False,
            "trade_id": None,
            "shares_bought": 0.0,
            "shares": 0.0,
            "average_price": 0.0,
            "error": "Empty response from CLOB",
        }

    if isinstance(resp, dict):
        # success can be a bool flag or a "matched" status string
        success   = resp.get("success", False) or resp.get("status") in ("matched", "MATCHED")
        order_id  = resp.get("orderID") or resp.get("order_id") or resp.get("id")
        error_msg = resp.get("errorMsg") or resp.get("error") or ""

        # makingAmount / takingAmount are in base units (USDC 6 decimals, shares 6 decimals).
        # FOK responses sometimes return top-level makingAmount=0 with fills inside matchedOrders.
        taking_raw = resp.get("takingAmount", "0") or "0"
        making_raw = resp.get("makingAmount", "0") or "0"
        try:
            shares_bought = float(making_raw) / 1e6
        except (ValueError, TypeError):
            shares_bought = 0.0
        # Fallback: sum makingAmount across matchedOrders if top-level is 0
        if shares_bought == 0:
            for fill in (resp.get("matchedOrders") or []):
                try:
                    shares_bought += float(fill.get("makingAmount", "0") or "0") / 1e6
                except (ValueError, TypeError):
                    pass
        try:
            usdc_spent = float(taking_raw) / 1e6
            avg_price  = (usdc_spent / shares_bought) if shares_bought > 0 else entry_price
        except (ValueError, TypeError, ZeroDivisionError):
            avg_price = entry_price
    else:
        success    = getattr(resp, "success", False) or getattr(resp, "status", "") in ("matched", "MATCHED")
        order_id   = getattr(resp, "orderID", None) or getattr(resp, "order_id", None)
        error_msg  = getattr(resp, "errorMsg", "") or getattr(resp, "error", "") or ""
        taking_raw = str(getattr(resp, "takingAmount", "0") or "0")
        making_raw = str(getattr(resp, "makingAmount", "0") or "0")
        try:
            shares_bought = float(making_raw) / 1e6
        except (ValueError, TypeError):
            shares_bought = 0.0
        if shares_bought == 0:
            for fill in (getattr(resp, "matchedOrders", None) or []):
                try:
                    val = fill.get("makingAmount") if isinstance(fill, dict) else getattr(fill, "makingAmount", "0")
                    shares_bought += float(val or "0") / 1e6
                except (ValueError, TypeError):
                    pass
        try:
            usdc_spent = float(taking_raw) / 1e6
            avg_price  = (usdc_spent / shares_bought) if shares_bought > 0 else entry_price
        except (ValueError, TypeError, ZeroDivisionError):
            avg_price = entry_price

    if error_msg:
        success = False

    return {
        "success":       bool(success),
        "trade_id":      order_id,
        "shares_bought": round(shares_bought, 4),
        "shares":        round(shares_bought, 4),
        "average_price": round(avg_price, 4),
        "error":         error_msg or None,
    }


# ---------------------------------------------------------------------------
# CLI: python poly_trade.py <condition_id> <yes|no> <amount_usdc> [--dry]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) < 4:
        print("Usage: python poly_trade.py <condition_id> <yes|no> <amount_usdc> [--dry]")
        sys.exit(1)

    cid    = sys.argv[1]
    side_  = sys.argv[2]
    amount = float(sys.argv[3])
    dry    = "--dry" in sys.argv

    if dry:
        print(f"[DRY] Would execute_trade({cid[:16]}..., {side_}, ${amount})")
        sys.exit(0)

    print(f"Executing trade: {side_.upper()} ${amount} on {cid[:16]}...")
    result = execute_trade(cid, side_, amount)
    if result["success"]:
        print(f"  OK  trade_id={result['trade_id']}  shares={result['shares_bought']:.2f}  avg_price={result['average_price']:.4f}")
    else:
        print(f"  FAIL: {result['error']}")
