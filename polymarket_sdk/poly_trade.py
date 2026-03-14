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
from py_clob_client.order_builder.constants import BUY, SELL

from .poly_auth import get_clob_client
from .poly_market import get_token_id


MIN_ORDER_USDC   = 1.0   # Polymarket minimum order size in USDC
PRICE_TOLERANCE  = 0.15  # Accept prices within 15% of best ask/bid for depth calculation
FOK_RETRY_FACTOR = 0.60  # On FOK rejection, retry with 60% of original amount


def execute_trade(
    market_id: str,
    side: str,
    amount_usdc: float,
) -> dict:
    """
    Execute a market-buy order on Polymarket CLOB.

    Uses MarketOrderArgs where `amount` is the USDC amount to spend.
    Pre-checks ask-side liquidity and caps the order size to what's available,
    preventing FOK "couldn't be fully filled" rejections.

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

    # ── Liquidity pre-check ──────────────────────────────────────────────────
    # Cap amount_usdc to available ask-side depth so FOK doesn't get killed.
    best_ask, avail_usdc = _get_ask_liquidity_usdc(client, token_id)
    if avail_usdc <= 0:
        return {
            "success": False,
            "trade_id": None,
            "shares_bought": 0.0,
            "shares": 0.0,
            "average_price": 0.0,
            "error": f"No ask-side liquidity (best_ask={best_ask:.4f}). Order skipped.",
        }

    if amount_usdc > avail_usdc:
        capped = round(avail_usdc * 0.95, 2)  # 5% haircut so we don't hit the edge
        print(
            f"[trade] liquidity cap: ${amount_usdc:.2f} → ${capped:.2f} "
            f"(avail=${avail_usdc:.2f} @ ask={best_ask:.4f})",
            flush=True,
        )
        amount_usdc = capped

    if amount_usdc < MIN_ORDER_USDC:
        return {
            "success": False,
            "trade_id": None,
            "shares_bought": 0.0,
            "shares": 0.0,
            "average_price": 0.0,
            "error": (
                f"After liquidity cap, order size ${amount_usdc:.2f} < "
                f"minimum ${MIN_ORDER_USDC:.2f}. Skipped."
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

    # Get mid price for average_price in response (informational only)
    entry_price = _get_mid_price(client, token_id) or 0.0

    try:
        resp   = client.post_order(signed_order, OrderType.FOK)
        result = _parse_order_response(resp, entry_price)
    except Exception as e:
        result = {
            "success": False,
            "trade_id": None,
            "shares_bought": 0.0,
            "shares": 0.0,
            "average_price": 0.0,
            "error": f"Order post failed: {e}",
        }

    # ── FOK retry: if rejected due to insufficient liquidity, halve and retry ──
    if not result["success"] and _is_fill_error(result.get("error", "")):
        retry_amount = round(amount_usdc * FOK_RETRY_FACTOR, 2)
        if retry_amount >= MIN_ORDER_USDC:
            print(
                f"[trade] FOK rejected — retrying with ${retry_amount:.2f} "
                f"(was ${amount_usdc:.2f})",
                flush=True,
            )
            order_args2 = MarketOrderArgs(token_id=token_id, amount=retry_amount, side=BUY)
            try:
                signed2 = client.create_market_order(order_args2)
                resp2   = client.post_order(signed2, OrderType.FOK)
                result  = _parse_order_response(resp2, entry_price)
            except Exception as _retry_err:
                result["error"] = f"Retry failed: {_retry_err}"

    return result


def _get_ask_liquidity_usdc(
    client: Any, token_id: str
) -> tuple[float, float]:
    """
    Return (best_ask_price, total_usdc_available) across ask levels within
    PRICE_TOLERANCE of the best ask.  Used to cap FOK order size before posting.
    """
    try:
        book = client.get_order_book(token_id)
        if not book:
            return 0.0, 0.0
        asks = getattr(book, "asks", None)
        if asks is None:
            asks = book.get("asks", []) if isinstance(book, dict) else []
        if not asks:
            return 0.0, 0.0

        def _p(a: Any) -> float:
            v = getattr(a, "price", None)
            return float(v if v is not None else a.get("price", 0))

        def _s(a: Any) -> float:
            v = getattr(a, "size", None)
            return float(v if v is not None else a.get("size", 0))

        sorted_asks = sorted(asks, key=_p)
        best_ask = _p(sorted_asks[0])
        if best_ask <= 0:
            return 0.0, 0.0

        ceiling = best_ask * (1 + PRICE_TOLERANCE)
        total_usdc = sum(
            _p(a) * _s(a) for a in sorted_asks if _p(a) <= ceiling
        )
        return best_ask, round(total_usdc, 4)
    except Exception:
        return 0.0, 0.0


def _get_bid_liquidity_shares(
    client: Any, token_id: str
) -> tuple[float, float]:
    """
    Return (best_bid_price, total_shares_available) across bid levels within
    PRICE_TOLERANCE of the best bid.  Used to cap SELL FOK order size.
    """
    try:
        book = client.get_order_book(token_id)
        if not book:
            return 0.0, 0.0
        bids = getattr(book, "bids", None)
        if bids is None:
            bids = book.get("bids", []) if isinstance(book, dict) else []
        if not bids:
            return 0.0, 0.0

        def _p(b: Any) -> float:
            v = getattr(b, "price", None)
            return float(v if v is not None else b.get("price", 0))

        def _s(b: Any) -> float:
            v = getattr(b, "size", None)
            return float(v if v is not None else b.get("size", 0))

        sorted_bids = sorted(bids, key=_p, reverse=True)
        best_bid = _p(sorted_bids[0])
        if best_bid <= 0:
            return 0.0, 0.0

        floor = best_bid * (1 - PRICE_TOLERANCE)
        total_shares = sum(_s(b) for b in sorted_bids if _p(b) >= floor)
        return best_bid, round(total_shares, 4)
    except Exception:
        return 0.0, 0.0


def _is_fill_error(error_msg: str) -> bool:
    """Return True if the error string indicates a FOK fill-size rejection."""
    if not error_msg:
        return False
    msg = error_msg.lower()
    return any(phrase in msg for phrase in (
        "could not be fully filled",
        "not fully filled",
        "insufficient liquidity",
        "fok",
        "fill or kill",
    ))


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


def sell_shares(
    market_id: str,
    side: str,
    shares: float,
) -> dict:
    """
    Sell existing conditional token shares via SELL market order.

    Unlike execute_trade() where amount is USDC, here amount is shares (tokens).
    Used for early exit when the signal flips direction mid-window.

    Args:
        market_id: condition_id (returned by resolve_market)
        side:      "yes" or "no" — the side you hold and want to sell
        shares:    number of tokens to sell

    Returns:
        {"success": bool, "trade_id": str|None, "error": str|None}
    """
    if shares <= 0:
        return {"success": False, "trade_id": None, "error": f"shares must be > 0, got {shares}"}

    client = get_clob_client()
    token_id = get_token_id(market_id, side)
    if not token_id:
        return {
            "success": False,
            "trade_id": None,
            "error": f"No token_id found for market_id={market_id} side={side}.",
        }

    # ── Bid-side liquidity check: cap shares to available bid depth ──────────
    best_bid, avail_shares = _get_bid_liquidity_shares(client, token_id)
    if avail_shares <= 0:
        return {
            "success": False,
            "trade_id": None,
            "error": f"No bid-side liquidity (best_bid={best_bid:.4f}). Sell skipped.",
        }
    if shares > avail_shares:
        capped = round(avail_shares * 0.95, 4)
        print(
            f"[sell] liquidity cap: {shares:.2f} → {capped:.2f} shares "
            f"(avail={avail_shares:.2f} @ bid={best_bid:.4f})",
            flush=True,
        )
        shares = capped

    order_args = MarketOrderArgs(
        token_id=token_id,
        amount=shares,
        side=SELL,
    )
    try:
        signed_order = client.create_market_order(order_args)
    except Exception as e:
        return {"success": False, "trade_id": None, "error": f"Order signing failed: {e}"}

    try:
        resp   = client.post_order(signed_order, OrderType.FOK)
        result = _parse_order_response(resp, 0.0)
    except Exception as e:
        result = {"success": False, "trade_id": None, "shares_bought": 0.0, "shares": 0.0, "average_price": 0.0, "error": f"Order post failed: {e}"}

    # ── FOK retry: halve shares and retry once on fill rejection ─────────────
    if not result["success"] and _is_fill_error(result.get("error", "")):
        retry_shares = round(shares * FOK_RETRY_FACTOR, 4)
        if retry_shares > 0:
            print(
                f"[sell] FOK rejected — retrying with {retry_shares:.2f} shares "
                f"(was {shares:.2f})",
                flush=True,
            )
            order_args2 = MarketOrderArgs(token_id=token_id, amount=retry_shares, side=SELL)
            try:
                signed2 = client.create_market_order(order_args2)
                resp2   = client.post_order(signed2, OrderType.FOK)
                result  = _parse_order_response(resp2, 0.0)
            except Exception as _retry_err:
                result["error"] = f"Retry failed: {_retry_err}"

    return {
        "success":  result["success"],
        "trade_id": result["trade_id"],
        "error":    result["error"],
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
