#!/usr/bin/env python3
"""
poly_history.py — Component 6: Trade History & P&L Backfill

Replaces:
  GET https://api.simmer.markets/api/sdk/trades?limit=200&venue=polymarket

Fetches trade fills from Polymarket's Data API and formats them for
fast_trader.py's backfill_trade_outcomes() function.

Endpoint: https://data-api.polymarket.com/activity?user={address}&limit={n}

P&L calculation (replaces Simmer's "cost" field):
  Win:  pnl = shares * 1.0 - amount_paid   (resolved YES and held YES shares)
  Loss: pnl = 0 - amount_paid              (resolved against your position)
  Open: pnl = None                         (market not yet resolved)
"""

if __package__ is None:
    import sys as _sys; from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent)); __package__ = "polymarket_sdk"

import json
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from .poly_auth import get_wallet_address

DATA_API = "https://data-api.polymarket.com"


def _http_get(url: str, timeout: int = 15) -> dict | list | None:
    try:
        req = Request(url, headers={"User-Agent": "polymarket-sdk/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
            return {"error": body.get("detail", str(e)), "status_code": e.code}
        except Exception:
            return {"error": str(e), "status_code": e.code}
    except (URLError, Exception) as e:
        return {"error": str(e)}


def get_trade_history(address: str | None = None, limit: int = 200) -> list[dict]:
    """
    Fetch recent trade activity for a wallet address.

    Args:
        address: Ethereum wallet address. Defaults to POLY_PRIVATE_KEY wallet.
        limit:   Maximum number of records to fetch.

    Returns:
        List of trade dicts, each shaped to match what backfill_trade_outcomes()
        expects from the old Simmer response:
        {
            "id":           str,     # order ID  (maps to trade_id in local log)
            "condition_id": str,     # market condition ID
            "side":         str,     # "yes" or "no"
            "shares":       float,
            "price":        float,
            "amount_paid":  float,   # USDC spent (from order fill)
            "timestamp":    int,     # unix timestamp
            "type":         str,     # "TRADE" | "REDEEM" etc
            "outcome":      str,     # "Yes" | "No"
            "market_resolved": bool,
            "payout":       float | None,   # USDC received on resolution (if resolved)
            "pnl":          float | None,   # computed pnl if resolvable
        }
    """
    if not address:
        try:
            address = get_wallet_address()
        except Exception as e:
            print(f"[poly_history] Cannot determine wallet address: {e}")
            return []

    url = f"{DATA_API}/activity?user={address}&limit={limit}"
    result = _http_get(url)

    if not result:
        return []
    if isinstance(result, dict) and result.get("error"):
        print(f"[poly_history] API error: {result['error']}")
        return []

    activities = result if isinstance(result, list) else result.get("history", [])
    trades = []
    for item in activities:
        if not item:
            continue
        t = _normalise_activity(item)
        if t:
            trades.append(t)
    return trades


def _normalise_activity(raw: dict) -> dict | None:
    """Map Polymarket Data API activity record to backfill-compatible shape."""
    trade_type = (raw.get("type") or "TRADE").upper()

    # Only process actual trades (not deposits, withdrawals, etc.)
    if trade_type not in ("TRADE", "REDEMPTION", "REDEEM"):
        return None

    outcome   = raw.get("outcome") or ""
    side      = outcome.lower() if outcome else "yes"
    shares    = float(raw.get("shares")  or raw.get("size")   or 0)
    price     = float(raw.get("price")   or 0)
    amount    = float(raw.get("amount")  or raw.get("usdcSize") or (shares * price))
    timestamp = int(raw.get("timestamp") or raw.get("createdAt") or 0)

    # P&L: available only after market resolution
    # Data API may include payout info for resolved markets
    payout   = raw.get("payout")
    resolved = bool(raw.get("marketClosed") or raw.get("closed") or raw.get("resolved"))

    pnl = None
    if resolved and payout is not None:
        try:
            pnl = round(float(payout) - amount, 4)
        except (ValueError, TypeError):
            pnl = None

    return {
        "id":              raw.get("id") or raw.get("transactionHash") or "",
        "condition_id":    raw.get("conditionId") or raw.get("market") or "",
        "side":            side,
        "shares":          round(shares, 4),
        "price":           round(price, 4),
        "amount_paid":     round(amount, 4),
        "timestamp":       timestamp,
        "type":            trade_type,
        "outcome":         outcome,
        "market_resolved": resolved,
        "payout":          float(payout) if payout is not None else None,
        "pnl":             pnl,
        # Simmer compat field: "cost" = proceeds from winning position
        # If we have payout data, expose it; otherwise None
        "cost":            float(payout) if payout is not None else None,
    }


def build_trade_index(address: str | None = None, limit: int = 200) -> dict:
    """
    Return a dict keyed by order ID for efficient lookup in backfill_trade_outcomes().

    Usage in backfill:
        trade_index = build_trade_index()
        for local_trade in local_trades:
            tid = local_trade.get("trade_id")
            if tid and tid in trade_index:
                remote = trade_index[tid]
                local_trade["pnl"] = remote["pnl"]
    """
    trades = get_trade_history(address, limit)
    return {t["id"]: t for t in trades if t.get("id")}


# ---------------------------------------------------------------------------
# CLI: python poly_history.py [limit]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(f"Fetching last {lim} trades...")
    trades = get_trade_history(limit=lim)

    if not trades:
        print("  No trade history found.")
        sys.exit(0)

    for t in trades:
        pnl_str = f"  P&L=${t['pnl']:+.2f}" if t["pnl"] is not None else ""
        print(
            f"  {t['id'][:12]}...  "
            f"{t['side'].upper():3}  "
            f"{t['shares']:.2f} shares @ ${t['price']:.4f}  "
            f"paid=${t['amount_paid']:.2f}{pnl_str}"
        )
