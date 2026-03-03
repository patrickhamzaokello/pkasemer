#!/usr/bin/env python3
"""
poly_positions.py — Component 4: Position Fetcher

Replaces: client.get_positions() → list of position dataclasses

Fetches open positions from Polymarket's Data API.
Returns a list of dicts matching the shape that fast_trader.py expects:
  {
    "market_id":     str,    # condition_id
    "question":      str,
    "side":          str,    # "yes" or "no"
    "shares_yes":    float,
    "shares_no":     float,
    "entry_price":   float,
    "current_price": float,
    "redeemable":    bool,
    "pnl":           float,
  }

Endpoint: https://data-api.polymarket.com/positions?user={address}&limit=500
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


def get_positions(address: str | None = None, size_threshold: float = 0.01) -> list[dict]:
    """
    Fetch all open positions for a wallet address.

    Args:
        address:        Ethereum wallet address. Defaults to POLY_PRIVATE_KEY address.
        size_threshold: Minimum position size to include (filters dust).

    Returns:
        List of position dicts matching fast_trader.py's expected shape.
        Empty list on error.
    """
    if not address:
        try:
            address = get_wallet_address()
        except Exception as e:
            print(f"[poly_positions] Cannot determine wallet address: {e}")
            return []

    url = f"{DATA_API}/positions?user={address}&sizeThreshold={size_threshold}&limit=500"
    result = _http_get(url)

    if not result:
        return []
    if isinstance(result, dict) and result.get("error"):
        print(f"[poly_positions] API error: {result['error']}")
        return []

    positions = result if isinstance(result, list) else result.get("positions", [])
    return [_normalise_position(p) for p in positions if p]


def _normalise_position(raw: dict) -> dict:
    """Map Polymarket Data API position response to fast_trader.py's expected shape."""
    outcome   = (raw.get("outcome") or "").lower()
    size      = float(raw.get("size") or 0)
    avg_price = float(raw.get("avgPrice") or raw.get("avg_price") or 0.5)

    initial_value = float(raw.get("initialValue") or (size * avg_price))
    current_value = float(raw.get("currentValue") or initial_value)
    pnl           = round(current_value - initial_value, 4)

    # Determine share counts by outcome
    shares_yes = size if outcome == "yes" else 0.0
    shares_no  = size if outcome == "no"  else 0.0

    return {
        "market_id":     raw.get("conditionId") or raw.get("market") or "",
        "question":      raw.get("title") or raw.get("question") or "",
        "side":          outcome or "yes",
        "shares_yes":    round(shares_yes, 4),
        "shares_no":     round(shares_no,  4),
        "entry_price":   round(avg_price, 4),
        "current_price": round(float(raw.get("curPrice") or avg_price), 4),
        "redeemable":    bool(raw.get("redeemable", False)),
        "pnl":           pnl,
        "closed":        bool(raw.get("closed", False)),
        "end_date":      raw.get("endDate") or "",
        "token_id":      raw.get("asset") or "",
        "outcome":       outcome,
    }


def is_redeemable(position: dict) -> bool:
    """Return True if this position has winning shares that can be redeemed."""
    return bool(position.get("redeemable")) and (
        position.get("shares_yes", 0) > 0 or position.get("shares_no", 0) > 0
    )


# ---------------------------------------------------------------------------
# CLI: python poly_positions.py [address]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    addr = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Fetching positions for {addr or 'wallet from POLY_PRIVATE_KEY'}...")
    positions = get_positions(addr)

    if not positions:
        print("  No open positions found.")
        sys.exit(0)

    for p in positions:
        side_shares = p["shares_yes"] if p["side"] == "yes" else p["shares_no"]
        redeem_tag = " [REDEEMABLE]" if p["redeemable"] else ""
        print(
            f"  {p['question'][:50]:50}  "
            f"{p['side'].upper():3}  "
            f"{side_shares:.2f} shares  "
            f"P&L=${p['pnl']:+.2f}{redeem_tag}"
        )
