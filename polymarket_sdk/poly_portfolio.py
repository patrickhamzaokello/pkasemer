#!/usr/bin/env python3
"""
poly_portfolio.py — Component 5: Portfolio & Balance

Replaces: client.get_portfolio() → { balance_usdc, ... }

Returns the same key names that fast_trader.py's calculate_position_size() uses:
  {
    "balance_usdc":   float,   # available USDC on CLOB
    "total_exposure": float,   # sum of current position values
    "positions_count": int,
  }

Primary source: CLOB /balance-allowance endpoint (via py-clob-client)
Supplemented with: Data API positions for total_exposure
"""

if __package__ is None:
    import sys as _sys; from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent)); __package__ = "polymarket_sdk"

import json
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

from .poly_auth import get_clob_client, get_wallet_address
from .poly_positions import get_positions

USDC_DECIMALS = 6  # USDC uses 6 decimal places on Polygon

# AssetType.COLLATERAL = USDC (the collateral token)
# AssetType.CONDITIONAL = conditional ERC-1155 tokens
_USDC_ASSET_TYPE = AssetType.COLLATERAL


def get_portfolio(address: str | None = None) -> dict:
    """
    Fetch USDC balance and position exposure.

    Args:
        address: Wallet address. Defaults to POLY_PRIVATE_KEY wallet.

    Returns:
        {
            "balance_usdc":    float,
            "total_exposure":  float,
            "positions_count": int,
            "error":           str | None,
        }
    """
    balance = _get_usdc_balance()
    exposure, count = _get_exposure(address)

    return {
        "balance_usdc":    balance,
        "total_exposure":  exposure,
        "positions_count": count,
        "error":           None,
    }


def _get_usdc_balance() -> float:
    """
    Fetch available USDC balance from CLOB via py-clob-client.
    Returns USDC amount as float (e.g. 5.0 = $5.00).
    """
    try:
        client = get_clob_client()
        params = BalanceAllowanceParams(
            asset_type=_USDC_ASSET_TYPE,
            signature_type=1,          # EOA wallet (0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE)
        )
        resp = client.get_balance_allowance(params)

        if not resp:
            return 0.0

        # Response format: {"balance": "5000000", "allowance": "..."}
        # Balance is in 6-decimal USDC units
        if isinstance(resp, dict):
            raw = resp.get("balance", "0") or "0"
        else:
            raw = getattr(resp, "balance", "0") or "0"

        return round(float(raw) / (10 ** USDC_DECIMALS), 4)

    except Exception as e:
        print(f"[poly_portfolio] Balance fetch error: {e}")
        return 0.0


def _get_exposure(address: str | None) -> tuple[float, int]:
    """
    Sum current position values for total exposure.
    Returns (total_exposure_usd, position_count).
    """
    try:
        if not address:
            address = get_wallet_address()
        positions = get_positions(address)
        if not positions:
            return 0.0, 0

        total = sum(
            p.get("shares_yes", 0) * p.get("entry_price", 0)
            + p.get("shares_no", 0) * (1 - p.get("entry_price", 0))
            for p in positions
        )
        return round(total, 4), len(positions)
    except Exception as e:
        print(f"[poly_portfolio] Exposure calc error: {e}")
        return 0.0, 0


# ---------------------------------------------------------------------------
# CLI: python poly_portfolio.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    print("Fetching portfolio...")
    portfolio = get_portfolio()
    print(f"  USDC balance:    ${portfolio['balance_usdc']:.2f}")
    print(f"  Total exposure:  ${portfolio['total_exposure']:.2f}")
    print(f"  Positions:       {portfolio['positions_count']}")
    if portfolio.get("error"):
        print(f"  Error:           {portfolio['error']}")
