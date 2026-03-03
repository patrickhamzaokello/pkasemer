"""
polymarket_sdk — Direct Polymarket Builder API SDK
==================================================

Drop-in replacement for simmer-sdk.

Quick start:
    from polymarket_sdk import get_poly_client
    client = get_poly_client()

    # Resolve a market (replaces import_market / quota-based system)
    result = client.import_market("btc-updown-5m-1748000000")
    condition_id = result.market_id

    # Execute a trade
    trade = client.trade(condition_id, side="yes", amount=5.0)

    # Query positions and portfolio
    positions = client.get_positions()
    portfolio = client.get_portfolio()

    # Redeem winning positions (on-chain)
    client.redeem(condition_id, side="yes")

Environment variables required:
    POLY_PRIVATE_KEY       Ethereum wallet private key (hex)
    POLY_API_KEY           L2 CLOB API key  (from poly_auth.derive_api_key())
    POLY_API_SECRET        L2 CLOB API secret
    POLY_API_PASSPHRASE    L2 CLOB API passphrase

One-time setup (run once, then save output to .env):
    python -m polymarket_sdk.poly_auth derive
"""

from .poly_client import PolyClient, get_poly_client
from .poly_auth   import derive_api_key, get_wallet_address
from .poly_market import resolve_market, get_token_id, get_market_info, warm_cache
from .poly_trade  import execute_trade
from .poly_positions import get_positions, is_redeemable
from .poly_portfolio import get_portfolio
from .poly_history   import get_trade_history, build_trade_index
from .poly_redeem    import redeem_position, redeem_all_redeemable, is_market_resolved

__all__ = [
    # Main client (primary entry point)
    "PolyClient",
    "get_poly_client",
    # Auth
    "derive_api_key",
    "get_wallet_address",
    # Market resolution
    "resolve_market",
    "get_token_id",
    "get_market_info",
    "warm_cache",
    # Trading
    "execute_trade",
    # Data queries
    "get_positions",
    "is_redeemable",
    "get_portfolio",
    "get_trade_history",
    "build_trade_index",
    # On-chain redemption
    "redeem_position",
    "redeem_all_redeemable",
    "is_market_resolved",
]

__version__ = "1.0.0"
