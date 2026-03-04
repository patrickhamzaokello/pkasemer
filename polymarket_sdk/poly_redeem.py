#!/usr/bin/env python3
"""
poly_redeem.py — Component 7: On-Chain Position Redemption

Replaces: client.redeem(market_id, side)

Redeems winning conditional tokens by calling the ConditionalTokens
contract's redeemPositions() function on Polygon.

This is an ON-CHAIN TRANSACTION — it costs a small amount of MATIC for gas.
The wallet (POLY_PRIVATE_KEY) must have MATIC for gas fees.

Contract addresses (Polygon mainnet, chain_id=137):
  ConditionalTokens (CTF): 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
  USDC (PolyUSDC / USDC.e): 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174

Usage:
  result = redeem_position(condition_id="0x...", side="yes")
  # {"success": True, "tx_hash": "0x...", "error": None}

Environment variables:
  POLY_PRIVATE_KEY   Required (signs the transaction)
  POLY_RPC_URL       Optional. Default: https://polygon-rpc.com
"""

if __package__ is None:
    import sys as _sys; from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent)); __package__ = "polymarket_sdk"

import os
from typing import Any

# CTF (ConditionalTokens) contract address on Polygon
CONDITIONAL_TOKENS_ADDR = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_POLYGON_ADDR       = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CHAIN_ID                = 137

# Public Polygon RPC endpoints tried in order.
# Set POLY_RPC_URL in .env to pin a specific one (e.g. an Alchemy/Infura URL).
_FALLBACK_RPCS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://polygon-mainnet.public.blastapi.io",
    "https://1rpc.io/matic",
    "https://polygon-rpc.com",
]

# Minimal ABI — redeemPositions + balance helpers
CTF_ABI = [
    {
        "name": "redeemPositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "outputs": [],
    },
    {
        "name": "payoutNumerators",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "index", "type": "uint256"},
        ],
        "outputs": [{"type": "uint256"}],
    },
    {
        "name": "payoutDenominator",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "outputs": [{"type": "uint256"}],
    },
    {
        "name": "getCollectionId",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSet", "type": "uint256"},
        ],
        "outputs": [{"type": "bytes32"}],
    },
    {
        "name": "getPositionId",
        "type": "function",
        "stateMutability": "pure",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "collectionId", "type": "bytes32"},
        ],
        "outputs": [{"type": "uint256"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "outputs": [{"type": "uint256"}],
    },
]


def _get_web3():
    """
    Import and return a connected Web3 instance.

    If POLY_RPC_URL is set in the environment, use it exclusively.
    Otherwise, try each URL in _FALLBACK_RPCS until one connects.
    """
    try:
        from web3 import Web3
    except ImportError:
        raise ImportError("web3 package not installed. Run: pip install web3>=6.0.0")

    # User-pinned RPC takes priority
    pinned = os.environ.get("POLY_RPC_URL", "").strip()
    if pinned:
        w3 = Web3(Web3.HTTPProvider(pinned))
        if w3.is_connected():
            return w3
        raise ConnectionError(f"Cannot connect to pinned POLY_RPC_URL: {pinned}")

    # Try fallbacks in order
    last_err = ""
    for url in _FALLBACK_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 8}))
            if w3.is_connected():
                return w3
            last_err = f"{url}: not connected"
        except Exception as e:
            last_err = f"{url}: {e}"

    raise ConnectionError(
        f"Could not connect to any Polygon RPC endpoint. Last error: {last_err}\n"
        "Set POLY_RPC_URL=https://... in your .env to use a dedicated endpoint."
    )


def _get_private_key() -> str:
    key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
    if not key:
        raise RuntimeError("POLY_PRIVATE_KEY environment variable is not set")
    if not key.startswith("0x"):
        key = "0x" + key
    return key


def is_market_resolved(condition_id: str) -> bool:
    """
    Check if a Polymarket binary market has been resolved on-chain.

    Returns True if payoutDenominator > 0 (meaning the oracle reported the result).
    """
    try:
        w3  = _get_web3()
        cid = _normalise_condition_id(condition_id, w3)
        ctf = w3.eth.contract(
            address=w3.to_checksum_address(CONDITIONAL_TOKENS_ADDR),
            abi=CTF_ABI,
        )
        denom = ctf.functions.payoutDenominator(cid).call()
        return denom > 0
    except Exception:
        return False


def get_winning_side(condition_id: str) -> str | None:
    """
    Return "yes", "no", or None if not yet resolved.

    For Polymarket binary markets:
      indexSet 1 (bit 0) = YES outcome
      indexSet 2 (bit 1) = NO  outcome
    """
    try:
        w3  = _get_web3()
        cid = _normalise_condition_id(condition_id, w3)
        ctf = w3.eth.contract(
            address=w3.to_checksum_address(CONDITIONAL_TOKENS_ADDR),
            abi=CTF_ABI,
        )
        denom = ctf.functions.payoutDenominator(cid).call()
        if denom == 0:
            return None  # not resolved

        yes_payout = ctf.functions.payoutNumerators(cid, 0).call()
        no_payout  = ctf.functions.payoutNumerators(cid, 1).call()

        if yes_payout > no_payout:
            return "yes"
        elif no_payout > yes_payout:
            return "no"
        else:
            return None  # tie / invalid
    except Exception:
        return None


def redeem_position(
    condition_id: str,
    side: str,
    gas_limit: int = 200_000,
    _nonce: int | None = None,
) -> dict:
    """
    Redeem a winning conditional token position by calling redeemPositions()
    on the Polymarket ConditionalTokens contract on Polygon.

    Args:
        condition_id: Market condition ID (hex string, e.g. "0x1234...")
        side:         "yes" or "no" — which tokens you hold
        gas_limit:    Gas limit for the transaction (default: 200,000)
        _nonce:       Optional nonce override. When redeeming multiple positions
                      pass an incrementing nonce to avoid replacement errors.

    Returns:
        {
            "success":  bool,
            "tx_hash":  str | None,
            "error":    str | None,
        }
    """
    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError:
        return {
            "success": False,
            "tx_hash": None,
            "error": "web3 or eth_account not installed. Run: pip install web3>=6.0.0",
        }

    try:
        w3          = _get_web3()
        private_key = _get_private_key()
        wallet      = Account.from_key(private_key)
        address     = wallet.address

        ctf = w3.eth.contract(
            address=w3.to_checksum_address(CONDITIONAL_TOKENS_ADDR),
            abi=CTF_ABI,
        )

        cid = _normalise_condition_id(condition_id, w3)

        # indexSets: 1 = YES (binary 01 = first partition), 2 = NO (binary 10)
        index_sets     = [1, 2]
        index_set_val  = index_sets[0]

        parent_collection_id = b"\x00" * 32  # bytes32(0) for top-level market

        # Pre-check: verify wallet holds tokens before submitting tx.
        # Attempting redeemPositions with a zero balance reverts on-chain.
        try:
            collection_id = ctf.functions.getCollectionId(
                parent_collection_id, cid, index_set_val
            ).call()
            position_id = ctf.functions.getPositionId(
                w3.to_checksum_address(USDC_POLYGON_ADDR), collection_id
            ).call()
            balance = ctf.functions.balanceOf(address, position_id).call()
            if balance == 0:
                return {
                    "success": False,
                    "tx_hash": None,
                    "error": (
                        f"No {side.upper()} tokens in wallet (balance=0) — "
                        "position already redeemed or tokens held elsewhere"
                    ),
                }
            print(f"  Token balance: {balance / 1e6:.4f} shares — proceeding", flush=True)
        except Exception as bal_err:
            print(f"  [warn] Balance pre-check failed: {bal_err} — proceeding anyway", flush=True)

        # Use caller-supplied nonce when batch-redeeming; otherwise fetch fresh
        nonce = _nonce if _nonce is not None else w3.eth.get_transaction_count(address, "latest")

        # Use EIP-1559 gas pricing (Polygon standard)
        try:
            base_fee    = w3.eth.get_block("latest")["baseFeePerGas"]
            priority    = w3.to_wei(30, "gwei")          # 30 GWEI tip
            max_fee     = base_fee * 2 + priority
            gas_params  = {"maxFeePerGas": max_fee, "maxPriorityFeePerGas": priority}
        except Exception:
            gas_params  = {"gasPrice": w3.eth.gas_price}

        tx = ctf.functions.redeemPositions(
            w3.to_checksum_address(USDC_POLYGON_ADDR),
            parent_collection_id,
            cid,
            index_sets,
        ).build_transaction({
            "chainId": CHAIN_ID,
            "from":    address,
            "nonce":   nonce,
            "gas":     gas_limit,
            **gas_params,
        })

        signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
        raw    = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
        tx_hash = w3.eth.send_raw_transaction(raw)
        tx_hex  = tx_hash.hex() if isinstance(tx_hash, bytes) else tx_hash

        # Print hash immediately — don't leave user wondering
        print(f"  TX submitted: {tx_hex}", flush=True)
        print(f"  Waiting for receipt (45s max)...", flush=True)

        try:
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=45)
            success = receipt.status == 1
            return {
                "success": success,
                "tx_hash": tx_hex,
                "error":   None if success else "Transaction reverted — check PolygonScan",
            }
        except Exception:
            # Timeout: tx was broadcast, may still be pending
            return {
                "success": True,   # treat as optimistic success — tx is in mempool
                "tx_hash": tx_hex,
                "error":   None,
                "pending": True,
            }

    except Exception as e:
        return {
            "success": False,
            "tx_hash": None,
            "error":   str(e),
        }


def redeem_all_redeemable(address: str | None = None) -> list[dict]:
    """
    Convenience: fetch all positions, find redeemable ones, and redeem them.

    Uses poly_positions.get_positions() to find positions where redeemable=True,
    then calls redeem_position() for each, sharing a single nonce counter across
    all transactions to avoid 'replacement transaction underpriced' errors.

    Returns list of results.
    """
    try:
        from .poly_positions import get_positions
    except ImportError:
        from polymarket_sdk.poly_positions import get_positions

    positions = get_positions(address)
    redeemable = [p for p in positions if p.get("redeemable") and p.get("market_id")]

    if not redeemable:
        return []

    # Fetch starting nonce once, increment manually for each tx
    try:
        from eth_account import Account
        w3          = _get_web3()
        private_key = _get_private_key()
        wallet_addr = Account.from_key(private_key).address
        nonce       = w3.eth.get_transaction_count(wallet_addr, "latest")
    except Exception as e:
        print(f"[poly_redeem] Could not initialise nonce: {e}")
        nonce = None

    results = []
    for pos in redeemable:
        cid  = pos.get("market_id", "")
        side = pos.get("side", "yes")
        print(f"[poly_redeem] Redeeming {pos.get('question', cid)[:50]} {side.upper()}...")
        result = redeem_position(cid, side, _nonce=nonce)

        if result["success"] and result.get("pending"):
            print(f"  PENDING  tx={result['tx_hash'][:20]}...  (check PolygonScan)")
        elif result["success"]:
            print(f"  OK  tx={result['tx_hash'][:20]}...")
        else:
            print(f"  SKIP: {result['error']}")

        results.append({"market_id": cid, "side": side, **result})

        # Advance nonce for next tx regardless of outcome
        if nonce is not None:
            nonce += 1

    return results


def _normalise_condition_id(condition_id: str, w3) -> bytes:
    """Convert hex condition_id string to bytes32 for contract call."""
    cid = condition_id.strip()
    if cid.startswith("0x"):
        cid = cid[2:]
    # Pad to 64 hex chars (32 bytes)
    cid = cid.zfill(64)
    return bytes.fromhex(cid)


# ---------------------------------------------------------------------------
# CLI: python polymarket_sdk/poly_redeem.py <condition_id> <yes|no>
#      python polymarket_sdk/poly_redeem.py --all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    # Add project root so absolute imports (polymarket_sdk.*) resolve
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    args = sys.argv[1:]
    if "--all" in args:
        print("Scanning for redeemable positions...")
        results = redeem_all_redeemable()
        if not results:
            print("  No redeemable positions found.")
    elif len(args) >= 2:
        cid_  = args[0]
        side_ = args[1]
        print(f"Redeeming {side_.upper()} on {cid_[:16]}...")
        r = redeem_position(cid_, side_)
        if r["success"]:
            print(f"  OK  tx={r['tx_hash']}")
        else:
            print(f"  FAIL: {r['error']}")
    else:
        print("Usage:")
        print("  python poly_redeem.py <condition_id> <yes|no>   # redeem specific position")
        print("  python poly_redeem.py --all                      # redeem all redeemable")
