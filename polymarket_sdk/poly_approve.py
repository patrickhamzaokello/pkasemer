#!/usr/bin/env python3
"""
poly_approve.py — One-time approval setup for Polymarket direct trading

Before trading on Polymarket's CLOB or redeeming positions, the exchange
contracts need permission to move tokens from your wallet.

Five approvals required (all idempotent — safe to re-run):
  1. USDC.approve(CTFExchange, MAX)             buy orders spend USDC
  2. USDC.approve(NegRiskCTFExchange, MAX)      same for multi-outcome markets
  3. CTF.setApprovalForAll(CTFExchange)         exchange transfers conditional tokens
  4. CTF.setApprovalForAll(NegRiskCTFExchange)  same for multi-outcome markets
  5. CTF.setApprovalForAll(NegRiskAdapter)      NegRisk adapter wraps/unwraps tokens

Run once per wallet. Skips any approval that is already set.

Usage:
  python polymarket_sdk/poly_approve.py           # check + set all approvals
  python polymarket_sdk/poly_approve.py --check   # read-only status check

Environment variables:
  POLY_PRIVATE_KEY   Required (signs transactions)
  POLY_RPC_URL       Optional (falls back to public Polygon RPCs)
"""

if __package__ is None:
    import sys as _sys; from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent)); __package__ = "polymarket_sdk"

import os

# ─── Contract addresses (Polygon mainnet, chain_id=137) ──────────────────────

USDC_ADDR                  = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDR                   = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
CTF_EXCHANGE_ADDR          = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE_ADDR = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
NEG_RISK_ADAPTER_ADDR      = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

CHAIN_ID   = 137
MAX_UINT256 = 2**256 - 1

# Minimum USDC allowance before we re-approve (1 billion USDC — effectively infinite)
MIN_ALLOWANCE = 10**6 * 1_000_000  # 1,000,000 USDC in 6-decimal units

_FALLBACK_RPCS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
    "https://polygon-mainnet.public.blastapi.io",
    "https://1rpc.io/matic",
    "https://polygon-rpc.com",
]

# ─── Minimal ABIs ─────────────────────────────────────────────────────────────

ERC20_ABI = [
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"type": "uint256"}],
    },
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"type": "bool"}],
    },
]

ERC1155_ABI = [
    {
        "name": "isApprovedForAll",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "operator", "type": "address"},
        ],
        "outputs": [{"type": "bool"}],
    },
    {
        "name": "setApprovalForAll",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"},
        ],
        "outputs": [],
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_web3():
    try:
        from web3 import Web3
    except ImportError:
        raise ImportError("web3 not installed. Run: pip install web3>=6.0.0")

    pinned = os.environ.get("POLY_RPC_URL", "").strip()
    if pinned:
        w3 = Web3(Web3.HTTPProvider(pinned))
        if w3.is_connected():
            return w3
        raise ConnectionError(f"Cannot connect to POLY_RPC_URL: {pinned}")

    for url in _FALLBACK_RPCS:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 8}))
            if w3.is_connected():
                return w3
        except Exception:
            continue

    raise ConnectionError("Could not connect to any Polygon RPC endpoint.")


def _get_private_key() -> str:
    key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
    if not key:
        raise RuntimeError("POLY_PRIVATE_KEY not set")
    return key if key.startswith("0x") else "0x" + key


def _gas_params(w3):
    """Return EIP-1559 gas params, fallback to legacy."""
    try:
        base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
        priority = w3.to_wei(30, "gwei")
        return {"maxFeePerGas": base_fee * 2 + priority, "maxPriorityFeePerGas": priority}
    except Exception:
        return {"gasPrice": w3.eth.gas_price}


def _send_tx(w3, tx, private_key: str) -> str:
    """Sign and broadcast a transaction. Returns tx hash hex."""
    signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
    raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
    tx_hash = w3.eth.send_raw_transaction(raw)
    return tx_hash.hex() if isinstance(tx_hash, bytes) else tx_hash


def _wait(w3, tx_hex: str, timeout: int = 60) -> bool:
    """Wait for receipt. Returns True on success, False on revert/timeout."""
    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hex, timeout=timeout)
        return receipt.status == 1
    except Exception:
        return False  # timeout — tx may still confirm


# ─── Status check ─────────────────────────────────────────────────────────────

def check_approvals(address: str | None = None) -> list[dict]:
    """
    Read current approval status for all 5 required approvals.

    Returns a list of dicts:
      {
        "label":    str,    human-readable description
        "type":     "erc20" | "erc1155",
        "approved": bool,
        "contract": str,
        "operator": str,
      }
    """
    from eth_account import Account

    w3          = _get_web3()
    private_key = _get_private_key()
    wallet      = address or Account.from_key(private_key).address

    usdc = w3.eth.contract(address=w3.to_checksum_address(USDC_ADDR), abi=ERC20_ABI)
    ctf  = w3.eth.contract(address=w3.to_checksum_address(CTF_ADDR),  abi=ERC1155_ABI)

    checks = [
        {
            "label":    "USDC → CTFExchange",
            "type":     "erc20",
            "contract": USDC_ADDR,
            "operator": CTF_EXCHANGE_ADDR,
            "approved": usdc.functions.allowance(
                wallet, w3.to_checksum_address(CTF_EXCHANGE_ADDR)
            ).call() >= MIN_ALLOWANCE,
        },
        {
            "label":    "USDC → NegRiskCTFExchange",
            "type":     "erc20",
            "contract": USDC_ADDR,
            "operator": NEG_RISK_CTF_EXCHANGE_ADDR,
            "approved": usdc.functions.allowance(
                wallet, w3.to_checksum_address(NEG_RISK_CTF_EXCHANGE_ADDR)
            ).call() >= MIN_ALLOWANCE,
        },
        {
            "label":    "CTF tokens → CTFExchange",
            "type":     "erc1155",
            "contract": CTF_ADDR,
            "operator": CTF_EXCHANGE_ADDR,
            "approved": ctf.functions.isApprovedForAll(
                wallet, w3.to_checksum_address(CTF_EXCHANGE_ADDR)
            ).call(),
        },
        {
            "label":    "CTF tokens → NegRiskCTFExchange",
            "type":     "erc1155",
            "contract": CTF_ADDR,
            "operator": NEG_RISK_CTF_EXCHANGE_ADDR,
            "approved": ctf.functions.isApprovedForAll(
                wallet, w3.to_checksum_address(NEG_RISK_CTF_EXCHANGE_ADDR)
            ).call(),
        },
        {
            "label":    "CTF tokens → NegRiskAdapter",
            "type":     "erc1155",
            "contract": CTF_ADDR,
            "operator": NEG_RISK_ADAPTER_ADDR,
            "approved": ctf.functions.isApprovedForAll(
                wallet, w3.to_checksum_address(NEG_RISK_ADAPTER_ADDR)
            ).call(),
        },
    ]
    return checks


# ─── Approval setter ──────────────────────────────────────────────────────────

def set_approvals(dry_run: bool = False) -> list[dict]:
    """
    Check all 5 approvals and submit transactions for any that are missing.

    Args:
        dry_run: If True, print what would be done but send no transactions.

    Returns:
        List of result dicts for each approval action taken:
        {"label": str, "status": "already_set"|"approved"|"failed", "tx_hash": str|None}
    """
    from eth_account import Account

    w3          = _get_web3()
    private_key = _get_private_key()
    wallet      = Account.from_key(private_key).address

    print(f"Wallet: {wallet}")
    print(f"RPC connected: {w3.eth.chain_id == CHAIN_ID}\n")

    checks  = check_approvals(wallet)
    nonce   = w3.eth.get_transaction_count(wallet, "latest")
    gas     = _gas_params(w3)
    results = []

    usdc = w3.eth.contract(address=w3.to_checksum_address(USDC_ADDR), abi=ERC20_ABI)
    ctf  = w3.eth.contract(address=w3.to_checksum_address(CTF_ADDR),  abi=ERC1155_ABI)

    for item in checks:
        label    = item["label"]
        approved = item["approved"]
        operator = w3.to_checksum_address(item["operator"])

        if approved:
            print(f"  ✓ {label}: already approved")
            results.append({"label": label, "status": "already_set", "tx_hash": None})
            continue

        if dry_run:
            print(f"  ~ {label}: NOT approved (would submit tx)")
            results.append({"label": label, "status": "dry_run", "tx_hash": None})
            continue

        print(f"  → {label}: approving...", end=" ", flush=True)

        try:
            if item["type"] == "erc20":
                tx = usdc.functions.approve(operator, MAX_UINT256).build_transaction({
                    "chainId": CHAIN_ID,
                    "from":    wallet,
                    "nonce":   nonce,
                    "gas":     100_000,
                    **gas,
                })
            else:
                tx = ctf.functions.setApprovalForAll(operator, True).build_transaction({
                    "chainId": CHAIN_ID,
                    "from":    wallet,
                    "nonce":   nonce,
                    "gas":     100_000,
                    **gas,
                })

            tx_hex = _send_tx(w3, tx, private_key)
            print(f"tx={tx_hex[:20]}... waiting...", end=" ", flush=True)
            ok = _wait(w3, tx_hex)
            status = "approved" if ok else "pending"
            print("OK" if ok else "PENDING (check PolygonScan)")
            results.append({"label": label, "status": status, "tx_hash": tx_hex})
            nonce += 1

        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"label": label, "status": "failed", "tx_hash": None, "error": str(e)})
            nonce += 1  # advance anyway to avoid stuck nonce

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────
# Usage:
#   python polymarket_sdk/poly_approve.py           # check + approve
#   python polymarket_sdk/poly_approve.py --check   # read-only

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    check_only = "--check" in sys.argv
    dry_run    = "--dry"   in sys.argv or check_only

    if check_only:
        print("Checking approval status (no transactions)...\n")
        try:
            checks = check_approvals()
            all_ok = True
            for c in checks:
                mark = "✓" if c["approved"] else "✗"
                print(f"  {mark} {c['label']}")
                if not c["approved"]:
                    all_ok = False
            print()
            if all_ok:
                print("All approvals set. Ready to trade.")
            else:
                print("Some approvals missing. Run without --check to set them.")
        except Exception as e:
            print(f"Error: {e}")
        sys.exit(0)

    print("Setting Polymarket approvals...\n")
    try:
        results = set_approvals(dry_run=dry_run)
        print()
        failed = [r for r in results if r["status"] == "failed"]
        if failed:
            print(f"WARNING: {len(failed)} approval(s) failed:")
            for r in failed:
                print(f"  ✗ {r['label']}: {r.get('error', 'unknown error')}")
            sys.exit(1)
        else:
            print("All approvals complete.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
