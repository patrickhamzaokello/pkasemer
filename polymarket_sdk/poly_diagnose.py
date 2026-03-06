#!/usr/bin/env python3
"""
poly_diagnose.py — Check wallet balances, allowances, and CLOB account state.

Runs all checks without sending any transactions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

import os
from web3 import Web3
from eth_account import Account
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import BalanceAllowanceParams, AssetType, ApiCreds

CLOB_HOST  = "https://clob.polymarket.com"
CHAIN_ID   = 137

USDC_ADDR         = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_EXCHANGE_ADDR = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

_FALLBACK_RPCS = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon.llamarpc.com",
]

ERC20_ABI = [
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "account", "type": "address"}], "outputs": [{"type": "uint256"}]},
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
     "outputs": [{"type": "uint256"}]},
]

def _w3():
    for url in _FALLBACK_RPCS:
        try:
            w = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 8}))
            if w.is_connected():
                return w
        except Exception:
            continue
    raise ConnectionError("No RPC")

def _clob_client(sig_type: int):
    key  = os.environ.get("POLY_PRIVATE_KEY", "").strip()
    if not key.startswith("0x"):
        key = "0x" + key
    ak = os.environ.get("POLY_API_KEY", "").strip()
    sc = os.environ.get("POLY_API_SECRET", "").strip()
    ap = os.environ.get("POLY_API_PASSPHRASE", "").strip()
    creds = ApiCreds(api_key=ak, api_secret=sc, api_passphrase=ap) if (ak and sc and ap) else None
    return ClobClient(host=CLOB_HOST, chain_id=CHAIN_ID, key=key,
                      creds=creds, signature_type=sig_type)

def fmt(raw_balance: int) -> str:
    return f"${raw_balance / 1e6:.4f}"

def main():
    key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
    if not key:
        print("ERROR: POLY_PRIVATE_KEY not set")
        sys.exit(1)
    if not key.startswith("0x"):
        key = "0x" + key
    wallet = Account.from_key(key).address

    print(f"Wallet : {wallet}\n")

    # ── 1. On-chain MATIC balance ─────────────────────────────────────────────
    print("── On-Chain (Polygon RPC) ───────────────────────────────────")
    w3   = _w3()
    usdc = w3.eth.contract(address=w3.to_checksum_address(USDC_ADDR), abi=ERC20_ABI)

    matic_wei   = w3.eth.get_balance(wallet)
    usdc_raw    = usdc.functions.balanceOf(wallet).call()
    allowance   = usdc.functions.allowance(
        wallet, w3.to_checksum_address(CTF_EXCHANGE_ADDR)
    ).call()

    print(f"  MATIC balance    : {matic_wei / 1e18:.6f} MATIC")
    print(f"  USDC balance     : {fmt(usdc_raw)}")
    print(f"  USDC allowance → CTFExchange (0x4D97...): {fmt(allowance)}")
    print()

    # ── 2. CLOB reported balances (both sig types) ────────────────────────────
    print("── CLOB API balances ────────────────────────────────────────")
    for sig_type, label in [(0, "EOA (sig_type=0)"), (1, "POLY_PROXY (sig_type=1)")]:
        try:
            client = _clob_client(sig_type)
            resp   = client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=sig_type)
            )
            if isinstance(resp, dict):
                bal = int(resp.get("balance", 0) or 0)
                alw = int(resp.get("allowance", 0) or 0)
            else:
                bal = int(getattr(resp, "balance", 0) or 0)
                alw = int(getattr(resp, "allowance", 0) or 0)
            print(f"  {label}")
            print(f"    balance   : {fmt(bal)}")
            print(f"    allowance : {fmt(alw)}")
        except Exception as e:
            print(f"  {label}: ERROR — {e}")
    print()

    # ── 3. Diagnosis ──────────────────────────────────────────────────────────
    print("── Diagnosis ────────────────────────────────────────────────")
    if usdc_raw < 2_000_000:   # less than $2
        print("  ✗ On-chain USDC balance is too low to place a $2 order.")
        print("    → Bridge/send USDC to Polygon wallet and retry.")
    elif allowance < 2_000_000:
        print("  ✗ USDC allowance to CTFExchange is insufficient.")
        print("    → Run: python poly_setup.py")
    else:
        print("  ✓ On-chain USDC and allowance look sufficient.")
        print("    If CLOB balance shows $0.00 for both sig types,")
        print("    the CLOB may require a deposit transaction (rare for EOA mode).")
    print()

if __name__ == "__main__":
    main()
