#!/usr/bin/env python3
"""
poly_auth.py — Component 1: Authentication & ClobClient Factory

Replaces: SimmerClient(api_key=..., venue="polymarket")

Provides:
  - get_clob_client()   → singleton ClobClient, ready for all CLOB calls
  - derive_api_key()    → one-time L1→L2 key derivation (run once, save to .env)
  - get_wallet_address() → returns the Ethereum address for POLY_PRIVATE_KEY

Environment variables required:
  POLY_PRIVATE_KEY       Ethereum private key (hex, with or without 0x prefix)
  POLY_API_KEY           L2 API key (from derive_api_key, or set manually)
  POLY_API_SECRET        L2 API secret
  POLY_API_PASSPHRASE    L2 API passphrase
  POLY_CHAIN_ID          Optional. Default: 137 (Polygon mainnet)
  POLY_FUNDER            Optional. Separate funder wallet address
"""

import os

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from eth_account import Account

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID  = 137  # Polygon mainnet

_client: ClobClient = None


def get_clob_client() -> ClobClient:
    """Return singleton ClobClient. Reads env vars on first call."""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def reset_client():
    """Force the singleton to be rebuilt on next get_clob_client() call."""
    global _client
    _client = None


def _build_client() -> ClobClient:
    private_key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
    if not private_key:
        raise RuntimeError(
            "POLY_PRIVATE_KEY environment variable is not set.\n"
            "Set it to your Ethereum wallet private key (hex string)."
        )

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    chain_id       = int(os.environ.get("POLY_CHAIN_ID", CHAIN_ID))
    api_key        = os.environ.get("POLY_API_KEY", "").strip()
    api_secret     = os.environ.get("POLY_API_SECRET", "").strip()
    api_passphrase = os.environ.get("POLY_API_PASSPHRASE", "").strip()
    funder         = os.environ.get("POLY_FUNDER", "").strip() or None

    if api_key and api_secret and api_passphrase:
        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
    else:
        # Auto-derive L2 credentials from the L1 wallet key.
        # This makes a signed request to the CLOB — save the output to .env
        # (POLY_API_KEY / POLY_API_SECRET / POLY_API_PASSPHRASE) to skip this
        # on every startup.
        print("[poly_auth] POLY_API_KEY not set — deriving L2 credentials from wallet key...")
        temp  = ClobClient(host=CLOB_HOST, chain_id=chain_id, key=private_key)
        raw   = temp.create_or_derive_api_creds()
        _key  = raw.api_key        if hasattr(raw, "api_key")        else raw["api_key"]
        _sec  = raw.api_secret     if hasattr(raw, "api_secret")     else raw["api_secret"]
        _pass = raw.api_passphrase if hasattr(raw, "api_passphrase") else raw["api_passphrase"]
        creds = ApiCreds(api_key=_key, api_secret=_sec, api_passphrase=_pass)
        print("[poly_auth] L2 credentials derived. Add these to your .env to skip this step:")
        print(f"  POLY_API_KEY={_key}")
        print(f"  POLY_API_SECRET={_sec}")
        print(f"  POLY_API_PASSPHRASE={_pass}")

    client = ClobClient(
        host=CLOB_HOST,
        chain_id=chain_id,
        key=private_key,
        creds=creds,
        funder=funder,
    )
    return client


def derive_api_key() -> dict:
    """
    One-time operation: derive L2 API credentials from your L1 wallet key.

    Run this once, then add the returned values to your .env:
        POLY_API_KEY=...
        POLY_API_SECRET=...
        POLY_API_PASSPHRASE=...

    Returns:
        dict with keys: api_key, api_secret, api_passphrase
    """
    private_key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
    if not private_key:
        raise RuntimeError("POLY_PRIVATE_KEY environment variable is not set")

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    chain_id = int(os.environ.get("POLY_CHAIN_ID", CHAIN_ID))

    # Build a key-only client (no creds yet) just for derivation
    temp_client = ClobClient(host=CLOB_HOST, chain_id=chain_id, key=private_key)

    # create_or_derive_api_creds: creates if not exists, derives if already exists
    creds = temp_client.create_or_derive_api_creds()

    if isinstance(creds, dict):
        return creds

    return {
        "api_key":        creds.api_key,
        "api_secret":     creds.api_secret,
        "api_passphrase": creds.api_passphrase,
    }


def get_wallet_address() -> str:
    """Return the Ethereum address derived from POLY_PRIVATE_KEY."""
    # Try using ClobClient.get_address() first (uses the signer built into client)
    try:
        client = get_clob_client()
        return client.get_address()
    except Exception:
        pass
    # Fallback: derive from private key directly
    private_key = os.environ.get("POLY_PRIVATE_KEY", "").strip()
    if not private_key:
        raise RuntimeError("POLY_PRIVATE_KEY environment variable is not set")
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    return Account.from_key(private_key).address


# ---------------------------------------------------------------------------
# CLI: run `python poly_auth.py derive` to get your L2 keys
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"

    if cmd == "derive":
        print("Deriving L2 API key from POLY_PRIVATE_KEY …")
        result = derive_api_key()
        print("\nAdd these to your .env file:")
        print(f"  POLY_API_KEY={result['api_key']}")
        print(f"  POLY_API_SECRET={result['api_secret']}")
        print(f"  POLY_API_PASSPHRASE={result['api_passphrase']}")
    elif cmd == "info":
        addr = get_wallet_address()
        print(f"Wallet address: {addr}")
        client = get_clob_client()
        print(f"CLOB host:      {CLOB_HOST}")
        print("ClobClient:     OK")
    else:
        print(f"Unknown command: {cmd}. Use: derive | info")
