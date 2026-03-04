#!/usr/bin/env python3
"""
poly_setup.py — One-time Polymarket wallet setup

Handles both pre-trade requirements in the correct order:
  Step 1: Derive L2 API credentials  (POLY_API_KEY / SECRET / PASSPHRASE)
  Step 2: Set on-chain approvals      (USDC + CTF tokens → Exchange contracts)

Run once per wallet. Safe to re-run — skips anything already done.

Usage:
  python poly_setup.py           # full setup (credentials + approvals)
  python poly_setup.py --check   # read-only status (no transactions)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from polymarket_sdk.poly_auth    import derive_api_key
from polymarket_sdk.poly_approve import check_approvals, set_approvals

CHECK_ONLY = "--check" in sys.argv

# ─────────────────────────────────────────────────────────────────────────────

def step1_credentials():
    import os
    print("─" * 60)
    print("STEP 1: L2 API Credentials")
    print("─" * 60)

    has_key  = bool(os.environ.get("POLY_API_KEY", "").strip())
    has_sec  = bool(os.environ.get("POLY_API_SECRET", "").strip())
    has_pass = bool(os.environ.get("POLY_API_PASSPHRASE", "").strip())

    if has_key and has_sec and has_pass:
        print("  ✓ POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE already set in .env\n")
        return True

    if CHECK_ONLY:
        missing = [k for k, v in [
            ("POLY_API_KEY", has_key),
            ("POLY_API_SECRET", has_sec),
            ("POLY_API_PASSPHRASE", has_pass),
        ] if not v]
        print(f"  ✗ Missing: {', '.join(missing)}")
        print("  Run without --check to derive them.\n")
        return False

    print("  Deriving L2 credentials from POLY_PRIVATE_KEY...")
    try:
        creds = derive_api_key()
    except Exception as e:
        print(f"  ✗ Failed: {e}\n")
        return False

    print()
    print("  Add these 3 lines to your .env file:")
    print()
    print(f"  POLY_API_KEY={creds['api_key']}")
    print(f"  POLY_API_SECRET={creds['api_secret']}")
    print(f"  POLY_API_PASSPHRASE={creds['api_passphrase']}")
    print()
    print("  ✓ Done — save the values above to .env before running the trader.\n")
    return True


def step2_approvals():
    print("─" * 60)
    print("STEP 2: On-Chain Token Approvals")
    print("─" * 60)
    print("  (Allows CTFExchange contracts to move USDC and conditional tokens)")
    print()

    if CHECK_ONLY:
        try:
            checks  = check_approvals()
            all_ok  = all(c["approved"] for c in checks)
            for c in checks:
                mark = "✓" if c["approved"] else "✗"
                print(f"  {mark} {c['label']}")
            print()
            if all_ok:
                print("  All approvals set.\n")
            else:
                print("  Some approvals missing. Run without --check to set them.\n")
            return all_ok
        except Exception as e:
            print(f"  ✗ Error reading approvals: {e}\n")
            return False

    try:
        results = set_approvals(dry_run=False)
        print()
        failed = [r for r in results if r["status"] == "failed"]
        if failed:
            for r in failed:
                print(f"  ✗ {r['label']}: {r.get('error', 'failed')}")
            print()
            return False
        return True
    except Exception as e:
        print(f"  ✗ Approval error: {e}\n")
        return False


def main():
    print()
    print("=" * 60)
    print("  Polymarket Wallet Setup")
    print("=" * 60)
    print()

    ok1 = step1_credentials()
    ok2 = step2_approvals()

    print("=" * 60)
    if ok1 and ok2:
        print("  Setup complete. Ready to trade.")
    else:
        print("  Setup incomplete — see errors above.")
        if not CHECK_ONLY:
            sys.exit(1)
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
