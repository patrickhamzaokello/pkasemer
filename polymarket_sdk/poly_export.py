#!/usr/bin/env python3
"""
poly_export.py — Export complete trade history to CSV

Fetches ALL trade history (paginated) for your wallet and writes two files:
  - active_trades.csv   : trades in markets that are still open / position held
  - closed_trades.csv   : trades in resolved or exited markets

Usage:
    python -m polymarket_sdk.poly_export
    python -m polymarket_sdk.poly_export --output-dir ./reports
    python -m polymarket_sdk.poly_export --address 0xABC... --output-dir ./reports
"""

if __package__ is None:
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent.parent))
    __package__ = "polymarket_sdk"

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from .poly_auth import get_wallet_address
from .poly_positions import get_positions

DATA_API  = "https://data-api.polymarket.com"
PAGE_SIZE = 500


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 20) -> dict | list | None:
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


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def _fetch_all_activity(address: str) -> list[dict]:
    """
    Paginate through the Data API /activity endpoint until all records are
    returned.  Uses offset-based pagination with PAGE_SIZE steps.
    """
    all_records: list[dict] = []
    offset = 0

    print(f"[poly_export] Fetching all trade history for {address}...")

    while True:
        url = f"{DATA_API}/activity?user={address}&limit={PAGE_SIZE}&offset={offset}"
        result = _http_get(url)

        if not result:
            break
        if isinstance(result, dict) and result.get("error"):
            print(f"[poly_export] API error at offset {offset}: {result['error']}")
            break

        page: list = result if isinstance(result, list) else result.get("history", [])
        if not page:
            break

        all_records.extend(page)
        print(f"[poly_export]   Fetched {len(all_records)} records so far...")

        if len(page) < PAGE_SIZE:
            # Received a partial page — we have reached the end
            break

        offset += PAGE_SIZE

    return all_records


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _ts_to_iso(ts) -> str:
    """Convert a unix timestamp (int or str) to a human-readable ISO string."""
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except (ValueError, OSError):
        return str(ts)


def _normalise_activity(raw: dict) -> dict | None:
    """
    Map one raw Data API activity record to our export shape.
    Returns None for non-trade activity (deposits, withdrawals, etc.).
    """
    trade_type = (raw.get("type") or "TRADE").upper()
    if trade_type not in ("TRADE", "REDEMPTION", "REDEEM"):
        return None

    outcome   = raw.get("outcome") or ""
    side      = outcome.lower() if outcome else "yes"
    shares    = float(raw.get("shares")   or raw.get("size")    or 0)
    price     = float(raw.get("price")    or 0)
    amount    = float(raw.get("amount")   or raw.get("usdcSize") or (shares * price))
    timestamp = int(raw.get("timestamp")  or raw.get("createdAt") or 0)
    payout_raw = raw.get("payout")
    resolved   = bool(
        raw.get("marketClosed") or raw.get("closed") or raw.get("resolved")
    )

    pnl = None
    if resolved and payout_raw is not None:
        try:
            pnl = round(float(payout_raw) - amount, 4)
        except (ValueError, TypeError):
            pnl = None

    return {
        "trade_id":        raw.get("id") or raw.get("transactionHash") or "",
        "condition_id":    raw.get("conditionId") or raw.get("market") or "",
        "token_id":        raw.get("asset") or "",
        "question":        raw.get("title") or raw.get("question") or "",
        "side":            side,
        "type":            trade_type,
        "shares":          round(shares, 4),
        "price":           round(price, 4),
        "amount_paid":     round(amount, 4),
        "timestamp":       timestamp,
        "timestamp_iso":   _ts_to_iso(timestamp),
        "market_resolved": resolved,
        "payout":          float(payout_raw) if payout_raw is not None else None,
        "pnl":             pnl,
    }


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_trades(
    address: str | None = None,
    output_dir: str = ".",
) -> tuple[str, str]:
    """
    Fetch all trade history and split into two CSV files.

    Args:
        address:    Wallet address. Defaults to the POLY_PRIVATE_KEY wallet.
        output_dir: Directory to write CSVs.  Created if it does not exist.

    Returns:
        (active_csv_path, closed_csv_path)
    """
    # Resolve wallet address
    if not address:
        try:
            address = get_wallet_address()
        except Exception as e:
            print(f"[poly_export] Cannot determine wallet address: {e}")
            sys.exit(1)

    # --- Step 1: Fetch raw activity (all pages) ---
    raw_records = _fetch_all_activity(address)
    print(f"[poly_export] Total raw activity records  : {len(raw_records)}")

    # --- Step 2: Normalise to trade records ---
    all_trades: list[dict] = []
    for raw in raw_records:
        t = _normalise_activity(raw)
        if t:
            all_trades.append(t)
    print(f"[poly_export] Trade/Redemption records    : {len(all_trades)}")

    # --- Step 3: Fetch current open positions ---
    print("[poly_export] Fetching current open positions...")
    positions = get_positions(address)

    # Set of condition_ids where the wallet currently holds shares
    active_condition_ids: set[str] = {
        p["market_id"]
        for p in positions
        if not p.get("closed")
        and (p.get("shares_yes", 0) > 0 or p.get("shares_no", 0) > 0)
    }
    position_by_condition: dict[str, dict] = {p["market_id"]: p for p in positions}

    # Back-fill question strings from position data when missing in history
    for trade in all_trades:
        if not trade["question"]:
            pos = position_by_condition.get(trade["condition_id"])
            if pos:
                trade["question"] = pos.get("question", "")

    # --- Step 4: Split active vs closed ----
    active_trades: list[dict] = []
    closed_trades: list[dict] = []
    redeemed_trades: list[dict] = []

    for trade in all_trades:
        cid = trade["condition_id"]
        if cid in active_condition_ids:
            pos = position_by_condition.get(cid, {})
            trade["current_price"]  = pos.get("current_price", "")
            trade["unrealized_pnl"] = pos.get("pnl", "")
            active_trades.append(trade)
        else:
            pnl = trade.get("pnl")
            if pnl is not None:
                if pnl > 0:
                    trade["result"] = "WIN"
                elif pnl < 0:
                    trade["result"] = "LOSS"
                else:
                    trade["result"] = "BREAK_EVEN"
            else:
                trade["result"] = "RESOLVED" if trade["market_resolved"] else "CLOSED"
            
            if(trade["type"] in ("REDEMPTION", "REDEEM")):
                redeemed_trades.append(trade)
            elif(trade["type"] == "TRADE"):
                closed_trades.append(trade)

    # --- Step 5: Write CSVs ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    active_path = str(out_path / "active_trades.csv")
    closed_path = str(out_path / "closed_trades.csv")
    redeemed_path = str(out_path / "redeemed_trades.csv")

    active_fields = [
        "trade_id",
        "timestamp_iso",
        "condition_id",
        "token_id",
        "question",
        "type",
        "side",
        "shares",
        "price",
        "amount_paid",
        "current_price",
        "unrealized_pnl",
    ]
    closed_fields = [
        "trade_id",
        "timestamp_iso",
        "condition_id",
        "token_id",
        "question",
        "type",
        "side",
        "shares",
        "price",
        "amount_paid",
        "payout",
        "pnl",
        "result",
        "market_resolved",
    ]

    with open(active_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=active_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(active_trades)

    with open(closed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=closed_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(closed_trades)

    with open(redeemed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=closed_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(redeemed_trades)

    print(f"[poly_export] Active trades : {len(active_trades):>5}  → {active_path}")
    print(f"[poly_export] Closed trades : {len(closed_trades):>5}  → {closed_path}")
    print(f"[poly_export] Closed trades : {len(redeemed_trades):>5}  → {redeemed_path}")

    return active_path, closed_path,redeemed_path


# ---------------------------------------------------------------------------
# CLI: python -m polymarket_sdk.poly_export [--output-dir DIR] [--address ADDR]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Export all Polymarket trade history to active_trades.csv and closed_trades.csv"
    )
    parser.add_argument(
        "--address",
        default=None,
        help="Wallet address (default: derived from POLY_PRIVATE_KEY)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write CSV files (default: current directory)",
    )
    args = parser.parse_args()

    active_csv, closed_csv, redeemed_csv = export_trades(
        address=args.address,
        output_dir=args.output_dir,
    )
    print("\nDone!")
    print(f"  Active trades : {active_csv}")
    print(f"  Closed trades : {closed_csv}")
    print(f"  redeemed trades : {redeemed_csv}")
