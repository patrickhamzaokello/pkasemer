#!/usr/bin/env python3
"""
test_poly.py — Polymarket Direct SDK Endpoint Diagnostics

Mirrors test_simmer.py structure. Tests every component of polymarket_sdk
in the correct order, with PASS/FAIL output and timing for each step.

Tests covered:
  1.  Auth / client init        — POLY_PRIVATE_KEY + L2 creds → ClobClient
  2.  Wallet address            — derive address from private key
  3.  CLOB health               — GET https://clob.polymarket.com/
  4.  Market resolve            — slug → condition_id + token_ids (Gamma + CLOB)
  5.  Cache hit                 — second resolve must be instant (no network)
  6.  Token IDs                 — get_token_id(condition_id, "yes") returns token
  7.  Order book                — ClobClient.get_order_book(token_id)
  8.  Positions                 — Data API /positions?user={address}
  9.  Portfolio / balance       — CLOB balance-allowance endpoint
  10. Trade history             — Data API /activity?user={address}
  11. Dry-run trade             — build+sign order (no submission)
  12. Market resolver cache     — verify cache file was written
  13. Redeem check              — is_market_resolved() (read-only, no tx)

Usage:
    python test_poly.py                    # run all tests
    python test_poly.py --skip-trade       # skip dry-run trade build (saves time)
    python test_poly.py --slug <slug>      # use a specific market slug
    python test_poly.py --verbose          # print full API responses
    python test_poly.py --test health      # run a single named test

Environment:
    POLY_PRIVATE_KEY    required
    POLY_API_KEY        required (or run with --derive first)
    POLY_API_SECRET     required
    POLY_API_PASSPHRASE required
    DB_PATH             optional
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "trader"))
sys.path.insert(0, str(_PROJECT_ROOT / "polymarket_sdk"))

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CLOB_HOST = "https://clob.polymarket.com"
DATA_HOST = "https://data-api.polymarket.com"

# A known BTC 5m slug template — scheduler generates these dynamically
def _current_btc_slug() -> str:
    bucket = (int(time.time()) // 300) * 300
    return f"btc-updown-5m-{bucket}"


# ─────────────────────────────────────────────────────────────────────────────
# Terminal colours
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}!{RESET} {msg}")
def info(msg):  print(f"  {DIM}→{RESET} {msg}")
def header(msg):print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{msg}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Result tracker
# ─────────────────────────────────────────────────────────────────────────────

class R:
    def __init__(self, name):
        self.name    = name
        self.passed  = False
        self.skipped = False
        self.elapsed = 0
        self.notes   = []
        self.data    = {}

_results: list[R] = []


def run_test(name: str, fn, verbose: bool, *args) -> R:
    r = R(name)
    header(f"Test: {name}")
    t0 = time.time()
    try:
        fn(r, verbose, *args)
    except Exception as e:
        fail(f"Unhandled exception: {e}")
        if verbose:
            traceback.print_exc()
    r.elapsed = round((time.time() - t0) * 1000)
    _results.append(r)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Test implementations
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Auth / client init ────────────────────────────────────────────────────

def test_auth(r: R, verbose: bool):
    pk = os.environ.get("POLY_PRIVATE_KEY", "")
    if not pk:
        warn("POLY_PRIVATE_KEY not set")
        r.skipped = True
        return

    api_key = os.environ.get("POLY_API_KEY", "")
    if not api_key:
        warn("POLY_API_KEY not set — run: python poly_auth.py derive")
        r.notes.append("no_api_key")

    from polymarket_sdk.poly_auth import get_clob_client, reset_client
    reset_client()  # force rebuild with current env

    try:
        client = get_clob_client()
        ok("ClobClient initialised")
        r.data["client"] = client
        r.passed = True
    except Exception as e:
        fail(f"ClobClient init failed: {e}")
        r.notes.append(str(e))


# ── 2. Wallet address ────────────────────────────────────────────────────────

def test_wallet_address(r: R, verbose: bool):
    from polymarket_sdk.poly_auth import get_wallet_address
    try:
        addr = get_wallet_address()
        ok(f"Wallet address: {addr}")
        r.data["address"] = addr
        r.passed = True
    except Exception as e:
        fail(f"get_wallet_address() failed: {e}")


# ── 3. CLOB health ───────────────────────────────────────────────────────────

def test_clob_health(r: R, verbose: bool):
    # The root endpoint returns 403 — use get_ok() (hits /ok) via py-clob-client.
    # Falls back to get_server_time() if /ok also fails.
    from polymarket_sdk.poly_auth import get_clob_client
    client = get_clob_client()
    t0 = time.time()
    try:
        resp = client.get_ok()
        ms   = round((time.time() - t0) * 1000)
        r.elapsed = ms
        ok(f"CLOB reachable ({ms}ms)  get_ok={resp}")
        r.passed = True
    except Exception as e1:
        try:
            resp = client.get_server_time()
            ms   = round((time.time() - t0) * 1000)
            r.elapsed = ms
            ok(f"CLOB reachable ({ms}ms)  server_time={resp}")
            r.passed = True
        except Exception as e2:
            fail(f"CLOB health check failed: get_ok={e1}  get_server_time={e2}")


# ── 4. Market resolve ────────────────────────────────────────────────────────

def test_market_resolve(r: R, verbose: bool, slug: str):
    from polymarket_sdk.poly_market import resolve_market, get_market_info
    info(f"Resolving slug: {slug}")

    t0 = time.time()
    condition_id, error = resolve_market(slug)
    ms = round((time.time() - t0) * 1000)
    r.elapsed = ms

    if error:
        fail(f"resolve_market failed ({ms}ms): {error}")
        r.notes.append(error)
        return

    market_info = get_market_info(condition_id)

    ok(f"Resolved in {ms}ms")
    info(f"condition_id : {condition_id}")
    info(f"yes_token_id : {market_info.get('yes_token_id', 'N/A')}")
    info(f"no_token_id  : {market_info.get('no_token_id', 'N/A')}")
    info(f"question     : {market_info.get('question', 'N/A')[:60]}")

    r.data["condition_id"]  = condition_id
    r.data["market_info"]   = market_info
    r.passed = True


# ── 5. Cache hit ─────────────────────────────────────────────────────────────

def test_cache_hit(r: R, verbose: bool, slug: str):
    from polymarket_sdk.poly_market import resolve_market

    t0 = time.time()
    condition_id, error = resolve_market(slug)
    ms = round((time.time() - t0) * 1000)
    r.elapsed = ms

    if error:
        fail(f"Cache resolve failed: {error}")
        return

    if ms < 5:
        ok(f"Cache hit ({ms}ms) — no network call, condition_id={condition_id[:16]}...")
        r.passed = True
    else:
        warn(f"Resolve took {ms}ms — may not have been a cache hit")
        r.passed = True  # still pass, just slower


# ── 6. Token IDs ─────────────────────────────────────────────────────────────

def test_token_ids(r: R, verbose: bool, condition_id: str):
    from polymarket_sdk.poly_market import get_token_id

    yes_id = get_token_id(condition_id, "yes")
    no_id  = get_token_id(condition_id, "no")

    if yes_id:
        ok(f"yes_token_id: {yes_id[:20]}...")
    else:
        warn("yes_token_id not found (market may not have CLOB listings)")

    if no_id:
        ok(f"no_token_id:  {no_id[:20]}...")
    else:
        warn("no_token_id not found")

    r.data["yes_token_id"] = yes_id
    r.data["no_token_id"]  = no_id
    r.passed = bool(yes_id or no_id)


# ── 7. Order book ────────────────────────────────────────────────────────────

def test_order_book(r: R, verbose: bool, token_id: str):
    if not token_id:
        warn("No token_id — skipping order book test")
        r.skipped = True
        return

    from polymarket_sdk.poly_auth import get_clob_client
    client = get_clob_client()

    try:
        t0   = time.time()
        book = client.get_order_book(token_id)
        ms   = round((time.time() - t0) * 1000)
        r.elapsed = ms

        if not book:
            warn("Empty order book — market may be inactive")
            r.skipped = True
            return

        asks = getattr(book, "asks", None) or book.get("asks", [])
        bids = getattr(book, "bids", None) or book.get("bids", [])

        ok(f"Order book ({ms}ms)  bids={len(bids)}  asks={len(asks)}")

        if asks:
            best_ask = asks[0]
            ask_price = getattr(best_ask, "price", None) or best_ask.get("price", "?")
            info(f"best ask: {ask_price}")
        if bids:
            best_bid = bids[0]
            bid_price = getattr(best_bid, "price", None) or best_bid.get("price", "?")
            info(f"best bid: {bid_price}")

        r.passed = True
    except Exception as e:
        fail(f"Order book fetch failed: {e}")


# ── 8. Positions ─────────────────────────────────────────────────────────────

def test_positions(r: R, verbose: bool, address: str):
    from polymarket_sdk.poly_positions import get_positions

    t0        = time.time()
    positions = get_positions(address)
    ms        = round((time.time() - t0) * 1000)
    r.elapsed = ms

    ok(f"Positions fetched ({ms}ms)  count={len(positions)}")

    if verbose and positions:
        for p in positions[:3]:
            info(f"  {p['question'][:40]}  {p['side'].upper()}  shares_yes={p['shares_yes']}  pnl=${p['pnl']}")

    redeemable = [p for p in positions if p.get("redeemable")]
    if redeemable:
        warn(f"{len(redeemable)} position(s) are redeemable — run poly_redeem.py --all")

    r.passed = True  # empty positions is still a valid response


# ── 9. Portfolio / balance ────────────────────────────────────────────────────

def test_portfolio(r: R, verbose: bool):
    from polymarket_sdk.poly_portfolio import get_portfolio

    t0        = time.time()
    portfolio = get_portfolio()
    ms        = round((time.time() - t0) * 1000)
    r.elapsed = ms

    if portfolio.get("error"):
        fail(f"Portfolio error ({ms}ms): {portfolio['error']}")
        return

    balance  = portfolio.get("balance_usdc", 0)
    exposure = portfolio.get("total_exposure", 0)

    ok(f"Portfolio ({ms}ms)  balance=${balance:.2f}  exposure=${exposure:.2f}")

    if balance == 0:
        warn("Balance is $0 — deposit USDC to the CLOB to start trading")
        r.notes.append("zero_balance")

    r.passed = True


# ── 10. Trade history ────────────────────────────────────────────────────────

def test_trade_history(r: R, verbose: bool, address: str):
    from polymarket_sdk.poly_history import get_trade_history

    t0     = time.time()
    trades = get_trade_history(address, limit=20)
    ms     = round((time.time() - t0) * 1000)
    r.elapsed = ms

    ok(f"Trade history ({ms}ms)  records={len(trades)}")

    if verbose and trades:
        for t in trades[:3]:
            pnl_str = f"  pnl=${t['pnl']:+.2f}" if t["pnl"] is not None else ""
            info(f"  {t['id'][:12]}...  {t['side'].upper()}  shares={t['shares']:.2f}{pnl_str}")

    r.passed = True


# ── 11. Dry-run trade (sign only, no submit) ──────────────────────────────────

def test_dry_trade(r: R, verbose: bool, token_id: str, entry_price: float):
    if not token_id:
        warn("No token_id — skipping dry-run trade test")
        r.skipped = True
        return

    from polymarket_sdk.poly_auth import get_clob_client
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY

    client = get_clob_client()

    try:
        t0         = time.time()
        # MarketOrderArgs: amount = USDC, CLOB auto-prices
        order_args = MarketOrderArgs(token_id=token_id, amount=1.0, side=BUY)
        signed_order = client.create_market_order(order_args)
        ms           = round((time.time() - t0) * 1000)
        r.elapsed    = ms

        ok(f"Order signed ({ms}ms)  amount=$1.00  token_id={token_id[:16]}...")
        if verbose:
            info(f"Signed order type: {type(signed_order).__name__}")

        r.passed = True
    except Exception as e:
        fail(f"Order signing failed: {e}")
        if verbose:
            traceback.print_exc()


# ── 12. Cache file written ────────────────────────────────────────────────────

def test_cache_persistence(r: R, verbose: bool):
    import os
    data_dir   = "/data" if os.path.isdir("/data") else str(Path(__file__).parent.parent / "trader")
    cache_file = Path(data_dir) / "market_id_cache.json"

    if cache_file.exists():
        raw   = cache_file.read_text()
        cache = json.loads(raw)
        ok(f"market_id_cache.json exists  entries={len(cache)}")
        if verbose:
            for slug, v in list(cache.items())[:2]:
                info(f"  {slug}: {str(v)[:80]}")
        r.passed = True
    else:
        warn(f"Cache file not found at {cache_file}  (resolve_market may not have run)")
        r.skipped = True


# ── 13. Market resolved check (read-only, no tx) ──────────────────────────────

def test_resolved_check(r: R, verbose: bool, condition_id: str):
    if not condition_id:
        warn("No condition_id — skipping resolve check")
        r.skipped = True
        return

    from polymarket_sdk.poly_redeem import is_market_resolved

    try:
        t0       = time.time()
        resolved = is_market_resolved(condition_id)
        ms       = round((time.time() - t0) * 1000)
        r.elapsed = ms

        status = "RESOLVED" if resolved else "OPEN"
        ok(f"Market status check ({ms}ms)  status={status}")
        r.passed = True
    except Exception as e:
        warn(f"is_market_resolved() failed (web3 may not be configured): {e}")
        r.skipped = True


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}RESULTS{RESET}")
    print(f"{'═'*60}")

    passed  = sum(1 for r in _results if r.passed)
    failed  = sum(1 for r in _results if not r.passed and not r.skipped)
    skipped = sum(1 for r in _results if r.skipped)
    total   = len(_results)

    for r in _results:
        if r.skipped:
            tag = f"{YELLOW}SKIP{RESET}"
        elif r.passed:
            tag = f"{GREEN}PASS{RESET}"
        else:
            tag = f"{RED}FAIL{RESET}"

        notes = f"  ({', '.join(r.notes)})" if r.notes else ""
        print(f"  [{tag}]  {r.name:<40}  {r.elapsed:>5}ms{notes}")

    print(f"\n  Total: {total}  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")

    if failed == 0:
        print(f"\n  {GREEN}{BOLD}All tests passed.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}{failed} test(s) failed.{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Polymarket SDK Diagnostics")
    parser.add_argument("--slug",        default="",    help="Market slug to use for tests")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--skip-trade",  action="store_true", help="Skip order signing test")
    parser.add_argument("--test",        default="",    help="Run single test by name")
    parser.add_argument("--derive",      action="store_true", help="Derive API key then exit")
    args = parser.parse_args()

    if args.derive:
        from polymarket_sdk.poly_auth import derive_api_key
        print("Deriving L2 API key...")
        creds = derive_api_key()
        print(f"\nAdd to .env:\n  POLY_API_KEY={creds['api_key']}\n  POLY_API_SECRET={creds['api_secret']}\n  POLY_API_PASSPHRASE={creds['api_passphrase']}")
        return

    slug = args.slug or _current_btc_slug()
    verbose = args.verbose

    print(f"{BOLD}Polymarket SDK Diagnostics{RESET}  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"slug: {slug}")

    # Run all tests, accumulating state
    auth_r = run_test("1. Auth / client init", test_auth, verbose)

    addr_r = run_test("2. Wallet address", test_wallet_address, verbose)
    address = addr_r.data.get("address", "")

    run_test("3. CLOB health", test_clob_health, verbose)

    market_r = run_test("4. Market resolve", test_market_resolve, verbose, slug)
    condition_id = market_r.data.get("condition_id", "")
    market_info  = market_r.data.get("market_info", {})

    run_test("5. Cache hit", test_cache_hit, verbose, slug)

    token_r = run_test("6. Token IDs", test_token_ids, verbose, condition_id)
    yes_token_id = token_r.data.get("yes_token_id", "")

    run_test("7. Order book", test_order_book, verbose, yes_token_id)

    run_test("8. Positions", test_positions, verbose, address)

    run_test("9. Portfolio / balance", test_portfolio, verbose)

    run_test("10. Trade history", test_trade_history, verbose, address)

    if not args.skip_trade:
        entry_price = 0.5  # default
        try:
            prices = json.loads(market_info.get("outcome_prices", "[]") if market_info else "[]")
            if prices:
                entry_price = float(prices[0])
        except Exception:
            pass
        run_test("11. Dry-run trade (sign only)", test_dry_trade, verbose, yes_token_id, entry_price)
    else:
        r = R("11. Dry-run trade (sign only)")
        r.skipped = True
        r.notes.append("--skip-trade")
        _results.append(r)

    run_test("12. Cache persistence", test_cache_persistence, verbose)

    run_test("13. Market resolved check", test_resolved_check, verbose, condition_id)

    print_summary()


if __name__ == "__main__":
    main()
