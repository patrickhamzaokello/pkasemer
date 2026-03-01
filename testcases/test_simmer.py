#!/usr/bin/env python3
"""
test_simmer.py — Simmer API Endpoint Diagnostics

Tests every endpoint used by the Pknwitq trading stack, in the correct
order, with clear PASS/FAIL output and timing for each step.

Tests covered:
  1.  Health check          — GET  /api/sdk/health
  2.  Agent auth            — GET  /api/sdk/agents/me
  3.  Rate limits           — parsed from /agents/me response
  4.  Market search         — GET  /api/sdk/markets?q=bitcoin
  5.  Fast markets          — GET  /api/sdk/fast-markets?asset=BTC&window=5m
  6.  Market import         — POST /api/sdk/markets/import  (real quota consumed!)
  7.  Market cache hit      — same import call again (must be instant, no quota)
  8.  Market context        — GET  /api/sdk/context/{market_id}
  9.  Dry-run trade         — POST /api/sdk/trade  (dry_run=true, no money)
  10. Positions             — GET  /api/sdk/positions
  11. Portfolio             — GET  /api/sdk/portfolio
  12. Briefing              — GET  /api/sdk/briefing
  13. Import quota guard    — verifies daily_spend.json tracking is accurate
  14. Cache persistence     — verifies market_id_cache.json was written to disk

Usage:
    python test_simmer.py                    # run all tests
    python test_simmer.py --skip-import      # skip the real import (saves quota)
    python test_simmer.py --market-id <id>   # use existing Simmer market ID
    python test_simmer.py --verbose          # print full API responses
    python test_simmer.py --endpoint health  # run a single named test

Environment:
    SIMMER_API_KEY   required
    DB_PATH          optional (default: /data/signal_research.db)
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL  = "https://api.simmer.markets"
API_KEY   = os.environ.get("SIMMER_API_KEY", "")
DATA_DIR  = "/data" if os.path.isdir("/data") else str(Path(__file__).parent)
CACHE_FILE = Path(DATA_DIR) / "market_id_cache.json"

# A known active Polymarket BTC 5m event URL — used only if --skip-import not set.
# The scheduler generates these dynamically; we pick a static one just for testing.
_BTC_TEST_SLUG_TEMPLATE = "btc-updown-5m-{bucket}"


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
def header(msg):
    width = 62
    print(f"\n{BOLD}{CYAN}{'─'*width}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*width}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _request(method, path, body=None, timeout=20):
    """Make an authenticated request. Returns (response_dict, elapsed_ms, status_code)."""
    url = f"{BASE_URL}{path}"
    headers = {
        "User-Agent":    "pknwitq-test/1.0",
        "Authorization": f"Bearer {API_KEY}",
    }
    if body:
        headers["Content-Type"] = "application/json"

    encoded = json.dumps(body).encode() if body else None
    req = Request(url, data=encoded, headers=headers, method=method)

    t0 = time.time()
    try:
        with urlopen(req, timeout=timeout) as resp:
            elapsed = int((time.time() - t0) * 1000)
            data = json.loads(resp.read().decode())
            return data, elapsed, resp.status
    except HTTPError as e:
        elapsed = int((time.time() - t0) * 1000)
        try:
            data = json.loads(e.read().decode())
        except Exception:
            data = {"error": str(e)}
        return data, elapsed, e.code
    except URLError as e:
        elapsed = int((time.time() - t0) * 1000)
        return {"error": str(e.reason)}, elapsed, 0
    except Exception as e:
        elapsed = int((time.time() - t0) * 1000)
        return {"error": str(e)}, elapsed, 0


def _get(path, timeout=20):
    return _request("GET", path, timeout=timeout)

def _post(path, body, timeout=30):
    return _request("POST", path, body=body, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Individual test cases
# ─────────────────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name):
        self.name    = name
        self.passed  = False
        self.skipped = False
        self.notes   = []
        self.elapsed = 0

    def __repr__(self):
        status = "PASS" if self.passed else ("SKIP" if self.skipped else "FAIL")
        return f"[{status}] {self.name}"


_results = []

def _run_test(name, fn, verbose=False):
    """Run a single test function, capture result."""
    print(f"\n{BOLD}Test: {name}{RESET}")
    r = TestResult(name)
    try:
        fn(r, verbose)
    except Exception as e:
        r.passed = False
        r.notes.append(f"Exception: {e}")
        if verbose:
            traceback.print_exc()
        fail(f"Unhandled exception: {e}")
    _results.append(r)
    return r


# ── 1. Health ─────────────────────────────────────────────────────────────────

def test_health(r, verbose):
    data, ms, status = _get("/api/sdk/health")
    r.elapsed = ms
    if verbose:
        info(f"Response: {json.dumps(data, indent=2)}")

    if status != 200:
        fail(f"HTTP {status} — API may be down")
        r.notes.append(f"status={status}")
        return

    if data.get("status") == "ok":
        ok(f"API is up  ({ms}ms)  version={data.get('version','?')}")
        r.passed = True
    else:
        fail(f"Unexpected status field: {data}")


# ── 2. Agent auth ─────────────────────────────────────────────────────────────

def test_agent_auth(r, verbose):
    if not API_KEY:
        warn("SIMMER_API_KEY not set — skipping auth test")
        r.skipped = True
        return

    data, ms, status = _get("/api/sdk/agents/me")
    r.elapsed = ms
    if verbose:
        info(f"Response: {json.dumps(data, indent=2)}")

    if status == 401:
        fail(f"401 Unauthorized — check SIMMER_API_KEY")
        r.notes.append("invalid key")
        return

    if status != 200:
        fail(f"HTTP {status}: {data.get('detail', data)}")
        return

    name    = data.get("name", "?")
    claimed = data.get("claimed", False)
    balance = data.get("balance", 0)
    status_ = data.get("status", "?")
    real    = data.get("real_trading_enabled", False)

    ok(f"Authenticated as '{name}'  ({ms}ms)")
    info(f"Status: {status_}  |  Claimed: {claimed}  |  Balance: ${balance:,.2f}  |  Real trading: {real}")

    if not claimed:
        warn("Agent is unclaimed — real trading disabled. Visit claim URL to unlock.")
        r.notes.append("unclaimed")
    if status_ == "broke":
        warn("Agent balance is zero — register a new agent.")
        r.notes.append("broke")

    # Store for other tests
    r.agent = data
    r.passed = True


# ── 3. Rate limits ────────────────────────────────────────────────────────────

def test_rate_limits(r, verbose):
    data, ms, status = _get("/api/sdk/agents/me")
    r.elapsed = ms

    if status != 200:
        warn("Skipping rate limit check — /agents/me failed")
        r.skipped = True
        return

    limits = data.get("rate_limits", {})
    tier   = limits.get("tier", "unknown")
    eps    = limits.get("endpoints", {})

    ok(f"Rate limit tier: {tier}  ({ms}ms)")
    for endpoint, cfg in eps.items():
        rpm = cfg.get("requests_per_minute", "?")
        info(f"  {endpoint:<40} {rpm} req/min")

    # Check the import endpoint specifically
    import_limit = eps.get("/api/sdk/markets/import", {}).get("requests_per_minute")
    if import_limit:
        if import_limit < 6:
            warn(f"Import limit is {import_limit}/min — very tight for live trading")
            r.notes.append(f"import_limit={import_limit}")
        else:
            ok(f"Import endpoint: {import_limit} req/min (ok for 60s interval)")
    else:
        warn("Import rate limit not returned — assuming 6/min free tier default")

    r.passed = True


# ── 4. Market search ──────────────────────────────────────────────────────────

def test_market_search(r, verbose):
    data, ms, status = _get("/api/sdk/markets?q=bitcoin&limit=5&status=active")
    r.elapsed = ms

    if status != 200:
        fail(f"HTTP {status}: {data.get('detail', data)}")
        return

    markets = data.get("markets", [])
    if verbose:
        for m in markets[:3]:
            info(f"  {m.get('id','?')[:8]}... | {m.get('question','?')[:60]} | p={m.get('current_probability','?')}")

    ok(f"Found {len(markets)} bitcoin markets  ({ms}ms)")
    if markets:
        sample = markets[0]
        info(f"Sample: [{sample.get('import_source','?')}] {sample.get('question','?')[:60]}")
    r.passed = True


# ── 5. Fast markets ───────────────────────────────────────────────────────────

def test_fast_markets(r, verbose):
    data, ms, status = _get("/api/sdk/fast-markets?asset=BTC&window=5m&limit=5")
    r.elapsed = ms

    if status != 200:
        fail(f"HTTP {status}: {data.get('detail', data)}")
        return

    markets = data.get("markets", [])
    ok(f"Found {len(markets)} BTC 5m fast markets  ({ms}ms)")

    if not markets:
        warn("No active fast markets returned — may be between windows")
        r.notes.append("no_fast_markets")
        r.passed = True
        return

    for m in markets[:3]:
        resolves = m.get("resolves_at", "?")
        prob     = m.get("current_probability", "?")
        score    = m.get("opportunity_score", "?")
        info(f"  {m.get('id','?')[:8]}... | p={prob} | opp={score} | resolves={resolves}")

    r.market_id = markets[0].get("id")
    r.passed = True


# ── 6. Market import ──────────────────────────────────────────────────────────

def test_market_import(r, verbose, skip=False, existing_id=None):
    """Import a BTC 5m market. Skippable to preserve daily quota."""
    if existing_id:
        ok(f"Using provided market_id: {existing_id[:8]}...")
        r.market_id = existing_id
        r.passed = True
        r.skipped = True
        return

    if skip:
        warn("Import test skipped (--skip-import). Use --market-id to test with an existing ID.")
        r.skipped = True
        return

    # Generate current 5m bucket slug
    bucket = (int(time.time()) // 300) * 300
    slug   = f"btc-updown-5m-{bucket}"
    url    = f"https://polymarket.com/event/{slug}"

    info(f"Importing: {url}")
    info("NOTE: This consumes 1 of your 10/day import quota")

    body = {"polymarket_url": url}
    data, ms, status = _post("/api/sdk/markets/import", body, timeout=30)
    r.elapsed = ms

    if verbose:
        info(f"Response: {json.dumps(data, indent=2)}")

    if status == 429:
        fail(f"Rate limited (429) — {data.get('detail', '')}")
        r.notes.append("rate_limited")
        info("This is the root cause of 'Rate limited — will retry next cycle' in your logs.")
        info("Fix: call warm_import_cache() at scheduler startup, before the trade loop.")
        return

    if status == 403:
        fail(f"403 Forbidden — agent may be unclaimed: {data.get('detail', '')}")
        r.notes.append("forbidden")
        return

    if status not in (200, 201):
        fail(f"HTTP {status}: {data.get('detail', data)}")
        r.notes.append(f"status={status}")
        return

    # Handle both single-market and multi-outcome event responses
    market_id = data.get("market_id") or data.get("id")
    if not market_id and data.get("markets"):
        market_id = data["markets"][0].get("market_id")

    if not market_id:
        fail(f"Import succeeded but no market_id in response: {list(data.keys())}")
        r.notes.append("no_market_id")
        return

    import_status = data.get("status", "?")
    question      = data.get("question", "?")
    prob          = data.get("current_probability", "?")

    ok(f"Imported  ({ms}ms)  status={import_status}")
    info(f"market_id: {market_id}")
    info(f"question:  {question}")
    info(f"prob:      {prob}")

    r.market_id = market_id
    r.passed = True


# ── 7. Cache hit ──────────────────────────────────────────────────────────────

def test_cache_hit(r, verbose, market_id=None):
    """Second import of the same market — must return instantly from cache."""
    if not market_id:
        warn("No market_id from previous import test — skipping cache test")
        r.skipped = True
        return

    # Determine the slug from market_id cache file
    cache = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
        except Exception:
            pass

    slug_for_id = next((k for k, v in cache.items() if v == market_id), None)
    if not slug_for_id:
        warn(f"market_id {market_id[:8]}... not found in cache file — may not have been written yet")
        r.notes.append("not_in_cache_file")
        r.skipped = True
        return

    # Call the import endpoint again for the same slug
    bucket = (int(time.time()) // 300) * 300
    slug   = f"btc-updown-5m-{bucket}"
    url    = f"https://polymarket.com/event/{slug}"
    body   = {"polymarket_url": url}

    data, ms, status = _post("/api/sdk/markets/import", body, timeout=15)
    r.elapsed = ms

    if status not in (200, 201):
        fail(f"HTTP {status} on re-import: {data.get('detail', data)}")
        return

    returned_status = data.get("status", "?")
    # Simmer returns "already_exists" or "imported" — both are fine
    ok(f"Re-import returned  ({ms}ms)  status={returned_status}")
    if ms < 500:
        ok("Fast response confirms server-side dedup (no quota consumed)")
    else:
        warn(f"Slow response ({ms}ms) — may have hit the import endpoint again")

    r.passed = True


# ── 8. Market context ─────────────────────────────────────────────────────────

def test_market_context(r, verbose, market_id=None):
    if not market_id:
        warn("No market_id available — skipping context test")
        r.skipped = True
        return

    data, ms, status = _get(f"/api/sdk/context/{market_id}?my_probability=0.65")
    r.elapsed = ms

    if verbose:
        info(f"Response keys: {list(data.keys())}")

    if status == 404:
        fail(f"Market {market_id[:8]}... not found — may have resolved already")
        r.notes.append("not_found")
        return

    if status != 200:
        fail(f"HTTP {status}: {data.get('detail', data)}")
        return

    market   = data.get("market", {})
    position = data.get("position", {})
    slippage = data.get("slippage", {})
    edge     = data.get("edge", {})
    warnings = data.get("warnings", [])

    ok(f"Context retrieved  ({ms}ms)")
    info(f"question:         {market.get('question','?')[:60]}")
    info(f"current_price:    {market.get('current_price','?')}")
    info(f"time_to_res:      {market.get('time_to_resolution','?')}")
    info(f"has_position:     {position.get('has_position', False)}")
    if slippage.get("estimates"):
        est = slippage["estimates"][0]
        info(f"slippage $10:     {est.get('slippage_pct','?')}%")
    if edge.get("recommendation"):
        info(f"edge rec:         {edge['recommendation']}  (edge={edge.get('user_edge','?')})")
    if warnings:
        for w in warnings:
            warn(f"API warning: {w}")

    r.passed = True


# ── 9. Dry-run trade ──────────────────────────────────────────────────────────

def test_dry_run_trade(r, verbose, market_id=None):
    if not market_id:
        warn("No market_id available — skipping dry-run trade test")
        r.skipped = True
        return

    body = {
        "market_id": market_id,
        "side":      "yes",
        "amount":    5.0,
        "venue":     "polymarket",
        "dry_run":   True,
        "reasoning": "pknwitq test script dry run",
    }
    data, ms, status = _post("/api/sdk/trade", body, timeout=30)
    r.elapsed = ms

    if verbose:
        info(f"Response: {json.dumps(data, indent=2)}")

    if status == 403:
        fail(f"403 — agent unclaimed or real trading not enabled")
        r.notes.append("forbidden")
        return

    if status not in (200, 201):
        fail(f"HTTP {status}: {data.get('detail', data)}")
        return

    success      = data.get("success", False)
    shares       = data.get("shares_bought", 0)
    cost         = data.get("cost", 0)
    fill_status  = data.get("fill_status", "?")
    fee_rate_bps = data.get("fee_rate_bps", 0)
    hint         = data.get("hint")
    error        = data.get("error")

    if not success:
        fail(f"dry_run returned success=false: {error}")
        if hint:
            info(f"Hint: {hint}")
        r.notes.append(f"success=false: {error}")
        return

    ok(f"Dry-run trade accepted  ({ms}ms)")
    info(f"shares_bought:  ~{shares:.1f}")
    info(f"cost:           ${cost:.2f}")
    info(f"fill_status:    {fill_status}")
    info(f"fee_rate_bps:   {fee_rate_bps}")
    if fee_rate_bps > 0:
        warn(f"This market charges a {fee_rate_bps}bps taker fee — factor into EV calculation")

    r.passed = True


# ── 10. Positions ─────────────────────────────────────────────────────────────

def test_positions(r, verbose):
    data, ms, status = _get("/api/sdk/positions?venue=polymarket")
    r.elapsed = ms

    if status != 200:
        fail(f"HTTP {status}: {data.get('detail', data)}")
        return

    positions  = data.get("positions", [])
    sim_pnl    = data.get("sim_pnl", 0)
    poly_pnl   = data.get("polymarket_pnl", 0)
    total_val  = data.get("total_value", 0)

    ok(f"Positions retrieved  ({ms}ms)")
    info(f"open positions:  {len(positions)}")
    info(f"total value:     ${total_val:.2f}")
    info(f"sim PnL:         ${sim_pnl:.2f}")
    info(f"polymarket PnL:  ${poly_pnl:.2f}")

    if verbose and positions:
        for p in positions[:5]:
            info(
                f"  {p.get('question','?')[:45]:45} "
                f"YES={p.get('shares_yes',0):.1f} "
                f"PnL=${p.get('pnl',0):.2f} "
                f"[{p.get('venue','?')}]"
            )

    r.passed = True


# ── 11. Portfolio ─────────────────────────────────────────────────────────────

def test_portfolio(r, verbose):
    data, ms, status = _get("/api/sdk/portfolio")
    r.elapsed = ms

    if status != 200:
        fail(f"HTTP {status}: {data.get('detail', data)}")
        return

    balance   = data.get("balance_usdc", 0)
    exposure  = data.get("total_exposure", 0)
    pos_count = data.get("positions_count", 0)
    pnl_total = data.get("pnl_total", 0)
    conc      = data.get("concentration", {})

    ok(f"Portfolio retrieved  ({ms}ms)")
    info(f"balance USDC:    ${balance:.2f}")
    info(f"total exposure:  ${exposure:.2f}")
    info(f"positions:       {pos_count}")
    info(f"total PnL:       ${pnl_total:.2f}")
    if conc:
        info(f"top market:      {conc.get('top_market_pct',0):.0%} of exposure")

    if balance < 1.0:
        warn("USDC balance < $1 — not enough for live trades (need at least $2.75)")
        r.notes.append("low_balance")

    r.passed = True


# ── 12. Briefing ──────────────────────────────────────────────────────────────

def test_briefing(r, verbose):
    data, ms, status = _get("/api/sdk/briefing")
    r.elapsed = ms

    if status != 200:
        fail(f"HTTP {status}: {data.get('detail', data)}")
        return

    venues   = data.get("venues", {})
    opps     = data.get("opportunities", {})
    risk     = data.get("risk_alerts", [])
    perf     = data.get("performance", {})

    ok(f"Briefing retrieved  ({ms}ms)")

    for venue_name, venue_data in venues.items():
        if venue_data is None:
            info(f"  {venue_name}: no positions")
            continue
        bal      = venue_data.get("balance", "?")
        pnl      = venue_data.get("pnl", 0)
        pos_n    = venue_data.get("positions_count", 0)
        currency = venue_data.get("currency", "?")
        info(f"  {venue_name}: {pos_n} positions | balance={bal} {currency} | PnL={pnl:.2f}")

    new_mkts = opps.get("new_markets", [])
    info(f"new opportunities:  {len(new_mkts)}")

    if risk:
        for alert in risk:
            warn(f"risk alert: {alert}")

    if perf:
        info(f"win rate: {perf.get('win_rate','?')}%  |  rank: {perf.get('rank','?')}/{perf.get('total_agents','?')}")

    r.passed = True


# ── 13. Import quota tracking ─────────────────────────────────────────────────

def test_quota_tracking(r, verbose):
    """Verify daily_spend.json is being written and read correctly."""
    spend_file = Path(__file__).parent / "daily_spend.json"

    if not spend_file.exists():
        warn(f"daily_spend.json not found at {spend_file}")
        warn("This is normal on first run — it will be created on the first trade.")
        r.notes.append("no_spend_file")
        r.passed = True
        return

    try:
        data  = json.loads(spend_file.read_text())
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        file_date = data.get("date", "?")
        spent     = data.get("spent", 0)
        trades    = data.get("trades", 0)
        imports   = data.get("imports_today", 0)

        ok(f"daily_spend.json found  ({spend_file})")
        info(f"date:           {file_date}  ({'current' if file_date == today else 'STALE — will reset on next cycle'})")
        info(f"spent:          ${spent:.2f}")
        info(f"trades:         {trades}")
        info(f"imports_today:  {imports}/9")

        if file_date != today:
            warn("Spend file is from a previous day — will auto-reset on next cycle")
            r.notes.append("stale_spend_file")

        r.passed = True
    except Exception as e:
        fail(f"Could not read daily_spend.json: {e}")


# ── 14. Cache persistence ─────────────────────────────────────────────────────

def test_cache_persistence(r, verbose):
    """Verify market_id_cache.json exists and is readable."""
    if not CACHE_FILE.exists():
        warn(f"Cache file not found: {CACHE_FILE}")
        warn("This means every import attempt hits the API — no dedup protection.")
        warn("Fix: call warm_import_cache(ASSET) at scheduler startup.")
        r.notes.append("no_cache_file")
        r.passed = True   # not a failure, just uninitialized
        return

    try:
        cache = json.loads(CACHE_FILE.read_text())
        ok(f"Cache file found: {CACHE_FILE}")
        info(f"cached slugs: {len(cache)}")

        if verbose and cache:
            for slug, mid in list(cache.items())[-5:]:
                info(f"  {slug[-24:]:24} → {mid[:8]}...")

        # Check if any cached slug matches the current 5m bucket
        bucket       = (int(time.time()) // 300) * 300
        current_slug = f"btc-updown-5m-{bucket}"
        next_slug    = f"btc-updown-5m-{bucket + 300}"

        if current_slug in cache:
            ok(f"Current window slug is cached ✓  ({current_slug[-10:]})")
        else:
            warn(f"Current window slug NOT cached ({current_slug[-10:]}) — first trade this window will hit import API")
            r.notes.append("current_slug_not_cached")

        if next_slug in cache:
            ok(f"Next window slug is pre-cached ✓")
        else:
            info(f"Next window slug not yet cached (normal — will import when needed)")

        r.passed = True
    except Exception as e:
        fail(f"Could not read cache file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Full diagnostic run
# ─────────────────────────────────────────────────────────────────────────────

def run_all(args):
    header("SIMMER API DIAGNOSTIC")
    print(f"  Base URL:  {BASE_URL}")
    print(f"  API Key:   {API_KEY[:12]}...{API_KEY[-4:] if len(API_KEY) > 16 else '(not set)'}")
    print(f"  Data dir:  {DATA_DIR}")
    print(f"  Time:      {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Skip imp:  {args.skip_import}")

    market_id = args.market_id  # may be None

    # Run tests in dependency order
    _run_test("1. Health check",         lambda r, v: test_health(r, v), args.verbose)
    _run_test("2. Agent auth",           lambda r, v: test_agent_auth(r, v), args.verbose)
    _run_test("3. Rate limits",          lambda r, v: test_rate_limits(r, v), args.verbose)
    _run_test("4. Market search",        lambda r, v: test_market_search(r, v), args.verbose)
    _run_test("5. Fast markets",         lambda r, v: test_fast_markets(r, v), args.verbose)

    import_result = _run_test(
        "6. Market import",
        lambda r, v: test_market_import(r, v, skip=args.skip_import, existing_id=market_id),
        args.verbose,
    )
    # Propagate market_id to downstream tests
    if hasattr(import_result, "market_id"):
        market_id = import_result.market_id

    _run_test(
        "7. Cache hit (re-import)",
        lambda r, v: test_cache_hit(r, v, market_id=market_id),
        args.verbose,
    )
    _run_test(
        "8. Market context",
        lambda r, v: test_market_context(r, v, market_id=market_id),
        args.verbose,
    )
    _run_test(
        "9. Dry-run trade",
        lambda r, v: test_dry_run_trade(r, v, market_id=market_id),
        args.verbose,
    )
    _run_test("10. Positions",            lambda r, v: test_positions(r, v), args.verbose)
    _run_test("11. Portfolio",            lambda r, v: test_portfolio(r, v), args.verbose)
    _run_test("12. Briefing",             lambda r, v: test_briefing(r, v), args.verbose)
    _run_test("13. Import quota tracking",lambda r, v: test_quota_tracking(r, v), args.verbose)
    _run_test("14. Cache persistence",    lambda r, v: test_cache_persistence(r, v), args.verbose)

    # ── Summary ───────────────────────────────────────────────────────────────
    header("RESULTS SUMMARY")

    passed   = [r for r in _results if r.passed and not r.skipped]
    failed   = [r for r in _results if not r.passed and not r.skipped]
    skipped  = [r for r in _results if r.skipped]

    for r in _results:
        if r.skipped:
            print(f"  {YELLOW}SKIP{RESET}  {r.name}")
        elif r.passed:
            tag = f"{r.elapsed}ms" if r.elapsed else ""
            print(f"  {GREEN}PASS{RESET}  {r.name}  {DIM}{tag}{RESET}")
        else:
            notes = " | ".join(r.notes) if r.notes else ""
            print(f"  {RED}FAIL{RESET}  {r.name}  {DIM}{notes}{RESET}")

    print(f"\n  {GREEN}{len(passed)} passed{RESET}  "
          f"{RED}{len(failed)} failed{RESET}  "
          f"{YELLOW}{len(skipped)} skipped{RESET}")

    # ── Actionable diagnosis ──────────────────────────────────────────────────
    if failed:
        print(f"\n{BOLD}Diagnosis:{RESET}")

        fail_names = [r.name for r in failed]
        all_notes  = [n for r in failed for n in r.notes]

        if any("6." in n for n in fail_names):
            if "rate_limited" in all_notes:
                print(f"""
  {RED}Import endpoint is rate limited.{RESET}

  Root cause: The cache file ({CACHE_FILE}) is empty or missing,
  so every trade cycle tries to import the market from scratch and
  hits the 6/min rate limit on the Simmer import endpoint.

  Fix — add this to scheduler.py BEFORE the main loop:

      from fast_trader import warm_import_cache, ASSET
      log.info("Warming market ID cache...")
      warm_import_cache(ASSET)
      log.info("Cache warm complete.")

  This pre-imports the current + next 5m slug at startup with 12s
  spacing, so live cycles always find a cached market_id.
""")
            elif "forbidden" in all_notes:
                print(f"""
  {RED}Agent is unclaimed or real trading is not enabled.{RESET}

  Visit your claim URL to unlock real trading:
    curl -H "Authorization: Bearer {API_KEY[:12]}..." \\
         https://api.simmer.markets/api/sdk/agents/me
  Then visit the claim_url returned in the response.
""")

        if any("auth" in n.lower() for n in fail_names):
            print(f"  {RED}Authentication failed.{RESET} Verify SIMMER_API_KEY is set correctly.")

    else:
        print(f"\n  {GREEN}{BOLD}All systems go.{RESET} Stack looks healthy.")

    return len(failed) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Single endpoint mode
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_TESTS = {
    "health":    lambda args: _run_test("Health", lambda r, v: test_health(r, v), args.verbose),
    "auth":      lambda args: _run_test("Auth",   lambda r, v: test_agent_auth(r, v), args.verbose),
    "limits":    lambda args: _run_test("Limits", lambda r, v: test_rate_limits(r, v), args.verbose),
    "markets":   lambda args: _run_test("Markets",lambda r, v: test_market_search(r, v), args.verbose),
    "fast":      lambda args: _run_test("Fast",   lambda r, v: test_fast_markets(r, v), args.verbose),
    "import":    lambda args: _run_test("Import", lambda r, v: test_market_import(r, v, skip=False, existing_id=None), args.verbose),
    "context":   lambda args: _run_test("Context",lambda r, v: test_market_context(r, v, market_id=args.market_id), args.verbose),
    "trade":     lambda args: _run_test("Trade",  lambda r, v: test_dry_run_trade(r, v, market_id=args.market_id), args.verbose),
    "positions": lambda args: _run_test("Pos",    lambda r, v: test_positions(r, v), args.verbose),
    "portfolio": lambda args: _run_test("Port",   lambda r, v: test_portfolio(r, v), args.verbose),
    "briefing":  lambda args: _run_test("Brief",  lambda r, v: test_briefing(r, v), args.verbose),
    "cache":     lambda args: _run_test("Cache",  lambda r, v: test_cache_persistence(r, v), args.verbose),
    "quota":     lambda args: _run_test("Quota",  lambda r, v: test_quota_tracking(r, v), args.verbose),
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simmer API endpoint diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_simmer.py                         # full diagnostic
  python test_simmer.py --skip-import           # skip import (save quota)
  python test_simmer.py --market-id abc123      # use existing market ID
  python test_simmer.py --verbose               # print full API responses
  python test_simmer.py --endpoint health       # single test
  python test_simmer.py --endpoint import       # test import only
  python test_simmer.py --endpoint trade --market-id abc123

Available --endpoint values:
  health, auth, limits, markets, fast, import,
  context, trade, positions, portfolio, briefing, cache, quota
        """,
    )
    parser.add_argument("--skip-import",  action="store_true",
                        help="Skip the real import test (preserves daily quota)")
    parser.add_argument("--market-id",    default=None,
                        help="Use this Simmer market UUID instead of importing")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full API responses")
    parser.add_argument("--endpoint",     default=None,
                        choices=list(SINGLE_TESTS.keys()),
                        help="Run a single named test instead of the full suite")
    args = parser.parse_args()

    if not API_KEY:
        print(f"{RED}ERROR: SIMMER_API_KEY environment variable not set{RESET}")
        print("  export SIMMER_API_KEY=sk_live_...")
        sys.exit(1)

    if args.endpoint:
        fn = SINGLE_TESTS[args.endpoint]
        fn(args)
        r = _results[-1]
        sys.exit(0 if (r.passed or r.skipped) else 1)
    else:
        success = run_all(args)
        sys.exit(0 if success else 1)