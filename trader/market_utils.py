#!/usr/bin/env python3
"""
market_utils.py — Shared market discovery utilities.

Extracted from fast_trader.py so both fast_trader.py and signal_research.py
can import without creating a circular dependency.
"""

import json
import time
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

GAMMA_API = "https://gamma-api.polymarket.com"

COIN_SLUGS = {
    "BTC": "btc",
    "ETH": "eth",
    "SOL": "sol",
    "XRP": "xrp",
}

# Default minimum seconds remaining before a market is considered tradeable.
# fast_trader.py overrides this via config; signal_research.py uses the default.
DEFAULT_MIN_TIME_REMAINING = 30


def _api_request(url, method="GET", data=None, headers=None, timeout=15):
    try:
        req_headers = headers or {}
        if "User-Agent" not in req_headers:
            req_headers["User-Agent"] = "simmer-fastloop/1.0"
        body = None
        if data:
            body = json.dumps(data).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        req = Request(url, data=body, headers=req_headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        try:
            error_body = json.loads(e.read().decode("utf-8"))
            return {"error": error_body.get("detail", str(e)), "status_code": e.code}
        except Exception:
            return {"error": str(e), "status_code": e.code}
    except URLError as e:
        return {"error": f"Connection error: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def get_fast_market_slugs(asset="BTC", include_next=True):
    """
    Generate Polymarket 5m fast market slugs using Unix timestamp bucketing.
    Format: {coin}-updown-5m-{unix_ts_rounded_to_5min}

    Returns current bucket slug and optionally the next one.
    """
    now = int(time.time())
    current_bucket = (now // 300) * 300
    next_bucket = current_bucket + 300

    coin = COIN_SLUGS.get(asset.upper(), asset.lower())
    slugs = [f"{coin}-updown-5m-{current_bucket}"]
    if include_next:
        slugs.append(f"{coin}-updown-5m-{next_bucket}")
    return slugs


def discover_fast_market_markets(asset="BTC", window="5m"):
    """
    Discover active Polymarket fast markets for a given asset and window.
    Queries the Gamma API for current and next 5m buckets.
    Returns a list of market dicts ready for find_best_fast_market().
    """
    now = datetime.now(timezone.utc)
    markets = []

    for slug in get_fast_market_slugs(asset, include_next=True):
        gamma_url = f"{GAMMA_API}/events?slug={slug}"
        result = _api_request(gamma_url)

        if not result or isinstance(result, dict) or len(result) == 0:
            continue

        event = result[0]

        # eventStartTime lives on the EVENT object, not the market
        event_start = None
        for key in ("startTime", "eventStartTime"):
            raw = event.get(key)
            if raw and "T" in str(raw):
                try:
                    event_start = datetime.fromisoformat(
                        raw.replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                    break
                except ValueError:
                    pass

        # Skip if start is more than 15 min away (not yet tradeable)
        if event_start and (event_start - now).total_seconds() > 900:
            continue

        for m in (event.get("markets") or []):
            if m.get("closed", False):
                continue
            if not m.get("acceptingOrders", True):
                continue

            end_time = None
            raw_end = m.get("endDate")
            if raw_end and "T" in str(raw_end):
                try:
                    end_time = datetime.fromisoformat(
                        raw_end.replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                except ValueError:
                    pass

            if not end_time:
                continue

            # Skip if expired more than 5 min ago
            if (now - end_time).total_seconds() > 300:
                continue

            def _first_not_none(*values):
                for v in values:
                    if v is not None:
                        return v
                return 0

            fee_bps = int(
                _first_not_none(
                    m.get("makerBaseFee"),
                    m.get("feeRateBps"),
                    m.get("fee_rate_bps"),
                )
            )

            markets.append({
                "question":       m.get("question") or event.get("title") or "",
                "slug":           slug,
                "condition_id":   m.get("conditionId", ""),
                "end_time":       end_time,
                "event_start":    event_start,
                "outcomes":       m.get("outcomes", []),
                "outcome_prices": m.get("outcomePrices", "[]"),
                "fee_rate_bps":   fee_bps,
                "lastTradePrice": m.get("lastTradePrice"),
                "bestBid":        m.get("bestBid"),
                "bestAsk":        m.get("bestAsk"),
                "spread":         m.get("spread"),
                "volumeClob":     m.get("volumeClob"),
                "volume24hr":     m.get("volume24hr"),
                "price_to_beat":  (event.get("eventMetadata") or {}).get("priceToBeat"),
            })

    return markets


def find_best_fast_market(markets, min_time_remaining=DEFAULT_MIN_TIME_REMAINING):
    """
    Pick the best market to trade from a list returned by discover_fast_market_markets().

    Priority:
    - Active window (event_start passed) with > 120s left  → highest priority
    - Active window but almost expired (< 120s)            → low priority
    - Pre-window (event_start in future, within 15 min)    → fallback only
    - Never enter with < min_time_remaining seconds left
    """
    now = datetime.now(timezone.utc)
    candidates = []

    for m in markets:
        event_start = m.get("event_start")
        end_time = m.get("end_time")

        if not end_time:
            continue

        remaining = (end_time - now).total_seconds()

        if remaining < min_time_remaining:
            continue

        window_open = True
        if event_start:
            secs_until_start = (event_start - now).total_seconds()
            if secs_until_start > 900:
                continue  # Too far in the future
            if secs_until_start > 0:
                window_open = False

        if window_open and remaining > 120:
            score = remaining + 600
        elif window_open:
            score = remaining * 0.5
        else:
            score = remaining * 0.8

        candidates.append((score, m))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]