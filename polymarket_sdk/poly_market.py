#!/usr/bin/env python3
"""
poly_market.py — Component 2: Market Resolver

Replaces: client.import_market(url) → market_id

Resolves a Polymarket slug (e.g. "btc-updown-5m-1748000000") to:
  - condition_id   : CTF hex identifier (used as the "market_id" throughout the SDK)
  - yes_token_id   : CLOB token ID for the YES outcome
  - no_token_id    : CLOB token ID for the NO outcome

Two-step resolution:
  1. Gamma API  → events?slug=...  → conditionId
  2. CLOB API   → /markets/{condition_id} → token_ids

No daily import quota. Resolve as often as needed.

Cache structure (market_id_cache.json):
  {
    "btc-updown-5m-1748000000": {
      "condition_id":  "0x...",
      "yes_token_id":  "123...",
      "no_token_id":   "456...",
      "question":      "Will BTC go up or down?",
      "end_time":      "2024-01-01T00:05:00+00:00"
    },
    ...
  }
"""

import json
import os
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

# Cache file — same path pattern as fast_trader.py uses
_DATA_DIR   = "/data" if os.path.isdir("/data") else str(Path(__file__).parent.parent / "trader")
_CACHE_FILE = Path(_DATA_DIR) / "market_id_cache.json"

_cache: dict = {}
_condition_cache: dict = {}   # condition_id → {yes_token_id, no_token_id}


def _load_cache() -> dict:
    global _cache, _condition_cache
    if _CACHE_FILE.exists():
        try:
            raw = json.loads(_CACHE_FILE.read_text())
            _cache = raw
            # Rebuild condition_cache index
            for slug, info in raw.items():
                if isinstance(info, dict) and "condition_id" in info:
                    _condition_cache[info["condition_id"]] = info
            return raw
        except Exception:
            pass
    _cache = {}
    return {}


def _save_cache():
    try:
        _CACHE_FILE.write_text(json.dumps(_cache, indent=2))
    except Exception:
        pass


def _http_get(url: str, timeout: int = 15) -> dict | list | None:
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


def resolve_market(slug: str) -> tuple[str | None, str | None]:
    """
    Resolve a slug to (condition_id, None) or (None, error_str).

    The condition_id is used as the "market_id" throughout the trading system.
    Full market info (including token_ids) is stored in the cache.

    Args:
        slug: Polymarket event slug, e.g. "btc-updown-5m-1748000000"

    Returns:
        (condition_id, None) on success
        (None, error_string) on failure
    """
    global _cache, _condition_cache

    if not _cache:
        _load_cache()

    # Cache hit — return condition_id immediately
    if slug in _cache:
        info = _cache[slug]
        if isinstance(info, dict) and info.get("condition_id"):
            return info["condition_id"], None
        elif isinstance(info, str):
            # Legacy format: value was bare market_id string — keep returning it
            return info, None

    # Step 1: Gamma API → get conditionId
    gamma_url = f"{GAMMA_API}/events?slug={slug}"
    result = _http_get(gamma_url)

    if not result or isinstance(result, dict) or len(result) == 0:
        return None, f"Gamma: no event found for slug '{slug}'"

    event = result[0]
    markets = event.get("markets") or []
    if not markets:
        return None, f"Gamma: event has no markets for slug '{slug}'"

    # Find the active market (not closed, accepting orders)
    market = None
    for m in markets:
        if not m.get("closed", False):
            market = m
            break
    if not market:
        market = markets[0]

    condition_id = market.get("conditionId", "") or market.get("condition_id", "")
    if not condition_id:
        return None, f"Gamma: no conditionId in market data for slug '{slug}'"

    # Step 2: CLOB API → get token_ids
    clob_url = f"{CLOB_API}/markets/{condition_id}"
    clob_data = _http_get(clob_url)

    yes_token_id = ""
    no_token_id  = ""

    if clob_data and not clob_data.get("error"):
        tokens = clob_data.get("tokens") or []
        for tok in tokens:
            outcome = (tok.get("outcome") or "").lower()
            if outcome == "yes":
                yes_token_id = tok.get("token_id", "")
            elif outcome == "no":
                no_token_id = tok.get("token_id", "")

        # Fallback: if outcomes not labelled, use position (0=YES, 1=NO)
        if not yes_token_id and len(tokens) >= 2:
            yes_token_id = tokens[0].get("token_id", "")
            no_token_id  = tokens[1].get("token_id", "")

    # Build cache entry
    end_date = market.get("endDate", "")
    question  = market.get("question") or event.get("title") or ""

    info = {
        "condition_id":  condition_id,
        "yes_token_id":  yes_token_id,
        "no_token_id":   no_token_id,
        "question":      question,
        "end_time":      end_date,
        "slug":          slug,
    }

    _cache[slug]                    = info
    _condition_cache[condition_id]  = info
    _save_cache()

    return condition_id, None


def get_token_id(market_id: str, side: str) -> str | None:
    """
    Return the CLOB token_id for a given condition_id and side.

    Args:
        market_id: condition_id (returned by resolve_market)
        side: "yes" or "no"

    Returns:
        token_id string, or None if not found
    """
    if not _cache:
        _load_cache()

    info = _condition_cache.get(market_id)
    if not info:
        # Try searching through slug cache
        for v in _cache.values():
            if isinstance(v, dict) and v.get("condition_id") == market_id:
                info = v
                _condition_cache[market_id] = info
                break

    if not info:
        return None

    if side.lower() == "yes":
        return info.get("yes_token_id") or None
    else:
        return info.get("no_token_id") or None


def get_market_info(market_id: str) -> dict | None:
    """Return cached full market info dict for a condition_id."""
    if not _cache:
        _load_cache()
    info = _condition_cache.get(market_id)
    if not info:
        for v in _cache.values():
            if isinstance(v, dict) and v.get("condition_id") == market_id:
                return v
    return info


def warm_cache(slugs: list[str]) -> None:
    """Pre-resolve a list of slugs, populating the cache."""
    if not _cache:
        _load_cache()
    for slug in slugs:
        if slug in _cache and isinstance(_cache[slug], dict) and _cache[slug].get("condition_id"):
            continue
        resolve_market(slug)


# ---------------------------------------------------------------------------
# CLI: python poly_market.py <slug>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python poly_market.py <slug>")
        sys.exit(1)

    slug = sys.argv[1]
    print(f"Resolving slug: {slug}")
    cid, err = resolve_market(slug)
    if err:
        print(f"  ERROR: {err}")
        sys.exit(1)

    info = get_market_info(cid)
    print(f"  condition_id : {cid}")
    print(f"  yes_token_id : {info.get('yes_token_id')}")
    print(f"  no_token_id  : {info.get('no_token_id')}")
    print(f"  question     : {info.get('question')}")
    print(f"  end_time     : {info.get('end_time')}")
