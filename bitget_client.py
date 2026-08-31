import base64
import hashlib
import hmac
import json
import logging
import time
from decimal import Decimal, InvalidOperation, ROUND_DOWN

import requests

from config import API_KEY, DRY_RUN, PASSPHRASE, SECRET_KEY, SLIPPAGE_RATE

logger = logging.getLogger(__name__)

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"

_contract_cache: dict = {}
_contract_cache_updated_at = 0.0
_CONTRACT_CACHE_TTL_SECONDS = 300

# Infrastructure / DeFi / AI universe — no meme coins.
# Selected by individual profit factor > 1.4 over 24 months with ADX>=40.
# DOGE (meme) and BNB (PF<1) removed; FET, SUI, ARB, AAVE, DOT, CRV added.
RESEARCH_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",    # Layer 1 + Payments
    "ADAUSDT", "LINKUSDT",                           # Layer 1 + Oracle
    "FETUSDT", "SUIUSDT",                            # AI + Layer 1
    "ARBUSDT", "DOTUSDT",                            # Layer 2 + Layer 1
    "AAVEUSDT", "CRVUSDT",                           # DeFi
]


def _sign(ts: str, method: str, path: str, body: str = "") -> str:
    message = ts + method.upper() + path + body
    mac = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()


def _headers(method: str, path: str, body: str = "") -> dict:
    timestamp = str(int(time.time() * 1000))
    return {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": _sign(timestamp, method, path, body),
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US",
    }


def _get(path: str, params: dict | None = None) -> dict:
    query = ("?" + "&".join(f"{key}={value}" for key, value in params.items())) if params else ""
    signed_path = path + query
    try:
        response = requests.get(
            BASE_URL + signed_path,
            headers=_headers("GET", signed_path),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.error(f"GET {path} failed: {exc}")
        return {}


def _post(path: str, data: dict) -> dict:
    body = json.dumps(data)
    try:
        response = requests.post(
            BASE_URL + path,
            headers=_headers("POST", path, body),
            data=body,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        # A timeout is an ambiguous exchange state. clientOid is logged by callers
        # so the order can be reconciled manually instead of blindly retried.
        logger.error(f"POST {path} failed: {exc}")
        return {}


def get_tickers() -> list:
    response = _get("/api/v2/mix/market/tickers", {"productType": PRODUCT_TYPE})
    if response.get("code") != "00000":
        logger.error(f"get_tickers: {response.get('msg')}")
        return []
    tickers = response.get("data", [])
    tickers.sort(key=lambda item: float(item.get("usdtVolume") or 0), reverse=True)
    return tickers


def get_top_symbols(count: int) -> list:
    """Return only research-universe contracts with verified active metadata."""
    selected = []
    for symbol in RESEARCH_UNIVERSE:
        info = get_contract_info(symbol)
        if (
            info
            and info.get("symbolStatus") == "normal"
            and _contract_constraints(info) is not None
        ):
            selected.append(symbol)
        elif not info:
            logger.warning(f"{symbol}: contract metadata unavailable; entry disabled")
        else:
            logger.warning(f"{symbol}: incomplete/inactive contract metadata; entry disabled")
        if len(selected) >= count:
            break
    return selected



# Meme-Coins und Low-Quality-Tokens: nie handeln, auch nicht dynamisch.
MEME_BLACKLIST = {
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT",
    "FLOKIUSDT", "BRETTUSDT", "MEMEUSDT", "PEOPLEUSDT", "TURBOCOMUSDT",
    "BABYDOGEUSDT", "ELONUSDT", "NEIROUSDT", "POPCATUSDT", "MOGUSDT",
    "TRUMPUSDT", "CATUSDT",
}

# Cache fuer den dynamischen Scanner
_dynamic_cache: list[str] = []
_dynamic_cache_ts: float = 0.0
_DYNAMIC_CACHE_TTL = 3600  # 1 Stunde


def get_dynamic_universe(count: int = 12, min_volume_usdt: float = 50_000_000) -> list[str]:
    """
    Dynamische Universum-Rotation: Base + Top-Volumen gefiltert.

    Gibt die vereinigte Menge aus:
      1. RESEARCH_UNIVERSE (immer dabei, getestet)
      2. Top-N nach 24h-Volumen von Bitget, exklusive Meme-Blacklist

    Die Reihenfolge ist: Base-Coins zuerst, dann Volumen-Rang.
    Der ADX-Filter in strategy.py entscheidet dann, wer wirklich tradet.
    """
    global _dynamic_cache, _dynamic_cache_ts
    import time as _time

    now = _time.time()
    if _dynamic_cache and now - _dynamic_cache_ts < _DYNAMIC_CACHE_TTL:
        return _dynamic_cache[:count]

    # Base-Universum immer dabei
    base = list(RESEARCH_UNIVERSE)

    # Ergaenze mit Top-Volumen-Coins
    try:
        tickers = get_tickers()  # schon nach Volumen sortiert
        for ticker in tickers:
            sym = ticker.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            if sym in MEME_BLACKLIST:
                continue
            vol = float(ticker.get("usdtVolume", 0) or 0)
            if vol < min_volume_usdt:
                continue
            # Prüfe ob der Contract aktiv und handelbar ist
            if sym not in base:
                info = get_contract_info(sym)
                if (info and info.get("symbolStatus") == "normal"
                        and _contract_constraints(info) is not None):
                    base.append(sym)
            if len(base) >= count * 2:  # genug Kandidaten
                break
    except Exception as exc:
        logger.warning(f"Dynamic universe scan failed: {exc}")

    # Deduplizieren, Reihenfolge beibehalten
    seen = set()
    result = []
    for sym in base:
        if sym not in seen:
            seen.add(sym)
            result.append(sym)

    _dynamic_cache = result
    _dynamic_cache_ts = now

    if len(result) > len(RESEARCH_UNIVERSE):
        extras = [s for s in result if s not in RESEARCH_UNIVERSE]
        logger.info(
            f"Dynamic universe: {len(result)} symbols "
            f"(base {len(RESEARCH_UNIVERSE)} + {len(extras)} dynamic: "
            f"{', '.join(extras[:5])}{'...' if len(extras) > 5 else ''})"
        )

    return result[:count]


def get_candles(symbol: str, granularity: str = "1H", limit: int = 300) -> list:
    response = _get("/api/v2/mix/market/candles", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "granularity": granularity,
        "limit": str(limit),
    })
    if response.get("code") != "00000":
        logger.warning(f"get_candles {symbol}: {response.get('msg')}")
        return []
    return response.get("data", [])


def get_ticker(symbol: str) -> dict:
    response = _get("/api/v2/mix/market/ticker", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
    })
    if response.get("code") != "00000":
        return {}
    data = response.get("data")
    return data[0] if isinstance(data, list) and data else data or {}


def get_current_price(symbol: str) -> float:
    ticker = get_ticker(symbol)
    return float(ticker.get("lastPr") or ticker.get("last") or 0)


_oi_cache: dict[str, tuple[float, float]] = {}


def get_open_interest(symbol: str) -> float:
    response = _get("/api/v2/mix/market/open-interest", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
    })
    if response.get("code") != "00000":
        return 0.0
    data = response.get("data", {})
    entries = data.get("openInterestList", []) if isinstance(data, dict) else []
    return float(entries[0].get("size", 0) or 0) if entries else 0.0


def get_oi_score(symbol: str) -> float:
    now = time.time()
    open_interest = get_open_interest(symbol)
    if open_interest == 0:
        return 0.0

    ticker = get_ticker(symbol)
    price_change = float(ticker.get("change24h") or ticker.get("changeUtc24h") or 0)
    score = 0.0
    if symbol in _oi_cache:
        previous_time, previous_oi = _oi_cache[symbol]
        age = now - previous_time
        if 30 <= age <= 1800 and previous_oi > 0:
            oi_change = (open_interest - previous_oi) / previous_oi * 100
            if oi_change > 0.3 and price_change > 0:
                score = min(50.0, oi_change * 8)
            elif oi_change > 0.3 and price_change < 0:
                score = max(-50.0, -oi_change * 6)
            elif oi_change < -0.3 and price_change > 0:
                score = 15.0
            elif oi_change < -0.3 and price_change < 0:
                score = max(-30.0, oi_change * 4)
            logger.debug(
                f"OI {symbol}: {previous_oi:.2f}->{open_interest:.2f} contracts "
                f"({oi_change:+.2f}%), price24h={price_change:+.3f}, score={score:+.1f}"
            )
    _oi_cache[symbol] = (now, open_interest)
    return round(score, 1)


def get_funding_rate(symbol: str) -> float:
    response = _get("/api/v2/mix/market/current-fund-rate", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
    })
    if response.get("code") == "00000":
        data = response.get("data")
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict):
            return float(data.get("fundingRate") or 0)
    return 0.0


FUNDING_BUSINESS_TYPES = frozenset({
    "contract_settle_fee",
    "funding_fee",
    "funding",
})


def _finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _error_result(message: str, **extra) -> dict:
    result = {"ok": False, "items": [], "error": message}
    result.update(extra)
    return result


def get_funding_schedule(symbol: str) -> dict:
    """Return the current public funding schedule for coverage validation."""
    response = _get("/api/v2/mix/market/funding-time", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
    })
    if response.get("code") != "00000":
        return {
            "ok": False,
            "error": f"funding-time failed: {response.get('msg') or 'invalid response'}",
        }
    data = response.get("data")
    rows = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    row = next(
        (item for item in rows if isinstance(item, dict)
         and (item.get("symbol") or symbol) == symbol),
        None,
    )
    if row is None:
        return {"ok": False, "error": "funding-time returned no matching symbol"}
    try:
        next_funding_time = int(row["nextFundingTime"])
        interval_hours = float(row["ratePeriod"])
        request_time = int(response["requestTime"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "funding-time returned invalid schedule fields"}
    if interval_hours <= 0 or interval_hours > 24 or next_funding_time <= request_time:
        return {"ok": False, "error": "funding-time returned an implausible schedule"}
    return {
        "ok": True,
        "symbol": symbol,
        "next_funding_time": next_funding_time,
        "interval_ms": int(interval_hours * 60 * 60 * 1000),
        "request_time": request_time,
        "error": None,
    }


def get_historical_funding_rates(symbol: str, start_time_ms: int | None = None,
                                 end_time_ms: int | None = None,
                                 page_size: int = 100,
                                 max_pages: int = 10) -> dict:
    """Fetch bounded public funding pages and report proven history coverage."""
    page_size = min(100, max(1, int(page_size)))
    max_pages = min(100, max(1, int(max_pages)))
    start_time_ms = int(start_time_ms) if start_time_ms is not None else None
    end_time_ms = int(end_time_ms) if end_time_ms is not None else None
    if start_time_ms is not None and end_time_ms is not None and start_time_ms > end_time_ms:
        return _error_result("start_time_ms is after end_time_ms", pages=0)

    found = {}
    observed_times = []
    previous_signature = None
    completed = False
    completion_reason = None
    pages = 0
    for page_no in range(1, max_pages + 1):
        response = _get("/api/v2/mix/market/history-fund-rate", {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "pageSize": str(page_size),
            "pageNo": str(page_no),
        })
        pages = page_no
        if response.get("code") != "00000":
            message = response.get("msg") or "empty or invalid API response"
            return _error_result(
                f"history-fund-rate failed: {message}", pages=pages,
                partial_count=len(found),
            )
        data = response.get("data", [])
        if isinstance(data, dict):
            rows = data.get("list") or data.get("fundingRateList") or []
        else:
            rows = data if isinstance(data, list) else []
        if not rows:
            completed = True
            completion_reason = "empty_page"
            break

        parsed_times = []
        signature = tuple(str(row.get("fundingTime")) for row in rows if isinstance(row, dict))
        if signature and signature == previous_signature:
            return _error_result(
                "history-fund-rate pagination repeated a page", pages=pages,
                partial_count=len(found),
            )
        previous_signature = signature

        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                funding_time = int(row["fundingTime"])
            except (KeyError, TypeError, ValueError):
                return _error_result(
                    "history-fund-rate contained an invalid fundingTime",
                    pages=pages, partial_count=len(found),
                )
            rate = _finite_float(row.get("fundingRate"))
            if rate is None:
                return _error_result(
                    "history-fund-rate contained an invalid fundingRate",
                    pages=pages, partial_count=len(found),
                )
            row_symbol = row.get("symbol") or symbol
            if row_symbol != symbol:
                continue
            parsed_times.append(funding_time)
            observed_times.append(funding_time)
            if start_time_ms is not None and funding_time < start_time_ms:
                continue
            if end_time_ms is not None and funding_time > end_time_ms:
                continue
            found[(row_symbol, funding_time)] = {
                "symbol": row_symbol,
                "funding_rate": rate,
                "funding_time": funding_time,
            }

        if len(rows) < page_size:
            completed = True
            completion_reason = "short_page"
            break
        if start_time_ms is not None and parsed_times and min(parsed_times) <= start_time_ms:
            completed = True
            completion_reason = "requested_start_reached"
            break

    if not completed:
        # A page cap is partial coverage, not a transport failure. Callers may
        # safely advance only trades whose own cursor is newer than the proven
        # observed boundary below.
        completion_reason = "page_limit"

    observed_times = sorted(set(observed_times))
    observed_start = observed_times[0] if observed_times else None
    observed_end = observed_times[-1] if observed_times else None
    start_covered = (
        start_time_ms is None
        or (observed_start is not None and observed_start <= start_time_ms)
    )
    schedule = get_funding_schedule(symbol)
    interval_ms = schedule.get("interval_ms") if schedule.get("ok") else None
    latest_expected_ms = None
    verified_through_ms = None
    right_edge_covered = False
    if interval_ms is not None and observed_end is not None:
        latest_expected_ms = int(schedule["next_funding_time"]) - int(interval_ms)
        schedule_tolerance_ms = 5 * 60 * 1000
        right_edge_covered = (
            abs(int(observed_end) - latest_expected_ms) <= schedule_tolerance_ms
            and int(schedule["request_time"]) < int(schedule["next_funding_time"])
        )
        target_end = (
            end_time_ms if end_time_ms is not None else int(schedule["request_time"])
        )
        if right_edge_covered and target_end < int(schedule["next_funding_time"]):
            verified_through_ms = int(target_end)
    return {
        "ok": True,
        "items": sorted(found.values(), key=lambda item: item["funding_time"]),
        "error": None,
        "pages": pages,
        "coverage": {
            "requested_start_ms": start_time_ms,
            "requested_end_ms": end_time_ms,
            "observed_start_ms": observed_start,
            "observed_end_ms": observed_end,
            "observed_times_ms": observed_times,
            "start_covered": start_covered,
            "recent_edge_loaded": right_edge_covered,
            "verified_through_ms": verified_through_ms,
            "schedule_interval_ms": interval_ms,
            "latest_expected_ms": latest_expected_ms,
            "schedule_error": schedule.get("error"),
            "completion_reason": completion_reason,
        },
    }


def get_historical_mark_price(symbol: str, settlement_time_ms: int,
                              window_ms: int = 180_000) -> dict:
    """Return the nearest public 1m mark candle price around a settlement."""
    try:
        settlement_time_ms = int(settlement_time_ms)
        window_ms = min(900_000, max(60_000, int(window_ms)))
    except (TypeError, ValueError):
        return {"ok": False, "mark_price": None, "error": "invalid settlement time"}

    response = _get("/api/v2/mix/market/history-mark-candles", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "granularity": "1m",
        "startTime": str(settlement_time_ms - window_ms),
        "endTime": str(settlement_time_ms + window_ms),
        "limit": "100",
    })
    if response.get("code") != "00000":
        return {
            "ok": False,
            "mark_price": None,
            "error": f"history-mark-candles failed: {response.get('msg') or 'empty or invalid API response'}",
        }
    rows = response.get("data")
    if not isinstance(rows, list) or not rows:
        return {"ok": False, "mark_price": None, "error": "no mark candles returned"}

    candles = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            candle_time = int(row[0])
        except (TypeError, ValueError):
            continue
        open_price = _finite_float(row[1])
        close_price = _finite_float(row[4])
        if open_price is None or close_price is None or open_price <= 0 or close_price <= 0:
            continue
        candles.append((candle_time, open_price, close_price))
    if not candles:
        return {"ok": False, "mark_price": None, "error": "mark candles were malformed"}

    candle_time, open_price, close_price = min(
        candles, key=lambda candle: abs(candle[0] - settlement_time_ms)
    )
    if abs(candle_time - settlement_time_ms) > window_ms:
        return {"ok": False, "mark_price": None, "error": "no mark candle inside window"}
    if candle_time >= settlement_time_ms:
        mark_price = open_price
        price_field = "open"
    else:
        mark_price = close_price
        price_field = "close"
    return {
        "ok": True,
        "mark_price": mark_price,
        "candle_time": candle_time,
        "price_field": price_field,
        "error": None,
    }


def get_funding_bills(*, symbol: str | None = None,
                      coin: str | None = MARGIN_COIN,
                      business_type: str | None = None,
                      id_less_than: str | None = None,
                      start_time_ms: int | None = None,
                      end_time_ms: int | None = None,
                      limit: int = 100,
                      max_pages: int = 10) -> dict:
    """Fetch actual private funding bills. This never calls a private API in DRY_RUN."""
    if DRY_RUN:
        return _error_result(
            "private funding bills are disabled in DRY_RUN",
            pages=0, end_id=id_less_than, skipped=True,
        )
    limit = min(100, max(1, int(limit)))
    max_pages = min(100, max(1, int(max_pages)))
    start_time_ms = int(start_time_ms) if start_time_ms is not None else None
    end_time_ms = int(end_time_ms) if end_time_ms is not None else None
    if start_time_ms is not None and end_time_ms is not None and start_time_ms > end_time_ms:
        return _error_result(
            "start_time_ms is after end_time_ms", pages=0, end_id=id_less_than,
        )

    cursor = str(id_less_than) if id_less_than else None
    found = {}
    observed_times = []
    completed = False
    completion_reason = None
    pages = 0
    for page_no in range(1, max_pages + 1):
        params = {"productType": PRODUCT_TYPE, "limit": str(limit)}
        if symbol:
            params["symbol"] = symbol
        if coin:
            params["coin"] = coin
        if business_type:
            params["businessType"] = business_type
        if cursor:
            params["idLessThan"] = cursor
        if start_time_ms is not None:
            params["startTime"] = str(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = str(end_time_ms)

        response = _get("/api/v2/mix/account/bill", params)
        pages = page_no
        if response.get("code") != "00000":
            message = response.get("msg") or "empty or invalid API response"
            return _error_result(
                f"account bill failed: {message}", pages=pages,
                end_id=cursor, partial_count=len(found),
            )
        data = response.get("data")
        if not isinstance(data, dict):
            return _error_result(
                "account bill returned malformed data", pages=pages,
                end_id=cursor, partial_count=len(found),
            )
        rows = data.get("bills")
        if not isinstance(rows, list):
            return _error_result(
                "account bill returned malformed bills", pages=pages,
                end_id=cursor, partial_count=len(found),
            )
        next_cursor = data.get("endId")
        parsed_times = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                created_time = int(row["cTime"])
            except (KeyError, TypeError, ValueError):
                return _error_result(
                    "account bill contained an invalid cTime", pages=pages,
                    end_id=cursor, partial_count=len(found),
                )
            parsed_times.append(created_time)
            observed_times.append(created_time)

            raw_type = str(row.get("businessType") or "").lower()
            if raw_type not in FUNDING_BUSINESS_TYPES:
                continue
            bill_id = str(row.get("billId") or "")
            amount = _finite_float(row.get("amount"))
            fee = _finite_float(row.get("fee") or 0)
            if not bill_id or amount is None or fee is None:
                return _error_result(
                    "funding bill contained an invalid id/amount/fee", pages=pages,
                    end_id=cursor, partial_count=len(found),
                )
            if start_time_ms is not None and created_time < start_time_ms:
                continue
            if end_time_ms is not None and created_time > end_time_ms:
                continue
            row_symbol = row.get("symbol") or symbol
            if not row_symbol or (symbol and row_symbol != symbol):
                continue
            found[bill_id] = {
                "bill_id": bill_id,
                "symbol": row_symbol,
                "amount": amount,
                "fee": fee,
                "business_type": raw_type,
                "coin": row.get("coin"),
                "balance": _finite_float(row.get("balance")),
                "created_time": created_time,
            }

        if not rows:
            completed = True
            completion_reason = "empty_page"
            cursor = str(next_cursor) if next_cursor else cursor
            break
        if start_time_ms is not None and parsed_times and min(parsed_times) <= start_time_ms:
            completed = True
            completion_reason = "requested_start_reached"
            cursor = str(next_cursor) if next_cursor else cursor
            break
        if len(rows) < limit or not next_cursor:
            completed = True
            completion_reason = "short_page" if len(rows) < limit else "no_cursor"
            cursor = str(next_cursor) if next_cursor else cursor
            break
        next_cursor = str(next_cursor)
        if next_cursor == cursor:
            return _error_result(
                "account bill pagination repeated endId", pages=pages,
                end_id=cursor, partial_count=len(found),
            )
        cursor = next_cursor

    if not completed:
        # Preserve partial pages and expose their boundary. Completeness is
        # established against expected public settlements by the caller.
        completion_reason = "page_limit"

    observed_start = min(observed_times) if observed_times else None
    observed_end = max(observed_times) if observed_times else None
    start_covered = (
        start_time_ms is None
        or (observed_start is not None and observed_start <= start_time_ms)
    )
    return {
        "ok": True,
        "items": sorted(found.values(), key=lambda item: item["created_time"]),
        "error": None,
        "pages": pages,
        "end_id": cursor,
        "coverage": {
            "requested_start_ms": start_time_ms,
            "requested_end_ms": end_time_ms,
            "observed_start_ms": observed_start,
            "observed_end_ms": observed_end,
            "start_covered": start_covered,
            "recent_edge_loaded": pages > 0,
            "completion_reason": completion_reason,
        },
    }


def get_contract_info(symbol: str) -> dict:
    global _contract_cache, _contract_cache_updated_at
    now = time.time()
    if _contract_cache and now - _contract_cache_updated_at < _CONTRACT_CACHE_TTL_SECONDS:
        return _contract_cache.get(symbol, {})

    response = _get("/api/v2/mix/market/contracts", {"productType": PRODUCT_TYPE})
    if response.get("code") != "00000":
        # Never use stale specifications for a new entry. Emergency closes keep
        # their already-normalized stored quantity in _normalize_quantity().
        return {}
    fresh = {
        contract["symbol"]: contract
        for contract in response.get("data", [])
        if contract.get("symbol")
    }
    _contract_cache = fresh
    _contract_cache_updated_at = now
    return _contract_cache.get(symbol, {})


def get_account_info() -> dict:
    if DRY_RUN:
        return {"available": 10_000.0, "equity": 10_000.0, "unrealizedPL": 0.0}
    response = _get("/api/v2/mix/account/accounts", {"productType": PRODUCT_TYPE})
    if response.get("code") == "00000":
        for account in response.get("data", []):
            if account.get("marginCoin") == MARGIN_COIN:
                return {
                    "available": float(account.get("available") or 0),
                    "equity": float(account.get("equity") or 0),
                    "unrealizedPL": float(account.get("unrealizedPL") or 0),
                }
    return {"available": 0.0, "equity": 0.0, "unrealizedPL": 0.0}


def set_leverage(symbol: str, leverage: int) -> bool:
    if DRY_RUN:
        return True
    succeeded = True
    for hold_side in ("long", "short"):
        response = _post("/api/v2/mix/account/set-leverage", {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "marginCoin": MARGIN_COIN,
            "leverage": str(leverage),
            "holdSide": hold_side,
        })
        if response.get("code") != "00000":
            succeeded = False
            logger.warning(f"set_leverage {symbol} {hold_side}: {response.get('msg')}")
    return succeeded


def _strict_positive_decimal(info: dict, key: str) -> Decimal | None:
    try:
        value = Decimal(str(info[key]))
        return value if value > 0 else None
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return None


def _contract_constraints(info: dict) -> tuple[Decimal, Decimal, Decimal] | None:
    step = _strict_positive_decimal(info, "sizeMultiplier")
    minimum = _strict_positive_decimal(info, "minTradeNum")
    min_notional = _strict_positive_decimal(info, "minTradeUSDT")
    if not all((step, minimum, min_notional)):
        return None
    return step, minimum, min_notional


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _normalize_quantity(symbol: str, quantity: float, price: float,
                        enforce_min_notional: bool = True) -> float:
    info = get_contract_info(symbol)
    constraints = _contract_constraints(info) if info else None
    if (
        not info
        or constraints is None
        or (enforce_min_notional and info.get("symbolStatus") != "normal")
    ):
        if not enforce_min_notional and quantity > 0:
            # The entry quantity was already normalized with verified metadata.
            # Preserve it for an emergency close rather than inventing a new step.
            logger.warning(
                f"{symbol}: current contract constraints unavailable during close; "
                "using the stored entry quantity unchanged"
            )
            return float(Decimal(str(quantity)))
        logger.error(f"{symbol}: complete active contract constraints required for entry")
        return 0.0

    step, minimum, min_notional = constraints
    try:
        requested = Decimal(str(quantity))
    except (InvalidOperation, TypeError, ValueError):
        return 0.0
    if requested <= 0:
        return 0.0
    normalized = (requested / step).to_integral_value(rounding=ROUND_DOWN) * step

    if normalized < minimum:
        logger.warning(
            f"{symbol}: quantity {normalized} is below minTradeNum {minimum}; order skipped"
        )
        return 0.0
    if enforce_min_notional and normalized * Decimal(str(price)) < min_notional:
        logger.warning(
            f"{symbol}: notional {normalized * Decimal(str(price)):.4f} is below "
            f"minTradeUSDT {min_notional}; order skipped"
        )
        return 0.0
    return float(normalized)


def _calc_size(symbol: str, notional_usdt: float, price: float) -> float:
    """Convert order notional to base quantity using Bitget's contract step."""
    if notional_usdt <= 0 or price <= 0:
        return 0.0
    return _normalize_quantity(symbol, notional_usdt / price, price)


def _client_order_id(symbol: str, side: str, prefix: str) -> str:
    return f"cb{prefix}{int(time.time() * 1000)}{symbol[:6]}{side[0]}"[:32]


def place_order(symbol: str, side: str, size_usdt: float,
                leverage: int, current_price: float) -> dict:
    quantity = _calc_size(symbol, size_usdt, current_price)
    if quantity <= 0:
        return {}
    client_oid = _client_order_id(symbol, side, "o")

    if DRY_RUN:
        fill_price = current_price * (1 + SLIPPAGE_RATE if side == "long" else 1 - SLIPPAGE_RATE)
        order_id = f"dry_{int(time.time() * 1000)}"
        logger.info(
            f"[DRY-RUN] OPEN {side.upper()} {symbol} @ {fill_price:.8g} | "
            f"qty={quantity} | notional~{quantity * fill_price:.2f} USDT | {leverage}x"
        )
        return {
            "orderId": order_id,
            "clientOid": client_oid,
            "dry_run": True,
            "quantity": quantity,
            "fill_price": fill_price,
            "notional_usdt": quantity * fill_price,
        }

    if not set_leverage(symbol, leverage):
        logger.error(f"place_order {symbol}: leverage setup failed; entry aborted")
        return {}

    response = _post("/api/v2/mix/order/place-order", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "marginMode": "isolated",
        "marginCoin": MARGIN_COIN,
        "size": _format_decimal(Decimal(str(quantity))),
        "side": "buy" if side == "long" else "sell",
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": client_oid,
    })
    if response.get("code") != "00000":
        logger.error(
            f"place_order {symbol} {side} clientOid={client_oid}: {response.get('msg')}"
        )
        return {}
    data = dict(response.get("data", {}))
    data.update({
        "clientOid": data.get("clientOid") or client_oid,
        "quantity": quantity,
        # V2 accepts the market order asynchronously; this is the observed estimate.
        "fill_price": current_price,
        "notional_usdt": quantity * current_price,
    })
    return data



def place_limit_order(symbol: str, side: str, size_usdt: float,
                      leverage: int, limit_price: float,
                      current_price: float) -> dict:
    """
    Place a limit entry order instead of a market order.

    The limit price sits slightly below the current price for longs (buying
    the pullback) and slightly above for shorts. If the order is not filled
    within one candle period, the caller should cancel it.

    In DRY_RUN mode, the order is filled if the limit price is between
    bar_low and bar_high of the current bar — a conservative approximation.
    Since we cannot know intrabar order during backtesting, DRY_RUN
    optimistically fills at the limit price (best case for limit orders).
    """
    quantity = _calc_size(symbol, size_usdt, limit_price)
    if quantity <= 0:
        return {}
    client_oid = _client_order_id(symbol, side, "l")

    if DRY_RUN:
        # In DRY_RUN: assume fill at limit price (optimistic for limits)
        fill_price = limit_price
        order_id = f"dry_lim_{int(time.time() * 1000)}"
        logger.info(
            f"[DRY-RUN] LIMIT {side.upper()} {symbol} @ {fill_price:.8g} "
            f"(market={current_price:.8g}) | qty={quantity} | "
            f"notional~{quantity * fill_price:.2f} USDT | {leverage}x"
        )
        return {
            "orderId": order_id,
            "clientOid": client_oid,
            "dry_run": True,
            "quantity": quantity,
            "fill_price": fill_price,
            "notional_usdt": quantity * fill_price,
            "order_type": "limit",
        }

    if not set_leverage(symbol, leverage):
        logger.error(f"place_limit_order {symbol}: leverage setup failed")
        return {}

    response = _post("/api/v2/mix/order/place-order", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "marginMode": "isolated",
        "marginCoin": MARGIN_COIN,
        "size": _format_decimal(Decimal(str(quantity))),
        "price": _format_decimal(Decimal(str(limit_price))),
        "side": "buy" if side == "long" else "sell",
        "tradeSide": "open",
        "orderType": "limit",
        "clientOid": client_oid,
        "force": "gtc",
    })
    if response.get("code") != "00000":
        logger.error(
            f"place_limit_order {symbol} {side} @ {limit_price}: "
            f"{response.get('msg')}"
        )
        return {}
    data = dict(response.get("data", {}))
    data.update({
        "clientOid": data.get("clientOid") or client_oid,
        "quantity": quantity,
        "fill_price": limit_price,
        "notional_usdt": quantity * limit_price,
        "order_type": "limit",
    })
    return data


def close_order(symbol: str, side: str, quantity: float,
                current_price: float) -> dict:
    normalized = _normalize_quantity(
        symbol, quantity, current_price, enforce_min_notional=False
    )
    if normalized <= 0:
        return {}
    client_oid = _client_order_id(symbol, side, "c")

    if DRY_RUN:
        fill_price = current_price * (1 - SLIPPAGE_RATE if side == "long" else 1 + SLIPPAGE_RATE)
        logger.info(
            f"[DRY-RUN] CLOSE {side.upper()} {symbol} @ {fill_price:.8g} | qty={normalized}"
        )
        return {
            "orderId": f"dry_close_{int(time.time() * 1000)}",
            "clientOid": client_oid,
            "dry_run": True,
            "quantity": normalized,
            "fill_price": fill_price,
        }

    response = _post("/api/v2/mix/order/place-order", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "marginMode": "isolated",
        "marginCoin": MARGIN_COIN,
        "size": _format_decimal(Decimal(str(normalized))),
        "side": "sell" if side == "long" else "buy",
        "tradeSide": "close",
        "orderType": "market",
        "clientOid": client_oid,
    })
    if response.get("code") != "00000":
        logger.error(
            f"close_order {symbol} clientOid={client_oid}: {response.get('msg')}"
        )
        return {}
    data = dict(response.get("data", {}))
    data.update({
        "clientOid": data.get("clientOid") or client_oid,
        "quantity": normalized,
        "fill_price": current_price,
    })
    return data
