"""
Externe Marktsentiment-Daten von Bitget Futures (oeffentliche Endpoints).

Liefert Echtzeit-Features die nicht im OHLCV-Stream enthalten sind:
  * Long/Short Account Ratio  - Crowd-Positionierung
  * Long/Short Position Ratio - Volumengewichtete Positionsverteilung
  * Taker Buy/Sell Volume     - Aggressivitaet
  * Orderbook Bid/Ask Imbalance - Liquiditaetsverteilung

Historische Tiefe nur ~30 Tage, deshalb nur als Live-Features nutzbar.
"""

import logging
import time
import requests

logger = logging.getLogger(__name__)

_BITGET_BASE = "https://api.bitget.com"
_PRODUCT_TYPE = "USDT-FUTURES"
_CACHE_TTL = 300
_REQUEST_TIMEOUT = 8

_cache: dict[str, dict] = {}
_cache_updated: dict[str, float] = {}

SENTIMENT_FEATURES = [
    "ls_account_ratio",
    "ls_position_ratio",
    "taker_buy_ratio",
    "taker_imbalance",
    "book_imbalance_5",
    "book_imbalance_20",
    "oi_change_pct",
]


def _is_fresh(symbol: str) -> bool:
    return time.time() - _cache_updated.get(symbol, 0) < _CACHE_TTL


def _fetch_ls_account(symbol: str) -> dict:
    try:
        r = requests.get(f"{_BITGET_BASE}/api/v2/mix/market/account-long-short",
                         params={"symbol": symbol, "productType": _PRODUCT_TYPE, "period": "1h"},
                         timeout=_REQUEST_TIMEOUT)
        data = r.json().get("data", [])
        if data:
            return {"ls_account_ratio": float(data[0].get("longShortAccountRatio", 1.0))}
    except Exception as e:
        logger.debug(f"LS account {symbol}: {e}")
    return {"ls_account_ratio": 1.0}


def _fetch_ls_position(symbol: str) -> dict:
    try:
        r = requests.get(f"{_BITGET_BASE}/api/v2/mix/market/position-long-short",
                         params={"symbol": symbol, "productType": _PRODUCT_TYPE, "period": "1h"},
                         timeout=_REQUEST_TIMEOUT)
        data = r.json().get("data", [])
        if data:
            return {"ls_position_ratio": float(data[0].get("longShortPositionRatio", 1.0))}
    except Exception as e:
        logger.debug(f"LS position {symbol}: {e}")
    return {"ls_position_ratio": 1.0}


def _fetch_taker(symbol: str) -> dict:
    try:
        r = requests.get(f"{_BITGET_BASE}/api/v2/mix/market/taker-buy-sell",
                         params={"symbol": symbol, "productType": _PRODUCT_TYPE, "period": "1h"},
                         timeout=_REQUEST_TIMEOUT)
        data = r.json().get("data", [])
        if data:
            buy = float(data[0].get("buyVolume", 0))
            sell = float(data[0].get("sellVolume", 0))
            total = buy + sell
            return {
                "taker_buy_ratio": buy / total if total > 0 else 0.5,
                "taker_imbalance": (buy - sell) / total if total > 0 else 0.0,
            }
    except Exception as e:
        logger.debug(f"Taker {symbol}: {e}")
    return {"taker_buy_ratio": 0.5, "taker_imbalance": 0.0}


def _fetch_book(symbol: str) -> dict:
    try:
        r = requests.get(f"{_BITGET_BASE}/api/v2/mix/market/merge-depth",
                         params={"symbol": symbol, "productType": _PRODUCT_TYPE, "limit": "20"},
                         timeout=_REQUEST_TIMEOUT)
        data = r.json().get("data", {})
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        b5 = sum(float(b[1]) for b in bids[:5]) if bids else 0
        a5 = sum(float(a[1]) for a in asks[:5]) if asks else 0
        t5 = b5 + a5
        b20 = sum(float(b[1]) for b in bids[:20]) if bids else 0
        a20 = sum(float(a[1]) for a in asks[:20]) if asks else 0
        t20 = b20 + a20
        return {
            "book_imbalance_5": (b5 - a5) / t5 if t5 > 0 else 0.0,
            "book_imbalance_20": (b20 - a20) / t20 if t20 > 0 else 0.0,
        }
    except Exception as e:
        logger.debug(f"Book {symbol}: {e}")
    return {"book_imbalance_5": 0.0, "book_imbalance_20": 0.0}


def _fetch_oi_change(symbol: str) -> dict:
    try:
        from bitget_client import get_open_interest
        current = get_open_interest(symbol)
        prev = _cache.get(symbol, {}).get("_oi_raw", current)
        change = (current - prev) / prev if prev > 0 else 0.0
        return {"oi_change_pct": change, "_oi_raw": current}
    except Exception as e:
        logger.debug(f"OI {symbol}: {e}")
    return {"oi_change_pct": 0.0, "_oi_raw": 0.0}


def refresh(symbol: str) -> dict:
    if _is_fresh(symbol):
        return _cache.get(symbol, _neutral())
    result = {}
    result.update(_fetch_ls_account(symbol))
    result.update(_fetch_ls_position(symbol))
    result.update(_fetch_taker(symbol))
    result.update(_fetch_book(symbol))
    result.update(_fetch_oi_change(symbol))
    _cache[symbol] = result
    _cache_updated[symbol] = time.time()
    return result


def get_features(symbol: str) -> dict:
    data = refresh(symbol)
    return {key: float(data.get(key, 0.0)) for key in SENTIMENT_FEATURES}


def _neutral() -> dict:
    return {key: 0.0 for key in SENTIMENT_FEATURES}
