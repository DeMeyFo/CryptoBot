import hmac
import hashlib
import base64
import time
import json
import logging
import requests
from config import API_KEY, SECRET_KEY, PASSPHRASE, DRY_RUN

logger = logging.getLogger(__name__)

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"

_contract_cache: dict = {}


# ── Authentication ─────────────────────────────────────────────────────────────

def _sign(ts: str, method: str, path: str, body: str = "") -> str:
    msg = ts + method.upper() + path + body
    mac = hmac.new(SECRET_KEY.encode(), msg.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()


def _headers(method: str, path: str, body: str = "") -> dict:
    ts = str(int(time.time() * 1000))
    return {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": _sign(ts, method, path, body),
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json",
        "locale": "en-US",
    }


def _get(path: str, params: dict = None) -> dict:
    query = ("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else ""
    full = path + query
    try:
        r = requests.get(BASE_URL + full, headers=_headers("GET", full), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"GET {path} failed: {e}")
        return {}


def _post(path: str, data: dict) -> dict:
    body = json.dumps(data)
    try:
        r = requests.post(BASE_URL + path, headers=_headers("POST", path, body),
                          data=body, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"POST {path} failed: {e}")
        return {}


# ── Market Data ────────────────────────────────────────────────────────────────

def get_tickers() -> list:
    resp = _get("/api/v2/mix/market/tickers", {"productType": PRODUCT_TYPE})
    if resp.get("code") != "00000":
        logger.error(f"get_tickers: {resp.get('msg')}")
        return []
    tickers = resp["data"]
    tickers.sort(key=lambda x: float(x.get("usdtVolume") or 0), reverse=True)
    return tickers


# Backtested whitelist – ordered by backtest performance (Profit Factor desc)
# Update this list after running new backtests.
WHITELIST = [
    "SOLUSDT",    # PF 1.33  WR 47%  (3m Binance)
    "TAOUSDT",    # PF 1.25  WR 48%  (3m Binance)
    "BTCUSDT",    # PF 1.36  WR 49%  (3m Binance)
    "CLUSDT",     # PF 3.30  WR 50%  (1H backtest – no Binance data)
    "BUSDT",      # PF 3.00  WR 57%  (1H backtest – no Binance data)
    "RAVEUSDT",   # PF inf   WR 100% (1H backtest – no Binance data)
]


def get_top_symbols(count: int) -> list:
    """Return the whitelisted symbols (up to count). Order = backtest rank."""
    return WHITELIST[:count]


def get_candles(symbol: str, granularity: str = "15m", limit: int = 200) -> list:
    resp = _get("/api/v2/mix/market/candles", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "granularity": granularity,
        "limit": str(limit),
    })
    if resp.get("code") != "00000":
        logger.warning(f"get_candles {symbol}: {resp.get('msg')}")
        return []
    return resp.get("data", [])


def get_ticker(symbol: str) -> dict:
    resp = _get("/api/v2/mix/market/ticker", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
    })
    if resp.get("code") != "00000":
        return {}
    data = resp.get("data")
    return data[0] if isinstance(data, list) else data or {}


def get_current_price(symbol: str) -> float:
    t = get_ticker(symbol)
    return float(t.get("lastPr") or t.get("last") or 0)


_oi_cache: dict[str, tuple[float, float]] = {}  # symbol → (unix_ts, oi_usdt)


def get_open_interest(symbol: str) -> float:
    """Return current open interest in USDT. Returns 0 on error."""
    resp = _get("/api/v2/mix/market/open-interest", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
    })
    if resp.get("code") != "00000":
        return 0.0
    data = resp.get("data", {})
    if isinstance(data, list):
        data = data[0] if data else {}
    # Top-level usdtSize
    usdt_size = data.get("usdtSize") or data.get("openInterestUSDT")
    if usdt_size:
        return float(usdt_size)
    # Nested openInterestList
    oi_list = data.get("openInterestList") or []
    if oi_list:
        return float(oi_list[0].get("usdtSize", 0) or 0)
    return 0.0


def get_oi_score(symbol: str) -> float:
    """
    Open Interest signal score in [-50, +50].

    Logic (standard futures interpretation):
      OI rising  + price rising  → longs entering   → bullish  (+)
      OI rising  + price falling → shorts entering  → bearish  (-)
      OI falling + price rising  → short squeeze    → mild +
      OI falling + price falling → longs liquidating→ bearish  (-)

    OI is compared to the cached value from the previous cycle (~5 min).
    Price direction is derived from the ticker's 24h change as a proxy.
    Returns 0.0 if OI data is unavailable or the cache is too fresh/stale.
    """
    import time as _t
    now = _t.time()

    oi = get_open_interest(symbol)
    if oi == 0:
        return 0.0

    ticker     = get_ticker(symbol)
    price_chg  = float(ticker.get("change24h") or ticker.get("changeUtc24h") or 0)
    price_up   = price_chg > 0
    price_down = price_chg < 0

    score = 0.0
    if symbol in _oi_cache:
        prev_ts, prev_oi = _oi_cache[symbol]
        age = now - prev_ts
        if 30 <= age <= 1800 and prev_oi > 0:
            oi_chg = (oi - prev_oi) / prev_oi * 100  # percent change

            if oi_chg > 0.3 and price_up:
                score = min(50.0, oi_chg * 8)    # longs building → bullish
            elif oi_chg > 0.3 and price_down:
                score = max(-50.0, -oi_chg * 6)  # shorts building → bearish
            elif oi_chg < -0.3 and price_up:
                score = 15.0                      # short squeeze → mild bull
            elif oi_chg < -0.3 and price_down:
                score = max(-30.0, oi_chg * 4)   # longs liquidating → bearish

            logger.debug(
                f"OI {symbol}: {prev_oi/1e6:.1f}M→{oi/1e6:.1f}M USDT "
                f"({oi_chg:+.2f}%) price24h={price_chg:+.3f} → score {score:+.1f}"
            )

    _oi_cache[symbol] = (now, oi)
    return round(score, 1)


def get_funding_rate(symbol: str) -> float:
    """
    Return the current funding rate as a decimal (e.g. 0.0001 = 0.01%).
    Positive = longs pay shorts (crowded longs → bearish bias).
    Negative = shorts pay longs (crowded shorts → bullish bias).
    """
    resp = _get("/api/v2/mix/market/current-fund-rate", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
    })
    if resp.get("code") == "00000":
        data = resp.get("data")
        if isinstance(data, list):
            data = data[0] if data else {}
        if isinstance(data, dict):
            return float(data.get("fundingRate") or 0)
    return 0.0


def get_contract_info(symbol: str) -> dict:
    if symbol in _contract_cache:
        return _contract_cache[symbol]
    resp = _get("/api/v2/mix/market/contracts", {"productType": PRODUCT_TYPE})
    if resp.get("code") == "00000":
        for c in resp.get("data", []):
            _contract_cache[c["symbol"]] = c
    return _contract_cache.get(symbol, {})


# ── Account ────────────────────────────────────────────────────────────────────

def get_account_info() -> dict:
    if DRY_RUN:
        return {"available": 10_000.0, "equity": 10_000.0, "unrealizedPL": 0.0}
    resp = _get("/api/v2/mix/account/accounts", {"productType": PRODUCT_TYPE})
    if resp.get("code") == "00000":
        for acc in resp.get("data", []):
            if acc.get("marginCoin") == MARGIN_COIN:
                return {
                    "available": float(acc.get("available") or 0),
                    "equity": float(acc.get("equity") or 0),
                    "unrealizedPL": float(acc.get("unrealizedPL") or 0),
                }
    return {"available": 0.0, "equity": 0.0, "unrealizedPL": 0.0}


# ── Orders ─────────────────────────────────────────────────────────────────────

def set_leverage(symbol: str, leverage: int) -> bool:
    if DRY_RUN:
        return True
    for hold_side in ("long", "short"):
        resp = _post("/api/v2/mix/account/set-leverage", {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "marginCoin": MARGIN_COIN,
            "leverage": str(leverage),
            "holdSide": hold_side,
        })
        if resp.get("code") != "00000":
            logger.warning(f"set_leverage {symbol} {hold_side}: {resp.get('msg')}")
    return True


def _calc_size(symbol: str, size_usdt: float, price: float) -> float:
    """Convert USDT amount to contract size respecting minimum order size."""
    info = get_contract_info(symbol)
    min_size = float(info.get("minTradeNum") or "0.001")
    size = round(size_usdt / price, 6)
    return max(size, min_size)


def place_order(symbol: str, side: str, size_usdt: float,
                leverage: int, current_price: float) -> dict:
    contracts = _calc_size(symbol, size_usdt, current_price)
    set_leverage(symbol, leverage)

    order_side = "buy" if side == "long" else "sell"

    if DRY_RUN:
        order_id = f"dry_{int(time.time() * 1000)}"
        logger.info(
            f"[DRY-RUN] OPEN {side.upper()} {symbol} @ {current_price} "
            f"| {contracts} contracts | {size_usdt} USDT | {leverage}x"
        )
        return {"orderId": order_id, "dry_run": True}

    resp = _post("/api/v2/mix/order/place-order", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "marginMode": "isolated",
        "marginCoin": MARGIN_COIN,
        "size": str(contracts),
        "side": order_side,
        "tradeSide": "open",
        "orderType": "market",
    })
    if resp.get("code") != "00000":
        logger.error(f"place_order {symbol} {side}: {resp.get('msg')}")
        return {}
    return resp.get("data", {})


def close_order(symbol: str, side: str, entry_price: float, size_usdt: float) -> dict:
    contracts = _calc_size(symbol, size_usdt, entry_price)
    close_side = "sell" if side == "long" else "buy"

    if DRY_RUN:
        logger.info(f"[DRY-RUN] CLOSE {side.upper()} {symbol} | {contracts} contracts")
        return {"dry_run": True}

    resp = _post("/api/v2/mix/order/place-order", {
        "symbol": symbol,
        "productType": PRODUCT_TYPE,
        "marginMode": "isolated",
        "marginCoin": MARGIN_COIN,
        "size": str(contracts),
        "side": close_side,
        "tradeSide": "close",
        "orderType": "market",
    })
    if resp.get("code") != "00000":
        logger.error(f"close_order {symbol}: {resp.get('msg')}")
        return {}
    return resp.get("data", {})
