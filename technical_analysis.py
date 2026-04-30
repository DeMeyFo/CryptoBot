import logging
import pandas as pd
import ta
from bitget_client import get_candles

logger = logging.getLogger(__name__)


def fetch_ohlcv(symbol: str, granularity: str = "15m", limit: int = 200) -> pd.DataFrame:
    raw = get_candles(symbol, granularity, limit)
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def calculate_signals(symbol: str, granularity: str = "15m") -> dict:
    """
    Returns TA signal dict with score in [-100, +100] and indicator values.
    Positive = bullish, negative = bearish.

    Components (max contribution):
      EMA crossover  ±30
      RSI            ±25
      MACD           ±25
      Bollinger Bands ±20
      Volume boost    ±10 % amplitude amplifier
    """
    df = fetch_ohlcv(symbol, granularity)
    if df.empty or len(df) < 60:
        logger.warning(f"{symbol}: not enough candles ({len(df)})")
        return {"score": 0, "indicators": {}, "error": "insufficient data"}

    close = df["close"]
    volume = df["volume"]
    score = 0
    ind: dict = {}

    # ── EMA Crossover (±30) ────────────────────────────────────────────────────
    ema9 = ta.trend.ema_indicator(close, window=9)
    ema21 = ta.trend.ema_indicator(close, window=21)
    ema50 = ta.trend.ema_indicator(close, window=50)

    v9, v21, v50 = ema9.iloc[-1], ema21.iloc[-1], ema50.iloc[-1]
    ind["ema9"] = round(v9, 6)
    ind["ema21"] = round(v21, 6)
    ind["ema50"] = round(v50, 6)

    if v9 > v21 > v50:
        score += 30
    elif v9 > v21:
        score += 15
    elif v9 < v21 < v50:
        score -= 30
    elif v9 < v21:
        score -= 15

    # ── RSI (±25) ─────────────────────────────────────────────────────────────
    rsi_s = ta.momentum.RSIIndicator(close, window=14).rsi()
    rsi = rsi_s.iloc[-1]
    ind["rsi"] = round(rsi, 2)

    if rsi < 25:
        score += 25
    elif rsi < 35:
        score += 15
    elif rsi < 45:
        score += 5
    elif rsi > 75:
        score -= 25
    elif rsi > 65:
        score -= 15
    elif rsi > 55:
        score -= 5

    # ── MACD (±25) ────────────────────────────────────────────────────────────
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_obj.macd()
    signal_line = macd_obj.macd_signal()
    histogram = macd_obj.macd_diff()

    if macd_line is not None and len(macd_line.dropna()) >= 2:
        ml = macd_line.iloc[-1]
        sl = signal_line.iloc[-1]
        hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        ind["macd"] = round(ml, 6)
        ind["macd_signal"] = round(sl, 6)
        ind["macd_hist"] = round(hist, 6)

        if ml > sl and prev_hist <= 0:
            score += 25
        elif ml > sl:
            score += 12
        elif ml < sl and prev_hist >= 0:
            score -= 25
        elif ml < sl:
            score -= 12

    # ── Bollinger Bands (±20) ─────────────────────────────────────────────────
    bb_obj = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_lower = bb_obj.bollinger_lband().iloc[-1]
    bb_upper = bb_obj.bollinger_hband().iloc[-1]
    price = close.iloc[-1]
    band_range = bb_upper - bb_lower

    ind["bb_lower"] = round(bb_lower, 6)
    ind["bb_upper"] = round(bb_upper, 6)
    ind["current_price"] = round(price, 6)

    if band_range > 0:
        bb_pct = (price - bb_lower) / band_range
        ind["bb_pct"] = round(bb_pct, 3)
        if bb_pct < 0.10:
            score += 20
        elif bb_pct < 0.25:
            score += 10
        elif bb_pct > 0.90:
            score -= 20
        elif bb_pct > 0.75:
            score -= 10

    # ── Volume Amplifier ──────────────────────────────────────────────────────
    vol_sma = ta.trend.sma_indicator(volume, window=20)
    sma_val = vol_sma.iloc[-1]
    cur_vol = volume.iloc[-1]
    if sma_val > 0:
        vol_ratio = cur_vol / sma_val
        ind["volume_ratio"] = round(vol_ratio, 2)
        if vol_ratio > 2.0:
            score = int(score * 1.15)
        elif vol_ratio > 1.5:
            score = int(score * 1.08)

    score = max(-100, min(100, score))
    ind["score"] = score
    return {"score": score, "indicators": ind}
