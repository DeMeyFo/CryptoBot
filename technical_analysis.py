import logging
import numpy as np
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


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Returns a Series with values 1 (bullish) or -1 (bearish).
    Standard Supertrend: upper/lower bands based on ATR.
    """
    high, low, close = df["high"], df["low"], df["close"]
    hl2 = (high + low) / 2

    atr = ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(np.nan, index=df.index)
    trend = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        # Final upper band
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        # Final lower band
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Trend
        if close.iloc[i] > final_upper.iloc[i]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    return trend


def calculate_signals(symbol: str, granularity: str = "15m") -> dict:
    """
    Returns TA signal dict with score in [-100, +100] and indicator values.

    Components:
      EMA crossover     ±20
      RSI               ±15
      Stochastic RSI    ±20
      MACD              ±20
      Bollinger Bands   ±15
      Supertrend        ±20
      Volume amplifier  up to ±10% multiplier on score
    """
    df = fetch_ohlcv(symbol, granularity)
    if df.empty or len(df) < 60:
        logger.warning(f"{symbol}: not enough candles ({len(df)})")
        return {"score": 0, "indicators": {}, "error": "insufficient data"}

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    volume = df["volume"]
    score = 0
    ind: dict = {}

    # ── EMA Crossover (±20) ───────────────────────────────────────────────────
    ema9  = ta.trend.ema_indicator(close, window=9)
    ema21 = ta.trend.ema_indicator(close, window=21)
    ema50 = ta.trend.ema_indicator(close, window=50)

    v9, v21, v50 = ema9.iloc[-1], ema21.iloc[-1], ema50.iloc[-1]
    ind["ema9"]  = round(v9, 6)
    ind["ema21"] = round(v21, 6)
    ind["ema50"] = round(v50, 6)

    if v9 > v21 > v50:
        score += 20
    elif v9 > v21:
        score += 10
    elif v9 < v21 < v50:
        score -= 20
    elif v9 < v21:
        score -= 10

    # ── RSI (±15) ─────────────────────────────────────────────────────────────
    rsi_s = ta.momentum.RSIIndicator(close, window=14).rsi()
    rsi = rsi_s.iloc[-1]
    ind["rsi"] = round(rsi, 2)

    if rsi < 25:
        score += 15
    elif rsi < 35:
        score += 8
    elif rsi < 45:
        score += 3
    elif rsi > 75:
        score -= 15
    elif rsi > 65:
        score -= 8
    elif rsi > 55:
        score -= 3

    # ── Stochastic RSI (±20) ──────────────────────────────────────────────────
    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    k = stoch_rsi.stochrsi_k()
    d = stoch_rsi.stochrsi_d()

    if k is not None and len(k.dropna()) >= 2:
        k_val  = k.iloc[-1]
        d_val  = d.iloc[-1]
        k_prev = k.iloc[-2]
        ind["stoch_rsi_k"] = round(k_val, 3)
        ind["stoch_rsi_d"] = round(d_val, 3)

        if k_val < 0.20 and k_val > k_prev:   # oversold + turning up
            score += 20
        elif k_val < 0.20:
            score += 10
        elif k_val > 0.80 and k_val < k_prev:  # overbought + turning down
            score -= 20
        elif k_val > 0.80:
            score -= 10

    # ── MACD (±20) ────────────────────────────────────────────────────────────
    macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line  = macd_obj.macd()
    signal_line = macd_obj.macd_signal()
    histogram  = macd_obj.macd_diff()

    if macd_line is not None and len(macd_line.dropna()) >= 2:
        ml        = macd_line.iloc[-1]
        sl        = signal_line.iloc[-1]
        hist      = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        ind["macd"]        = round(ml, 6)
        ind["macd_signal"] = round(sl, 6)
        ind["macd_hist"]   = round(hist, 6)

        if ml > sl and prev_hist <= 0:    # bullish crossover
            score += 20
        elif ml > sl:
            score += 10
        elif ml < sl and prev_hist >= 0:  # bearish crossover
            score -= 20
        elif ml < sl:
            score -= 10

    # ── Bollinger Bands (±15) ─────────────────────────────────────────────────
    bb_obj   = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_lower = bb_obj.bollinger_lband().iloc[-1]
    bb_upper = bb_obj.bollinger_hband().iloc[-1]
    price    = close.iloc[-1]
    band_range = bb_upper - bb_lower

    ind["bb_lower"]      = round(bb_lower, 6)
    ind["bb_upper"]      = round(bb_upper, 6)
    ind["current_price"] = round(price, 6)

    if band_range > 0:
        bb_pct = (price - bb_lower) / band_range
        ind["bb_pct"] = round(bb_pct, 3)
        if bb_pct < 0.10:
            score += 15
        elif bb_pct < 0.25:
            score += 7
        elif bb_pct > 0.90:
            score -= 15
        elif bb_pct > 0.75:
            score -= 7

    # ── Supertrend (±20) ──────────────────────────────────────────────────────
    try:
        trend = _supertrend(df, period=10, multiplier=3.0)
        st_val = int(trend.iloc[-1])
        st_prev = int(trend.iloc[-2])
        ind["supertrend"] = st_val

        if st_val == 1 and st_prev == -1:   # fresh bullish flip
            score += 20
        elif st_val == 1:
            score += 10
        elif st_val == -1 and st_prev == 1: # fresh bearish flip
            score -= 20
        elif st_val == -1:
            score -= 10
    except Exception as e:
        logger.debug(f"Supertrend error {symbol}: {e}")

    # ── ATR (returned for dynamic SL/TP, not scored) ──────────────────────────
    atr_series = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    atr_val = atr_series.iloc[-1]
    ind["atr"] = round(atr_val, 8)

    # ── ADX – trend strength (returned for ADX filter, not scored here) ───────
    adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
    adx_val = adx_obj.adx().iloc[-1]
    ind["adx"] = round(adx_val, 2) if not __import__("math").isnan(adx_val) else 0.0

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
    return {"score": score, "indicators": ind, "atr": atr_val}
