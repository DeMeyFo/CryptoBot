import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ta

from bitget_client import get_candles
from config import (
    ADX_ENTRY_THRESHOLD,
    CANDLE_LIMIT,
    DONCHIAN_PERIOD,
    EMA_FAST_PERIOD,
    EMA_HISTORY_CANDLES,
    EMA_MID_PERIOD,
    EMA_SLOW_PERIOD,
    STRATEGY_TIMEFRAME,
)

logger = logging.getLogger(__name__)

_INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
}


def fetch_ohlcv(symbol: str, granularity: str = STRATEGY_TIMEFRAME,
                limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """Fetch sorted Bitget candles and convert numeric columns safely."""
    raw = get_candles(symbol, granularity, limit)
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"],
    )
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True
    )
    df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def closed_candles(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Exclude Bitget's currently forming candle so signals use closed bars only."""
    if df.empty:
        return df
    seconds = _INTERVAL_SECONDS.get(granularity.lower())
    if not seconds:
        return df
    last_open = df["timestamp"].iloc[-1]
    now = datetime.now(timezone.utc).timestamp()
    if last_open.timestamp() + seconds > now:
        return df.iloc[:-1].copy()
    return df.copy()


def _finite_ema(series: pd.Series, span: int,
                history: int = EMA_HISTORY_CANDLES) -> pd.Series:
    """EMA with an explicit finite history, identical for live and backtest."""
    values = series.to_numpy(dtype=float)
    valid = ~np.isnan(values)
    alpha = 2.0 / (span + 1.0)
    lag_weights = alpha * np.power(1.0 - alpha, np.arange(history, dtype=float))
    numerator = np.convolve(np.where(valid, values, 0.0), lag_weights, mode="full")[:len(values)]
    denominator = np.convolve(valid.astype(float), lag_weights, mode="full")[:len(values)]
    output = np.divide(
        numerator,
        denominator,
        out=np.full(len(values), np.nan, dtype=float),
        where=denominator > 0,
    )
    output[:span - 1] = np.nan
    return pd.Series(output, index=series.index)


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the exact causal features shared by live trading and backtesting."""
    frame = df.copy()
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]

    frame["ema_fast"] = _finite_ema(close, EMA_FAST_PERIOD)
    frame["ema_mid"] = _finite_ema(close, EMA_MID_PERIOD)
    frame["ema_slow"] = _finite_ema(close, EMA_SLOW_PERIOD)
    frame["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    frame["atr"] = ta.volatility.AverageTrueRange(
        high, low, close, window=14
    ).average_true_range()

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    frame["adx"] = adx.adx()
    frame["di_pos"] = adx.adx_pos()
    frame["di_neg"] = adx.adx_neg()

    # shift(1) is essential: the signal candle cannot be part of its own breakout level.
    frame["donchian_high"] = high.shift(1).rolling(DONCHIAN_PERIOD).max()
    frame["donchian_low"] = low.shift(1).rolling(DONCHIAN_PERIOD).min()
    frame["volume_sma"] = volume.rolling(20).mean()

    # Diagnostic only; it does not participate in the validated entry rules.
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    frame["macd_hist"] = macd.macd_diff()
    return frame


def evaluate_signal(frame: pd.DataFrame, index: int = -1) -> dict:
    """
    Evaluate the validated 1H trend-breakout setup at one closed candle.

    Entry rules are deliberately boolean rather than an additive mixture:
    Donchian-55 breakout + EMA 20/50/200 structure + ADX/DI + RSI direction.
    """
    if frame.empty:
        return {"score": 0.0, "entry_signal": "hold", "error": "empty data"}

    i = index if index >= 0 else len(frame) + index
    if i < 0 or i >= len(frame):
        return {"score": 0.0, "entry_signal": "hold", "error": "invalid index"}

    row = frame.iloc[i]
    required = [
        "ema_fast", "ema_mid", "ema_slow", "rsi", "atr", "adx",
        "di_pos", "di_neg", "donchian_high", "donchian_low",
    ]
    if any(pd.isna(row[name]) for name in required) or float(row["atr"]) <= 0:
        return {"score": 0.0, "entry_signal": "hold", "error": "indicators unavailable"}

    price = float(row["close"])
    ema_fast = float(row["ema_fast"])
    ema_mid = float(row["ema_mid"])
    ema_slow = float(row["ema_slow"])
    rsi = float(row["rsi"])
    atr = float(row["atr"])
    adx = float(row["adx"])
    di_pos = float(row["di_pos"])
    di_neg = float(row["di_neg"])
    breakout_high = float(row["donchian_high"])
    breakout_low = float(row["donchian_low"])

    long_checks = {
        "breakout": price > breakout_high,
        "ema_structure": price > ema_fast > ema_mid > ema_slow,
        "adx": adx >= ADX_ENTRY_THRESHOLD,
        "di_direction": di_pos > di_neg,
        "rsi_direction": rsi >= 50.0,
    }
    short_checks = {
        "breakout": price < breakout_low,
        "ema_structure": price < ema_fast < ema_mid < ema_slow,
        "adx": adx >= ADX_ENTRY_THRESHOLD,
        "di_direction": di_neg > di_pos,
        "rsi_direction": rsi <= 50.0,
    }

    if all(long_checks.values()):
        entry_signal = "long"
        score = 100.0
    elif all(short_checks.values()):
        entry_signal = "short"
        score = -100.0
    else:
        # Diagnostic funnel score. Only entry_signal may open a trade.
        weights = {
            "breakout": 35.0,
            "ema_structure": 30.0,
            "adx": 10.0,
            "di_direction": 15.0,
            "rsi_direction": 10.0,
        }
        long_score = sum(weights[name] for name, ok in long_checks.items() if ok)
        short_score = sum(weights[name] for name, ok in short_checks.items() if ok)
        score = max(-99.0, min(99.0, long_score - short_score))
        entry_signal = "hold"

    volume_sma = float(row["volume_sma"]) if not pd.isna(row["volume_sma"]) else 0.0
    volume_ratio = float(row["volume"]) / volume_sma if volume_sma > 0 else 0.0
    timestamp = row["timestamp"]
    timestamp_text = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)

    indicators = {
        "ema20": round(ema_fast, 8),
        "ema50": round(ema_mid, 8),
        "ema200": round(ema_slow, 8),
        "rsi": round(rsi, 2),
        "atr": round(atr, 8),
        "adx": round(adx, 2),
        "di_pos": round(di_pos, 2),
        "di_neg": round(di_neg, 2),
        "donchian_high": round(breakout_high, 8),
        "donchian_low": round(breakout_low, 8),
        "volume_ratio": round(volume_ratio, 2),
        "macd_hist": round(float(row["macd_hist"]), 8) if not pd.isna(row["macd_hist"]) else 0.0,
        "current_price": round(price, 8),
        "candle_open": round(float(row["open"]), 8),
        "candle_high": round(float(row["high"]), 8),
        "candle_low": round(float(row["low"]), 8),
        "signal_timestamp": timestamp_text,
        "entry_signal": entry_signal,
        "long_checks": long_checks,
        "short_checks": short_checks,
        "score": score,
    }
    return {
        "score": score,
        "entry_signal": entry_signal,
        "indicators": indicators,
        "atr": atr,
        "timestamp": timestamp_text,
    }


def calculate_signals(symbol: str, granularity: str = STRATEGY_TIMEFRAME) -> dict:
    """Return a signal calculated exclusively from completed candles."""
    df = closed_candles(fetch_ohlcv(symbol, granularity, CANDLE_LIMIT), granularity)
    minimum = max(EMA_HISTORY_CANDLES, DONCHIAN_PERIOD)
    if df.empty or len(df) < minimum:
        logger.warning(f"{symbol}: not enough closed candles ({len(df)}/{minimum})")
        return {
            "score": 0.0,
            "entry_signal": "hold",
            "indicators": {},
            "error": "insufficient closed candle data",
        }

    result = evaluate_signal(prepare_indicators(df))
    if result.get("error"):
        logger.warning(f"{symbol}: signal calculation failed: {result['error']}")
    return result


def calculate_signal_history(symbol: str, after_timestamp: str,
                             granularity: str = STRATEGY_TIMEFRAME) -> dict:
    """
    Return every closed candle snapshot after ``after_timestamp`` in order.

    Trailing stops are path-dependent. If the required first candle is no longer
    available, fail without advancing the database cursor instead of silently
    jumping to the newest bar.
    """
    try:
        after = pd.Timestamp(after_timestamp)
        after = after.tz_localize("UTC") if after.tzinfo is None else after.tz_convert("UTC")
    except (TypeError, ValueError):
        return {"snapshots": [], "error": "invalid trailing cursor timestamp"}

    history_limit = max(CANDLE_LIMIT, 1000)
    frame = closed_candles(fetch_ohlcv(symbol, granularity, history_limit), granularity)
    minimum = max(EMA_HISTORY_CANDLES, DONCHIAN_PERIOD)
    if frame.empty or len(frame) < minimum:
        return {"snapshots": [], "error": "insufficient trailing history"}

    prepared = prepare_indicators(frame)
    target_indices = prepared.index[prepared["timestamp"] > after].tolist()
    if not target_indices:
        return {"snapshots": [], "error": None}

    interval_seconds = _INTERVAL_SECONDS.get(granularity.lower())
    if not interval_seconds:
        return {"snapshots": [], "error": f"unsupported interval {granularity}"}
    expected = after + pd.Timedelta(seconds=interval_seconds)
    first_timestamp = prepared.loc[target_indices[0], "timestamp"]
    if first_timestamp != expected:
        return {
            "snapshots": [],
            "error": (
                f"trailing history gap: expected {expected.isoformat()}, "
                f"first available {first_timestamp.isoformat()}"
            ),
        }

    snapshots = []
    previous = after
    for index in target_indices:
        timestamp = prepared.loc[index, "timestamp"]
        if timestamp != previous + pd.Timedelta(seconds=interval_seconds):
            return {
                "snapshots": [],
                "error": f"non-contiguous trailing candles near {timestamp.isoformat()}",
            }
        result = evaluate_signal(prepared, int(index))
        if result.get("error"):
            return {
                "snapshots": [],
                "error": f"invalid trailing indicators at {timestamp.isoformat()}",
            }
        snapshots.append(result)
        previous = timestamp
    return {"snapshots": snapshots, "error": None}
