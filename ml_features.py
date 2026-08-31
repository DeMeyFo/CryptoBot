# ml_features.py

import numpy as np
import pandas as pd
from technical_analysis import prepare_indicators, fetch_ohlcv, closed_candles
from config import STRATEGY_TIMEFRAME, CANDLE_LIMIT


def build_features(symbol: str, indicators: dict | None = None) -> dict:
    """
    Baut den vollständigen Feature-Vektor für ein Symbol zum aktuellen Zeitpunkt.
    Verwendet prepare_indicators() als Basis und ergänzt abgeleitete Features.
    """
    df = closed_candles(fetch_ohlcv(symbol, STRATEGY_TIMEFRAME, CANDLE_LIMIT),
                        STRATEGY_TIMEFRAME)
    if df.empty or len(df) < 200:
        return _empty_features()

    frame = prepare_indicators(df)
    frame = _add_derived_features(frame)
    return _extract_feature_row(frame, index=-1)


def build_features_from_frame(frame: pd.DataFrame, index: int = -1) -> dict:
    """
    Feature-Extraktion aus einem vorbereiteten DataFrame.
    Für Training und Backtest, wenn die Candles bereits geladen sind.
    """
    enriched = _add_derived_features(frame)
    return _extract_feature_row(enriched, index=index)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gibt einen DataFrame mit allen Feature-Spalten zurück.
    Für Batch-Training: jede Zeile ist ein Zeitpunkt.
    """
    frame = prepare_indicators(df)
    frame = _add_derived_features(frame)
    return frame[FEATURE_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Feature-Definitionen (Anforderung 3.3, 6.1, 6.2)
# ---------------------------------------------------------------------------

# Basis-Features aus prepare_indicators():
#   adx, rsi, ema_fast, ema_mid, ema_slow, atr, di_pos, di_neg,
#   donchian_high, donchian_low, volume_sma, macd_hist

# Abgeleitete Features:
DERIVED_FEATURES = [
    "ema_fast_mid_spread",     # (ema_fast - ema_mid) / ema_mid
    "ema_mid_slow_spread",     # (ema_mid - ema_slow) / ema_slow
    "atr_pct",                 # atr / close * 100
    "donchian_width_norm",     # (donchian_high - donchian_low) / close
    "rolling_volatility_14",   # close.pct_change().rolling(14).std()
    "adx_slope_5",             # adx.diff(5) – ADX-Trend ueber 5 Perioden
    "rsi_momentum_5",          # rsi.diff(5) – RSI-Momentum
    "di_spread",               # di_pos - di_neg
    "volume_ratio",            # volume / volume_sma
    # Cross-timeframe and volatility regime features
    "returns_1h",              # 1-bar close-to-close return
    "returns_4h",              # 4-bar return (4H momentum on 1H data)
    "returns_24h",             # 24-bar return (daily momentum)
    "volatility_ratio",        # short vol / long vol (regime shift)
    "adx_rsi_interaction",     # adx * abs(rsi - 50) / 50 — trend + momentum
    "close_vs_bb_mid",         # (close - bb_mid) / atr — position in range
]

BASE_FEATURES = [
    "adx", "rsi", "atr", "di_pos", "di_neg", "macd_hist",
]

FEATURE_COLUMNS = BASE_FEATURES + DERIVED_FEATURES


def _add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Ergänzt den DataFrame um abgeleitete kausale Features."""
    f = frame.copy()
    close = f["close"]

    f["ema_fast_mid_spread"] = (f["ema_fast"] - f["ema_mid"]) / f["ema_mid"]
    f["ema_mid_slow_spread"] = (f["ema_mid"] - f["ema_slow"]) / f["ema_slow"]
    f["atr_pct"] = f["atr"] / close * 100
    f["donchian_width_norm"] = (f["donchian_high"] - f["donchian_low"]) / close
    f["rolling_volatility_14"] = close.pct_change().rolling(14).std()
    f["adx_slope_5"] = f["adx"].diff(5)
    f["rsi_momentum_5"] = f["rsi"].diff(5)
    f["di_spread"] = f["di_pos"] - f["di_neg"]
    f["volume_ratio"] = f["volume"] / f["volume_sma"]

    # Cross-timeframe momentum
    f["returns_1h"] = close.pct_change(1)
    f["returns_4h"] = close.pct_change(4)
    f["returns_24h"] = close.pct_change(24)

    # Volatility regime: ratio of short-term to long-term volatility.
    # When short vol >> long vol, a regime shift may be starting.
    short_vol = close.pct_change().rolling(7).std()
    long_vol = close.pct_change().rolling(28).std()
    f["volatility_ratio"] = short_vol / long_vol.replace(0, float("nan"))

    # Interaction: strong ADX + extreme RSI = high-conviction trend
    f["adx_rsi_interaction"] = f["adx"] * (f["rsi"] - 50).abs() / 50

    # Position relative to Bollinger midline, normalised by ATR
    bb_mid = close.rolling(20).mean()
    f["close_vs_bb_mid"] = (close - bb_mid) / f["atr"].replace(0, float("nan"))

    return f


def _extract_feature_row(frame: pd.DataFrame, index: int = -1) -> dict:
    """Extrahiert einen einzelnen Feature-Vektor als dict."""
    i = index if index >= 0 else len(frame) + index
    if i < 0 or i >= len(frame):
        return _empty_features()
    row = frame.iloc[i]
    return {col: float(row[col]) if pd.notna(row[col]) else 0.0
            for col in FEATURE_COLUMNS}


def _empty_features() -> dict:
    return {col: 0.0 for col in FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Kausalitätstest (Anforderung 6.4)
# ---------------------------------------------------------------------------
def causality_test(df: pd.DataFrame) -> bool:
    """
    Automatisierter Kausalitätstest: Bestätigt, dass kein Feature
    zum Zeitpunkt t Daten aus Zeitpunkten > t verwendet.

    Methode: Einen Wert in der Zukunft manipulieren und prüfen,
    dass Features der Vergangenheit sich nicht ändern.
    """
    if len(df) < 50:
        return False

    frame_original = prepare_indicators(df.copy())
    frame_original = _add_derived_features(frame_original)
    original_features = _extract_feature_row(frame_original, index=-10)

    # Manipuliere die letzten 5 Zeilen
    df_modified = df.copy()
    df_modified.iloc[-5:, df_modified.columns.get_loc("close")] *= 2.0
    df_modified.iloc[-5:, df_modified.columns.get_loc("high")] *= 2.0
    df_modified.iloc[-5:, df_modified.columns.get_loc("volume")] *= 10.0

    frame_modified = prepare_indicators(df_modified)
    frame_modified = _add_derived_features(frame_modified)
    modified_features = _extract_feature_row(frame_modified, index=-10)

    # Features bei Index -10 dürfen sich nicht ändern
    for key in FEATURE_COLUMNS:
        if abs(original_features[key] - modified_features[key]) > 1e-10:
            return False
    return True
