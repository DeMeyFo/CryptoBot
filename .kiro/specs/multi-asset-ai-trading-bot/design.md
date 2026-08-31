# Design Document: ML-Meta-Strategie-Schicht für CryptoBot

## Overview

Dieses Design beschreibt die Architektur einer ML-gestützten Meta-Strategie-Schicht, die als intelligenter Gatekeeper zwischen der validierten trend-breakout-v3-Strategie (`strategy.py`) und der Orderausführung in der Trading-Loop (`main.py`) agiert. Der ML-Layer entscheidet, WANN die bestehende Strategie handeln darf, und skaliert Positionsgrößen proportional zur ML-Konfidenz. Die Kernstrategie-Logik bleibt unangetastet.

Die Architektur folgt dem bestehenden Prozessmodell: ein einziger Python-Prozess, SQLite für Persistenz, Konfiguration über `os.getenv()` mit Defaults. Der ML-Layer ist vollständig opt-in (`ML_GATE_ENABLED=false` als Default) und fällt bei jedem Fehler auf das Verhalten ohne ML zurück.

## Architecture

### Komponentendiagramm

```
┌──────────────────────────────────────────────────────────────────┐
│  main.py – Trading Loop                                          │
│                                                                  │
│  scan_new_entries()                                              │
│    │                                                             │
│    ├─ analyze_symbol(symbol) ──► strategy.py (unverändert)       │
│    │    returns: action, ta_score, indicators, atr, timestamp    │
│    │                                                             │
│    ├─ ml_gate.evaluate(symbol, analysis) ──► ml_gate.py [NEU]    │
│    │    │                                                        │
│    │    ├─ ml_features.build_features(symbol) ──► ml_features.py │
│    │    ├─ RegimeClassifier.predict(features) ──► ml_trainer.py  │
│    │    ├─ LLMRegimeAdvisor.assess(indicators) ──► ml_gate.py    │
│    │    └─ returns: GateDecision(action, risk_scale, confidence) │
│    │                                                             │
│    ├─ ml_prediction_log.record(prediction) ──► ml_prediction_log │
│    │                                                             │
│    └─ calculate_position_notional(..., risk_scale=decision.risk_scale)
│         ──► risk_manager.py (unverändert)                        │
│                                                                  │
│  exchange_adapter.py ◄── abstrakte Schnittstelle                 │
│    ├─ BitgetAdapter (kapselt bitget_client.py)                   │
│    └─ [MT5Adapter] (spätere Erweiterung)                         │
└──────────────────────────────────────────────────────────────────┘
```

### Datenfluss: Entry-Signal mit ML-Gate

1. `scan_new_entries()` ruft `analyze_symbol(symbol)` auf (unverändert).
2. Wenn `analysis["action"]` nicht `"hold"` ist, ruft die Trading-Loop `ml_gate.evaluate()` auf.
3. `ml_gate.evaluate()` baut Features über `ml_features.build_features()`, fragt den `RegimeClassifier` und optional den `LLMRegimeAdvisor`, berechnet den gewichteten `Confidence_Score` und gibt eine `GateDecision` zurück.
4. Bei `GateDecision.action == "block"` wird das Signal unterdrückt und protokolliert.
5. Bei `GateDecision.action == "allow"` wird `risk_scale` an `calculate_position_notional()` übergeben.
6. `ml_prediction_log.record()` persistiert die Vorhersage in SQLite.
7. Wenn `ML_GATE_ENABLED=false`, gibt `ml_gate.evaluate()` sofort `GateDecision(action="allow", risk_scale=1.0, confidence=1.0)` zurück — identisch zu keinem ML-Layer.

## Components and Interfaces

### 1. ml_gate.py — ML-Gate Orchestrator

Zentrale Fassade, die den gesamten ML-Entscheidungspfad kapselt. Kein anderes Modul muss die einzelnen ML-Komponenten kennen.

```python
# ml_gate.py

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration (Anforderung 9)
# ---------------------------------------------------------------------------
ML_GATE_ENABLED = os.getenv("ML_GATE_ENABLED", "false").lower() == "true"
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.5"))
ML_MIN_RISK_SCALE = float(os.getenv("ML_MIN_RISK_SCALE", "0.5"))
ML_GB_WEIGHT = float(os.getenv("ML_GB_WEIGHT", "0.7"))
ML_LLM_WEIGHT = float(os.getenv("ML_LLM_WEIGHT", "0.3"))
ML_LLM_ENABLED = os.getenv("ML_LLM_ENABLED", "false").lower() == "true"
ML_MODEL_DIR = os.getenv("ML_MODEL_DIR", ".models/")
ML_RETRAIN_INTERVAL_DAYS = int(os.getenv("ML_RETRAIN_INTERVAL_DAYS", "30"))

_DISCLAIMER = (
    "ML-Gate ist ein experimentelles Filter-System. "
    "Vergangene Backtest-Ergebnisse garantieren keine zukünftige Performance."
)


def validate_config() -> None:
    """Prüft ML-Konfiguration beim Start. Wirft ValueError bei Ungültigkeit."""
    if ML_CONFIDENCE_THRESHOLD < 0.0 or ML_CONFIDENCE_THRESHOLD > 1.0:
        raise ValueError(
            f"ML_CONFIDENCE_THRESHOLD muss in [0.0, 1.0] liegen, "
            f"ist {ML_CONFIDENCE_THRESHOLD}"
        )
    if ML_MIN_RISK_SCALE < 0.0 or ML_MIN_RISK_SCALE > 1.0:
        raise ValueError(
            f"ML_MIN_RISK_SCALE muss in [0.0, 1.0] liegen, "
            f"ist {ML_MIN_RISK_SCALE}"
        )
    if ML_GB_WEIGHT < 0.0 or ML_LLM_WEIGHT < 0.0:
        raise ValueError("ML_GB_WEIGHT und ML_LLM_WEIGHT dürfen nicht negativ sein")
    if abs((ML_GB_WEIGHT + ML_LLM_WEIGHT) - 1.0) > 1e-9:
        raise ValueError(
            f"ML_GB_WEIGHT + ML_LLM_WEIGHT müssen 1.0 ergeben, "
            f"ist {ML_GB_WEIGHT + ML_LLM_WEIGHT}"
        )
    if ML_RETRAIN_INTERVAL_DAYS < 1:
        raise ValueError(
            f"ML_RETRAIN_INTERVAL_DAYS muss >= 1 sein, "
            f"ist {ML_RETRAIN_INTERVAL_DAYS}"
        )


# ---------------------------------------------------------------------------
# Datenmodell
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateDecision:
    """Ergebnis einer ML-Gate-Evaluation."""
    action: str            # "allow" | "block" | "hold"
    risk_scale: float      # 0.0 – 1.0, Skalierungsfaktor für Positionsgröße
    confidence: float      # 0.0 – 1.0, kombinierter Confidence_Score
    regime: str            # "trending" | "choppy" | "neutral"
    filter_reason: str     # "" wenn allow, sonst Grund
    gb_probability: float  # Regime_Classifier-Rohwert
    llm_confidence: float | None  # LLM-Konfidenz oder None
    llm_regime: str | None        # LLM-Regime oder None
    model_version: str     # Modellversion des geladenen Classifiers
    latency_ms: float      # Gesamtdauer der Gate-Evaluation


# ---------------------------------------------------------------------------
# Bypass-Entscheidung (ML deaktiviert oder Fallback)
# ---------------------------------------------------------------------------
_BYPASS = GateDecision(
    action="allow",
    risk_scale=1.0,
    confidence=1.0,
    regime="trending",
    filter_reason="",
    gb_probability=0.5,
    llm_confidence=None,
    llm_regime=None,
    model_version="none",
    latency_ms=0.0,
)


# ---------------------------------------------------------------------------
# Kernlogik
# ---------------------------------------------------------------------------
_classifiers: dict[str, "RegimeClassifier"] = {}  # pro Symbol


def init(symbols: list[str]) -> None:
    """
    Lädt Modelle beim Bot-Start. Bei Fehler: Fallback-Modus.
    Wird aus main.py nach init_db() aufgerufen.
    """
    if not ML_GATE_ENABLED:
        logger.info("ML-Gate: deaktiviert (ML_GATE_ENABLED=false)")
        return
    validate_config()
    logger.info(_DISCLAIMER)
    _log_config_banner()
    for symbol in symbols:
        try:
            from ml_trainer import RegimeClassifier
            classifier = RegimeClassifier.load_latest(symbol, ML_MODEL_DIR)
            _classifiers[symbol] = classifier
            logger.info(
                f"ML-Gate: Modell geladen für {symbol}: "
                f"version={classifier.metadata.get('version', 'unknown')} "
                f"path={classifier.model_path}"
            )
        except Exception as exc:
            logger.warning(
                f"ML-Gate: Modell für {symbol} nicht ladbar ({exc}); "
                "Fallback auf Default-Werte"
            )


def evaluate(symbol: str, analysis: dict) -> GateDecision:
    """
    Evaluiert ein Entry-Signal durch den ML-Layer.
    Fängt ALLE Exceptions ab und gibt im Fehlerfall _BYPASS zurück.

    Anforderung 15.4: Keine Exception darf die Trading_Loop unterbrechen.
    """
    if not ML_GATE_ENABLED:
        return _BYPASS

    start_time = time.monotonic()
    try:
        return _evaluate_inner(symbol, analysis, start_time)
    except Exception as exc:
        elapsed = (time.monotonic() - start_time) * 1000
        logger.error(
            f"ML-Gate: Exception bei {symbol}, Fallback auf Bypass: {exc}",
            exc_info=True,
        )
        return GateDecision(
            action="allow", risk_scale=1.0, confidence=0.5,
            regime="trending", filter_reason="ml_exception_fallback",
            gb_probability=0.5, llm_confidence=None, llm_regime=None,
            model_version="fallback", latency_ms=elapsed,
        )


def _evaluate_inner(symbol: str, analysis: dict, start_time: float) -> GateDecision:
    """Interne Gate-Logik ohne Fallback-Wrapping."""
    from ml_features import build_features

    # 1) Features berechnen
    features = build_features(symbol, analysis.get("indicators", {}))

    # 2) Gradient-Boosting-Vorhersage
    classifier = _classifiers.get(symbol)
    if classifier is None:
        # Anforderung 3.5: kein Modell → Default
        gb_regime = "trending"
        gb_probability = 0.5
        model_version = "no_model"
    else:
        try:
            gb_result = classifier.predict(features)
            gb_regime = gb_result["regime"]
            gb_probability = gb_result["probability"]
            model_version = classifier.metadata.get("version", "unknown")
        except Exception as exc:
            # Anforderung 15.1: Inferenz-Exception → Default
            logger.warning(f"ML-Gate: Classifier-Exception für {symbol}: {exc}")
            gb_regime = "trending"
            gb_probability = 0.5
            model_version = "fallback"

    # 3) LLM-Regime-Advisor (optional)
    llm_confidence = None
    llm_regime = None
    if ML_LLM_ENABLED:
        try:
            llm_result = _query_llm_advisor(symbol, analysis.get("indicators", {}))
            llm_confidence = llm_result["confidence"]
            llm_regime = llm_result["regime"]
        except Exception as exc:
            # Anforderung 15.2 / 4.4: LLM-Fehler → nur GB verwenden
            logger.warning(f"ML-Gate: LLM-Exception für {symbol}: {exc}")

    # 4) Gewichteter Confidence_Score
    if llm_confidence is not None:
        confidence = ML_GB_WEIGHT * gb_probability + ML_LLM_WEIGHT * llm_confidence
    else:
        confidence = gb_probability

    # 5) Gate-Entscheidung
    regime = gb_regime
    if regime == "choppy" and confidence < ML_CONFIDENCE_THRESHOLD:
        action = "block"
        filter_reason = "ml_regime_blocked"
        risk_scale = 0.0
    else:
        action = "allow"
        filter_reason = ""
        risk_scale = _compute_risk_scale(confidence)

    elapsed = (time.monotonic() - start_time) * 1000
    return GateDecision(
        action=action,
        risk_scale=risk_scale,
        confidence=confidence,
        regime=regime,
        filter_reason=filter_reason,
        gb_probability=gb_probability,
        llm_confidence=llm_confidence,
        llm_regime=llm_regime,
        model_version=model_version,
        latency_ms=elapsed,
    )


def _compute_risk_scale(confidence: float) -> float:
    """
    Anforderung 2.1: Risk_Scale linear zwischen ML_MIN_RISK_SCALE und 1.0,
    proportional zum Confidence_Score.
    """
    scale = ML_MIN_RISK_SCALE + (1.0 - ML_MIN_RISK_SCALE) * confidence
    return max(ML_MIN_RISK_SCALE, min(1.0, scale))


def _query_llm_advisor(symbol: str, indicators: dict) -> dict:
    """
    Anforderung 4: Claude-LLM-Regime-Advisor.
    Gibt {"regime": str, "confidence": float} zurück.
    """
    from ml_gate_llm import LLMRegimeAdvisor
    advisor = LLMRegimeAdvisor()
    return advisor.assess(symbol, indicators)


def rollback_model(symbol: str, version: str) -> bool:
    """Anforderung 12.4: Rollback auf eine frühere Modellversion."""
    from ml_trainer import RegimeClassifier
    try:
        classifier = RegimeClassifier.load_version(symbol, version, ML_MODEL_DIR)
        _classifiers[symbol] = classifier
        logger.info(f"ML-Gate: Rollback für {symbol} auf Version {version}")
        return True
    except Exception as exc:
        logger.error(f"ML-Gate: Rollback fehlgeschlagen für {symbol}: {exc}")
        return False


def _log_config_banner() -> None:
    """Anforderung 9.4 / 13.3: ML-Konfiguration im Startbanner."""
    logger.info("--- ML-Gate Konfiguration ---")
    logger.info(f"  ML_CONFIDENCE_THRESHOLD: {ML_CONFIDENCE_THRESHOLD}")
    logger.info(f"  ML_MIN_RISK_SCALE:       {ML_MIN_RISK_SCALE}")
    logger.info(f"  ML_GB_WEIGHT:            {ML_GB_WEIGHT}")
    logger.info(f"  ML_LLM_WEIGHT:           {ML_LLM_WEIGHT}")
    logger.info(f"  ML_LLM_ENABLED:          {ML_LLM_ENABLED}")
    logger.info(f"  ML_MODEL_DIR:            {ML_MODEL_DIR}")
    logger.info(f"  ML_RETRAIN_INTERVAL_DAYS:{ML_RETRAIN_INTERVAL_DAYS}")
    logger.info("-----------------------------")
```

### 2. ml_features.py — Kausale Feature-Pipeline

Erweitert die bestehende `prepare_indicators()` um abgeleitete Features für das ML-Modell. Alle Features sind kausal: zum Zeitpunkt t werden ausschließlich Daten aus t und früheren Zeitpunkten verwendet.

```python
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
    "adx_slope_5",             # adx.diff(5) – ADX-Trend über 5 Perioden
    "rsi_momentum_5",          # rsi.diff(5) – RSI-Momentum
    "di_spread",               # di_pos - di_neg
    "volume_ratio",            # volume / volume_sma
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
```

### 3. ml_trainer.py — Walk-Forward-Training und Regime-Klassifikator

Implementiert das Training nach Walk-Forward-Protokoll und die `RegimeClassifier`-Klasse für Inferenz.

```python
# ml_trainer.py

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label-Generierung (Anforderung 5.7)
# ---------------------------------------------------------------------------
ML_LABEL_HORIZON = int(os.getenv("ML_LABEL_HORIZON", "24"))
ML_LABEL_ATR_THRESHOLD = float(os.getenv("ML_LABEL_ATR_THRESHOLD", "1.5"))


def generate_labels(df: pd.DataFrame, horizon: int = ML_LABEL_HORIZON,
                    atr_threshold: float = ML_LABEL_ATR_THRESHOLD) -> pd.Series:
    """
    Label: trending (1) wenn die max. Preisbewegung in den nächsten
    `horizon` Candles > atr_threshold * ATR ist, sonst choppy (0).

    Verwendet nur Daten der folgenden Candles → nicht kausal für Features,
    aber korrekt als Label (supervised signal).
    """
    close = df["close"]
    atr = df["atr"]
    future_max_move = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - horizon):
        window = close.iloc[i + 1:i + 1 + horizon]
        max_move = (window.max() - window.min()) / close.iloc[i]
        future_max_move.iloc[i] = max_move
    label = (future_max_move > atr_threshold * atr / close).astype(float)
    return label


# ---------------------------------------------------------------------------
# Walk-Forward-Split (Anforderung 5.1, 5.2, 6.3)
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardSplit:
    train_start: int      # Index im DataFrame
    train_end: int
    val_start: int
    val_end: int


def walk_forward_splits(n_samples: int, train_ratio: float = 0.7,
                        n_splits: int = 1) -> list[WalkForwardSplit]:
    """
    Erzeugt chronologisch geordnete Train/Val-Splits.
    Validierung folgt zeitlich auf Training, ohne Überlappung.
    """
    splits = []
    split_size = n_samples // n_splits
    for i in range(n_splits):
        start = i * split_size
        end = min(start + split_size, n_samples)
        boundary = start + int((end - start) * train_ratio)
        splits.append(WalkForwardSplit(
            train_start=start,
            train_end=boundary,
            val_start=boundary,
            val_end=end,
        ))
    return splits


# ---------------------------------------------------------------------------
# RegimeClassifier (Anforderung 3.1, 3.4, 3.5)
# ---------------------------------------------------------------------------
class RegimeClassifier:
    """Gradient-Boosting-Binärklassifikator für Marktregime."""

    def __init__(self, model, feature_columns: list[str], metadata: dict):
        self.model = model
        self.feature_columns = feature_columns
        self.metadata = metadata
        self.model_path = metadata.get("path", "unknown")

    def predict(self, features: dict) -> dict:
        """
        Anforderung 3.4: Gibt Klassifikation und Wahrscheinlichkeit zurück.
        """
        import numpy as np
        feature_array = np.array(
            [[features.get(col, 0.0) for col in self.feature_columns]]
        )
        proba = self.model.predict_proba(feature_array)[0]
        trending_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        regime = "trending" if trending_prob >= 0.5 else "choppy"
        return {
            "regime": regime,
            "probability": trending_prob,
        }

    @classmethod
    def load_latest(cls, symbol: str, model_dir: str) -> "RegimeClassifier":
        """Lädt das neueste Modell für ein Symbol aus model_dir."""
        base = Path(model_dir)
        candidates = sorted(
            [d for d in base.iterdir()
             if d.is_dir() and d.name.startswith(f"{symbol}_")],
            key=lambda d: d.name,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"Kein Modell für {symbol} in {model_dir}"
            )
        return cls._load_from_dir(candidates[0])

    @classmethod
    def load_version(cls, symbol: str, version: str,
                     model_dir: str) -> "RegimeClassifier":
        """Lädt eine spezifische Modellversion (Anforderung 12.4)."""
        target = Path(model_dir) / f"{symbol}_{version}"
        if not target.is_dir():
            raise FileNotFoundError(f"Modellversion {version} nicht gefunden: {target}")
        return cls._load_from_dir(target)

    @classmethod
    def _load_from_dir(cls, path: Path) -> "RegimeClassifier":
        model_file = path / "model.pkl"
        meta_file = path / "metadata.json"
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        with open(meta_file) as f:
            metadata = json.load(f)
        metadata["path"] = str(path)
        return cls(
            model=model,
            feature_columns=metadata.get("feature_columns", []),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# WalkForwardTrainer (Anforderung 5, 12, 13)
# ---------------------------------------------------------------------------
class WalkForwardTrainer:
    """Walk-Forward-Training mit Modell-Persistenz und Validierung."""

    def __init__(self, model_dir: str = ".models/"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self, symbol: str, df: pd.DataFrame,
              feature_columns: list[str]) -> dict:
        """
        Führt einen vollständigen Walk-Forward-Trainingszyklus durch.

        Returns: dict mit Validierungsmetriken und Modellpfad.
        """
        from ml_features import build_feature_matrix
        from xgboost import XGBClassifier

        frame = build_feature_matrix(df)
        labels = generate_labels(df)

        # NaN-Zeilen entfernen
        valid_mask = frame.notna().all(axis=1) & labels.notna()
        frame = frame[valid_mask]
        labels = labels[valid_mask]

        if len(frame) < 100:
            raise ValueError(f"Zu wenige Datenpunkte: {len(frame)}")

        splits = walk_forward_splits(len(frame))
        split = splits[-1]

        X_train = frame.iloc[split.train_start:split.train_end]
        y_train = labels.iloc[split.train_start:split.train_end]
        X_val = frame.iloc[split.val_start:split.val_end]
        y_val = labels.iloc[split.val_start:split.val_end]

        # Anforderung 5.2: Hyperparameter-Suche nur auf Trainingsblock
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Anforderung 5.5: Validierungsmetriken
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_val, y_proba)),
        }

        # Anforderung 5.6: Vergleich mit aktuellem Modell
        current_auc = self._current_model_auc(symbol)
        if current_auc is not None and metrics["auc_roc"] < current_auc:
            logger.warning(
                f"Historisch gemessen: Neues Modell (AUC={metrics['auc_roc']:.4f}) "
                f"schlechter als aktuelles (AUC={current_auc:.4f}). "
                f"Bestehendes Modell beibehalten."
            )
            return {"status": "rejected", "metrics": metrics, "reason": "auc_regression"}

        # Anforderung 13.4: Metriken-Prefix
        logger.info(
            f"Historisch gemessen: {symbol} Walk-Forward-Validierung: "
            f"Accuracy={metrics['accuracy']:.4f} "
            f"AUC-ROC={metrics['auc_roc']:.4f} "
            f"F1={metrics['f1_score']:.4f}"
        )

        # Anforderung 12.1, 12.2: Persistenz
        model_path = self._save_model(symbol, model, feature_columns, metrics, df)

        return {
            "status": "accepted",
            "metrics": metrics,
            "model_path": str(model_path),
        }

    def historical_predictions(self, symbol: str, df: pd.DataFrame,
                               feature_columns: list[str]) -> list[tuple]:
        """
        Anforderung 11.1: Historische Regime-Vorhersagen für Backtest.
        Returns: [(timestamp, regime, confidence_score), ...]
        """
        from ml_features import build_feature_matrix
        classifier = RegimeClassifier.load_latest(symbol, str(self.model_dir))
        features = build_feature_matrix(df)
        predictions = []
        for i in range(len(features)):
            row = features.iloc[i]
            if row.isna().any():
                continue
            feature_dict = row.to_dict()
            result = classifier.predict(feature_dict)
            timestamp = df.iloc[i]["timestamp"]
            predictions.append((
                timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                result["regime"],
                result["probability"],
            ))
        return predictions

    def _save_model(self, symbol: str, model, feature_columns: list[str],
                    metrics: dict, df: pd.DataFrame) -> Path:
        """Anforderung 12.1, 12.2: Modell + Metadaten speichern."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dir_name = f"{symbol}_{timestamp}"
        model_dir = self.model_dir / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        data_hash = hashlib.sha256(
            df.to_csv(index=False).encode()
        ).hexdigest()[:16]

        metadata = {
            "symbol": symbol,
            "version": timestamp,
            "feature_columns": feature_columns,
            "training_period": {
                "start": str(df.iloc[0]["timestamp"]),
                "end": str(df.iloc[-1]["timestamp"]),
            },
            "validation_metrics": metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_hash": data_hash,
            "hyperparameters": model.get_params(),
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return model_dir

    def _current_model_auc(self, symbol: str) -> float | None:
        """Lädt die AUC-ROC des aktuellen Modells für den Vergleich."""
        try:
            classifier = RegimeClassifier.load_latest(symbol, str(self.model_dir))
            return classifier.metadata.get("validation_metrics", {}).get("auc_roc")
        except FileNotFoundError:
            return None
```

### 4. ml_gate_llm.py — Claude-LLM Regime-Advisor

```python
# ml_gate_llm.py

import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
ML_LLM_TIMEOUT = int(os.getenv("ML_LLM_TIMEOUT", "10"))


class LLMRegimeAdvisor:
    """
    Anforderung 4: Qualitative Markteinschätzung über Claude-API.
    Übergibt ausschließlich aggregierte Indikator-Zusammenfassungen.
    """

    SYSTEM_PROMPT = (
        "Du bist ein Marktregime-Analyst. Analysiere die folgenden technischen "
        "Indikatoren und klassifiziere das Marktregime als trending, choppy "
        "oder neutral. Gib eine Konfidenz zwischen 0.0 und 1.0 an. "
        "Antworte ausschließlich im JSON-Format: "
        '{"regime": "trending|choppy|neutral", "confidence": 0.0-1.0, '
        '"reasoning": "kurze Begründung"}'
    )

    def assess(self, symbol: str, indicators: dict) -> dict:
        """
        Anforderung 4.1, 4.2: Nur aggregierte Indikatoren, keine Roh-OHLCV.
        Anforderung 4.6: Antwortzeit messen und protokollieren.
        """
        summary = self._build_indicator_summary(symbol, indicators)

        start = time.monotonic()
        try:
            response = self._call_claude(summary)
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(f"LLM-Advisor: {symbol} Antwortzeit={elapsed_ms:.0f}ms")

        return self._parse_response(response)

    def _build_indicator_summary(self, symbol: str, indicators: dict) -> str:
        """Anforderung 4.2: Nur aggregierte Zusammenfassungen."""
        return (
            f"Symbol: {symbol}\n"
            f"ADX: {indicators.get('adx', 'N/A')}\n"
            f"RSI: {indicators.get('rsi', 'N/A')}\n"
            f"EMA20: {indicators.get('ema20', 'N/A')}\n"
            f"EMA50: {indicators.get('ema50', 'N/A')}\n"
            f"EMA200: {indicators.get('ema200', 'N/A')}\n"
            f"DI+: {indicators.get('di_pos', 'N/A')}\n"
            f"DI-: {indicators.get('di_neg', 'N/A')}\n"
            f"ATR: {indicators.get('atr', 'N/A')}\n"
            f"Volume Ratio: {indicators.get('volume_ratio', 'N/A')}\n"
            f"MACD Histogram: {indicators.get('macd_hist', 'N/A')}\n"
            f"Donchian High: {indicators.get('donchian_high', 'N/A')}\n"
            f"Donchian Low: {indicators.get('donchian_low', 'N/A')}\n"
            f"Current Price: {indicators.get('current_price', 'N/A')}"
        )

    def _call_claude(self, summary: str) -> dict:
        """HTTP-Aufruf an die Claude-API."""
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 256,
                "system": self.SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": summary}],
            },
            timeout=ML_LLM_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, api_response: dict) -> dict:
        """Parst die Claude-Antwort und extrahiert Regime + Konfidenz."""
        content = api_response.get("content", [{}])
        text = content[0].get("text", "{}") if content else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Ungültige LLM-Antwort: {text[:200]}")

        regime = parsed.get("regime", "neutral")
        if regime not in ("trending", "choppy", "neutral"):
            regime = "neutral"
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        return {"regime": regime, "confidence": confidence}
```

### 5. ml_prediction_log.py — Vorhersage-Logging

```python
# ml_prediction_log.py

import json
import logging
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH

logger = logging.getLogger(__name__)


def _conn():
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_ml_tables():
    """
    Anforderung 14.1, 14.2, 14.3, 14.4:
    Neue Tabellen anlegen, bestehende nicht modifizieren.
    Wird aus init_db() aufgerufen.
    """
    with _conn() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol              TEXT    NOT NULL,
                prediction_at       TEXT    NOT NULL,
                predicted_regime    TEXT    NOT NULL CHECK(
                    predicted_regime IN ('trending', 'choppy', 'neutral')
                ),
                confidence_score    REAL    NOT NULL,
                gb_probability      REAL    NOT NULL,
                llm_confidence      REAL,
                llm_regime          TEXT,
                top_features        TEXT,
                action              TEXT    NOT NULL CHECK(
                    action IN ('allow', 'block', 'hold')
                ),
                risk_scale          REAL    NOT NULL,
                filter_reason       TEXT,
                model_version       TEXT,
                training_period     TEXT,
                latency_ms          REAL,
                actual_regime       TEXT,
                actual_performance  REAL,
                outcome_updated_at  TEXT
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS ml_model_metadata (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol              TEXT    NOT NULL,
                version             TEXT    NOT NULL,
                model_path          TEXT    NOT NULL,
                training_start      TEXT,
                training_end        TEXT,
                validation_start    TEXT,
                validation_end      TEXT,
                feature_columns     TEXT,
                hyperparameters     TEXT,
                accuracy            REAL,
                precision_score     REAL,
                recall              REAL,
                f1_score            REAL,
                auc_roc             REAL,
                data_hash           TEXT,
                trained_at          TEXT    NOT NULL,
                is_active           INTEGER DEFAULT 0,
                UNIQUE(symbol, version)
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_time
            ON ml_predictions(symbol, prediction_at)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_predictions_outcome
            ON ml_predictions(outcome_updated_at)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_model_metadata_symbol
            ON ml_model_metadata(symbol, is_active)
        """)
        connection.commit()


def record(decision, symbol: str, top_features: dict | None = None) -> int:
    """
    Anforderung 7.1: Persistiert eine ML-Vorhersage.
    Returns: prediction_id für die Referenz in der signals-Tabelle.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as connection:
        cursor = connection.execute("""
            INSERT INTO ml_predictions
                (symbol, prediction_at, predicted_regime, confidence_score,
                 gb_probability, llm_confidence, llm_regime, top_features,
                 action, risk_scale, filter_reason, model_version,
                 latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, now, decision.regime, decision.confidence,
            decision.gb_probability, decision.llm_confidence, decision.llm_regime,
            json.dumps(top_features) if top_features else None,
            decision.action, decision.risk_scale, decision.filter_reason,
            decision.model_version, decision.latency_ms,
        ))
        connection.commit()
        return cursor.lastrowid


def update_outcome(prediction_id: int, actual_regime: str,
                   actual_performance: float) -> None:
    """Anforderung 7.2: Tatsächliches Ergebnis nachträglich ergänzen."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as connection:
        connection.execute("""
            UPDATE ml_predictions
            SET actual_regime=?, actual_performance=?, outcome_updated_at=?
            WHERE id=?
        """, (actual_regime, actual_performance, now, prediction_id))
        connection.commit()


def query_accuracy(start_date: str, end_date: str) -> dict:
    """
    Anforderung 7.3: Vorhersagegenauigkeit für einen Zeitraum.
    """
    with _conn() as connection:
        rows = connection.execute("""
            SELECT predicted_regime, actual_regime, action, confidence_score
            FROM ml_predictions
            WHERE prediction_at BETWEEN ? AND ?
              AND actual_regime IS NOT NULL
        """, (start_date, end_date)).fetchall()

    if not rows:
        return {"total": 0, "accuracy": 0.0, "block_rate": 0.0}

    total = len(rows)
    correct = sum(1 for row in rows if row[0] == row[1])
    blocked = sum(1 for row in rows if row[2] == "block")

    all_rows = _conn().execute("""
        SELECT COUNT(*) FROM ml_predictions
        WHERE prediction_at BETWEEN ? AND ?
    """, (start_date, end_date)).fetchone()
    total_with_pending = all_rows[0] if all_rows else total

    return {
        "total": total,
        "total_with_pending": total_with_pending,
        "accuracy": correct / total if total > 0 else 0.0,
        "block_rate": blocked / total_with_pending if total_with_pending > 0 else 0.0,
    }
```

### 6. exchange_adapter.py — Exchange-Adapter-Abstraktion

```python
# exchange_adapter.py

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExchangeAdapter(ABC):
    """
    Anforderung 10.1: Abstrakte Schnittstelle für Marktdaten und Orderausführung.
    Ermöglicht spätere Erweiterung auf Forex/MT5.
    """

    @abstractmethod
    def get_candles(self, symbol: str, granularity: str, limit: int) -> list:
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def place_order(self, symbol: str, side: str, size_usdt: float,
                    leverage: int, current_price: float) -> dict:
        ...

    @abstractmethod
    def close_order(self, symbol: str, side: str, quantity: float,
                    observed_price: float) -> dict:
        ...

    @abstractmethod
    def get_account_info(self) -> dict:
        ...

    @abstractmethod
    def get_top_symbols(self, count: int) -> list[str]:
        ...


class BitgetAdapter(ExchangeAdapter):
    """
    Anforderung 10.2: Konkrete Implementierung für Bitget.
    Kapselt die bestehenden Funktionen aus bitget_client.py.
    """

    def get_candles(self, symbol: str, granularity: str, limit: int) -> list:
        from bitget_client import get_candles
        return get_candles(symbol, granularity, limit)

    def get_current_price(self, symbol: str) -> float:
        from bitget_client import get_current_price
        return get_current_price(symbol)

    def place_order(self, symbol: str, side: str, size_usdt: float,
                    leverage: int, current_price: float) -> dict:
        from bitget_client import place_order
        return place_order(symbol, side, size_usdt, leverage, current_price)

    def close_order(self, symbol: str, side: str, quantity: float,
                    observed_price: float) -> dict:
        from bitget_client import close_order
        return close_order(symbol, side, quantity, observed_price)

    def get_account_info(self) -> dict:
        from bitget_client import get_account_info
        return get_account_info()

    def get_top_symbols(self, count: int) -> list[str]:
        from bitget_client import get_top_symbols
        return get_top_symbols(count)


# Anforderung 10.4: Adapter-Auswahl via Umgebungsvariable
EXCHANGE_ADAPTER = os.getenv("EXCHANGE_ADAPTER", "bitget")

_ADAPTERS = {
    "bitget": BitgetAdapter,
}


def get_adapter() -> ExchangeAdapter:
    """Factory: Erzeugt den konfigurierten Exchange-Adapter."""
    adapter_class = _ADAPTERS.get(EXCHANGE_ADAPTER)
    if adapter_class is None:
        raise ValueError(
            f"Unbekannter EXCHANGE_ADAPTER: {EXCHANGE_ADAPTER}. "
            f"Verfügbar: {sorted(_ADAPTERS.keys())}"
        )
    return adapter_class()
```

## Integration in bestehende Module

### Änderungen in main.py

Die einzige signifikante Änderung ist die Einbindung des ML-Gates in `scan_new_entries()`. Die Änderung ist minimal und respektiert die bestehende Kontrollflussstruktur.

```python
# In main.py – scan_new_entries(), nach analyze_symbol():

import ml_gate
import ml_prediction_log

# ... bestehender Code bis analysis = analyze_symbol(symbol) ...

# NEU: ML-Gate-Evaluation
gate_decision = ml_gate.evaluate(symbol, analysis)

if gate_decision.action == "block":
    logger.info(
        f"{symbol}: ML-Gate blockiert ({gate_decision.filter_reason}) "
        f"regime={gate_decision.regime} confidence={gate_decision.confidence:.3f}"
    )
    ml_prediction_log.record(gate_decision, symbol)
    continue

# NEU: risk_scale an Positionsgrößenberechnung übergeben
risk_scale = gate_decision.risk_scale if ml_gate.ML_GATE_ENABLED else 1.0

# ... bestehender Sizing-Code ...
notional = calculate_position_notional(
    equity=equity,
    entry_price=observed_price,
    stop_loss=stop_loss,
    max_notional=_available_notional_cap(account),
    symbol_exposure_usdt=symbol_exposure.get(symbol, 0.0),
    gross_exposure_usdt=gross_exposure,
    risk_scale=risk_scale,  # NEU: ML-skaliert
)

# NEU: Prediction-Logging nach Order-Platzierung
prediction_id = ml_prediction_log.record(gate_decision, symbol)
```

### Änderungen in main.py – Startup

```python
# In main() – nach init_db():

from ml_prediction_log import init_ml_tables
init_ml_tables()

if ml_gate.ML_GATE_ENABLED:
    symbols = get_top_symbols(TOP_COINS_COUNT)
    ml_gate.init(symbols)
```

### Änderungen in database.py

Die bestehende `init_db()` wird um einen Aufruf von `init_ml_tables()` ergänzt. Das bestehende Schema wird nicht modifiziert. Für die `ml_prediction_id`-Referenz in der `signals`-Tabelle (Anforderung 14.5) wird eine neue optionale Spalte über `ALTER TABLE ... ADD COLUMN` hinzugefügt — analog zum bestehenden Migrationsmuster.

```python
# Am Ende von init_db() in database.py:

signal_columns = _cols(connection, "signals")
if "ml_prediction_id" not in signal_columns:
    connection.execute(
        "ALTER TABLE signals ADD COLUMN ml_prediction_id INTEGER"
    )
```

### Änderungen in config.py

Alle ML-spezifischen Umgebungsvariablen werden in `ml_gate.py` gelesen, nicht in `config.py`. Das folgt dem Prinzip der Modul-Autonomie: die ML-Konfiguration lebt dort, wo sie gebraucht wird. `config.py` bleibt unverändert.

## Data Models

### ml_predictions (Anforderung 14.1)

| Spalte | Typ | Beschreibung |
|---|---|---|
| id | INTEGER PK | Auto-Increment |
| symbol | TEXT NOT NULL | Handelssymbol |
| prediction_at | TEXT NOT NULL | UTC ISO-8601 Zeitstempel |
| predicted_regime | TEXT NOT NULL | "trending" \| "choppy" \| "neutral" |
| confidence_score | REAL NOT NULL | Kombinierter Score 0.0–1.0 |
| gb_probability | REAL NOT NULL | Regime_Classifier-Rohwert |
| llm_confidence | REAL | LLM-Konfidenz (NULL wenn deaktiviert) |
| llm_regime | TEXT | LLM-Regime (NULL wenn deaktiviert) |
| top_features | TEXT | JSON der Top-Features mit Werten |
| action | TEXT NOT NULL | "allow" \| "block" \| "hold" |
| risk_scale | REAL NOT NULL | Positionsgrößen-Skalierung |
| filter_reason | TEXT | Filtergrund bei Blockierung |
| model_version | TEXT | Modellversion des Classifiers |
| training_period | TEXT | Trainingszeitraum des Modells |
| latency_ms | REAL | Evaluationsdauer in Millisekunden |
| actual_regime | TEXT | Tatsächliches Regime (nachträglich) |
| actual_performance | REAL | Realisierte Performance (nachträglich) |
| outcome_updated_at | TEXT | Zeitpunkt der Ergebnis-Ergänzung |

### ml_model_metadata (Anforderung 14.2)

| Spalte | Typ | Beschreibung |
|---|---|---|
| id | INTEGER PK | Auto-Increment |
| symbol | TEXT NOT NULL | Handelssymbol |
| version | TEXT NOT NULL | Versionsbezeichner (Timestamp) |
| model_path | TEXT NOT NULL | Pfad zur Modelldatei |
| training_start | TEXT | Beginn Trainingsperiode |
| training_end | TEXT | Ende Trainingsperiode |
| validation_start | TEXT | Beginn Validierungsperiode |
| validation_end | TEXT | Ende Validierungsperiode |
| feature_columns | TEXT | JSON-Liste der Feature-Spalten |
| hyperparameters | TEXT | JSON der Hyperparameter |
| accuracy | REAL | Validierungs-Accuracy |
| precision_score | REAL | Validierungs-Precision |
| recall | REAL | Validierungs-Recall |
| f1_score | REAL | Validierungs-F1 |
| auc_roc | REAL | Validierungs-AUC-ROC |
| data_hash | TEXT | Hash der Trainingsdaten |
| trained_at | TEXT NOT NULL | Trainingszeitstempel |
| is_active | INTEGER | 1 wenn aktuell eingesetzt |

### Modell-Dateisystem (Anforderung 12.2)

```
.models/
├── BTCUSDT_20250101T120000Z/
│   ├── model.pkl
│   └── metadata.json
├── BTCUSDT_20250201T120000Z/
│   ├── model.pkl
│   └── metadata.json
└── ETHUSDT_20250101T120000Z/
    ├── model.pkl
    └── metadata.json
```

## Error Handling

Die Fehlerresilienz folgt dem Prinzip: Der ML-Layer darf den Bot nie zum Absturz bringen. Jede Komponente hat einen definierten Fallback.

| Fehlerfall | Fallback | Anforderung |
|---|---|---|
| Modell nicht ladbar | `ML_GATE_ENABLED` effektiv false | 15.3 |
| Classifier-Inferenz-Exception | Confidence=0.5, Regime=trending, Risk_Scale=1.0 | 15.1 |
| LLM-API nicht erreichbar/Timeout | Nur GB-Wahrscheinlichkeit verwenden | 15.2, 4.4 |
| LLM-API-Antwort ungültig | Nur GB-Wahrscheinlichkeit verwenden | 15.2 |
| Feature-Berechnung fehlerhaft | Bypass (wie ML deaktiviert) | 15.4 |
| ml_prediction_log.record() fehlt | Log-Warnung, Trade fortsetzen | 15.4 |
| Beliebige Exception in evaluate() | Bypass zurückgeben, Exception loggen | 15.4 |

Die äußere `try/except`-Klammer in `ml_gate.evaluate()` stellt sicher, dass keine ML-Exception in die Trading-Loop durchsickert (Anforderung 15.4). Der `_BYPASS`-Wert ist funktional identisch zu `ML_GATE_ENABLED=false`.

## Sicherheitsinvarianten (Anforderung 8)

Der ML-Layer ändert KEINE bestehende Risikologik:

1. **Stop-Loss**: `best_sl_tp()` und `next_atr_trailing_stop()` werden nicht aufgerufen oder modifiziert durch den ML-Layer.
2. **Exposure-Limits**: `exposure_headroom()` läuft unverändert; `risk_scale` reduziert nur den Risk-Budget-Anteil innerhalb von `calculate_position_notional()`.
3. **Daily-Loss-Limit**: `_daily_loss_limit_hit()` wird vor dem ML-Gate geprüft und ist nicht durch ML beeinflussbar.
4. **MAX_OPEN_POSITIONS**: Die Prüfung in `scan_new_entries()` findet vor dem ML-Gate statt.
5. **Exit-Entscheidungen**: Der ML-Layer beeinflusst ausschließlich Entry-Zeitpunkt und Positionsgröße. Trailing-Stops, Stop-Loss-Exits und Position-Management bleiben beim `Risk_Manager`.
6. **DRY_RUN**: Der ML-Layer liest `DRY_RUN` nicht direkt und hat keinen Zugriff auf Order-Funktionen. Die Order-Platzierung bleibt in `main.py`.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: ML-Gate-Bypass bei Deaktivierung

*For any* Entry-Signal der Core_Strategy, wenn `ML_GATE_ENABLED=false` ist, soll `ml_gate.evaluate()` eine `GateDecision` mit `action="allow"` und `risk_scale=1.0` zurückgeben, identisch zum Verhalten ohne ML-Erweiterung.

**Validates: Requirements 1.4, 2.4**

### Property 2: Choppy-Regime-Blockierung

*For any* Symbol und *for any* Confidence_Score unter dem konfigurierten Schwellenwert (`ML_CONFIDENCE_THRESHOLD`), wenn der Regime_Classifier das Regime als "choppy" klassifiziert, soll die `GateDecision.action` den Wert `"block"` haben und `filter_reason` den Wert `"ml_regime_blocked"`.

**Validates: Requirements 1.2**

### Property 3: Trending-Regime-Durchlass

*For any* Symbol und *for any* Confidence_Score über dem konfigurierten Schwellenwert, wenn der Regime_Classifier das Regime als "trending" klassifiziert, soll die `GateDecision.action` den Wert `"allow"` haben.

**Validates: Requirements 1.3**

### Property 4: Risk-Scale-Bereichsinvariante

*For any* Confidence_Score im Bereich [0.0, 1.0], soll der berechnete `risk_scale` immer im Bereich [`ML_MIN_RISK_SCALE`, 1.0] liegen und monoton steigend mit dem Confidence_Score sein.

**Validates: Requirements 2.1**

### Property 5: Bestehende Exposure-Limits als harte Obergrenze

*For any* Kombination von `equity`, `entry_price`, `stop_loss`, `risk_scale` und Exposure-Werten, soll das Ergebnis von `calculate_position_notional()` niemals die bestehenden Exposure-Caps (`MAX_SYMBOL_EXPOSURE_PCT`, `MAX_GROSS_EXPOSURE_PCT`, `MAX_POSITION_PCT`) überschreiten, unabhängig vom `risk_scale`-Wert.

**Validates: Requirements 2.3, 8.1**

### Property 6: Feature-Kausalität

*For any* DataFrame mit OHLCV-Daten und *for any* Zeitpunkt t (Index i) im DataFrame, soll eine Manipulation der Daten an Zeitpunkten > t die Features bei Zeitpunkt t nicht verändern.

**Validates: Requirements 6.1, 6.2, 6.4**

### Property 7: Walk-Forward chronologische Trennung

*For any* Walk-Forward-Split, soll der letzte Zeitpunkt des Trainingsblocks strikt vor dem ersten Zeitpunkt des Validierungsblocks liegen, ohne Überlappung.

**Validates: Requirements 5.1, 6.3**

### Property 8: Gewichtete Confidence-Score-Berechnung

*For any* `gb_probability` in [0.0, 1.0] und *for any* `llm_confidence` in [0.0, 1.0], soll der finale `Confidence_Score` gleich `ML_GB_WEIGHT * gb_probability + ML_LLM_WEIGHT * llm_confidence` sein, und bei deaktiviertem oder fehlendem LLM soll `Confidence_Score` gleich `gb_probability` sein.

**Validates: Requirements 4.3, 4.4, 4.5**

### Property 9: Konfigurationsvalidierung

*For any* Kombination von `ML_GB_WEIGHT` und `ML_LLM_WEIGHT` deren Summe nicht 1.0 ergibt, oder *for any* negativer Schwellenwert, oder *for any* `ML_CONFIDENCE_THRESHOLD` außerhalb [0.0, 1.0], soll `validate_config()` einen `ValueError` auslösen.

**Validates: Requirements 9.3**

### Property 10: Fehlerresilienz — Exception-Isolation

*For any* Exception, die innerhalb von `ml_gate.evaluate()` auftritt (in Classifier, LLM-Advisor oder Feature-Pipeline), soll `evaluate()` niemals eine Exception an den Aufrufer weitergeben, sondern stattdessen eine gültige `GateDecision` mit Fallback-Werten zurückgeben.

**Validates: Requirements 15.1, 15.2, 15.4**

### Property 11: Vorhersage-Logging Vollständigkeit

*For any* `GateDecision`, die von `ml_gate.evaluate()` zurückgegeben wird, soll der zugehörige `ml_prediction_log.record()`-Aufruf einen Datensatz in der `ml_predictions`-Tabelle erzeugen, der alle Pflichtfelder (symbol, prediction_at, predicted_regime, confidence_score, gb_probability, action, risk_scale) enthält.

**Validates: Requirements 7.1, 7.4**

### Property 12: Modell-AUC-Regression verhindert Deployment

*For any* neu trainiertes Modell, dessen AUC-ROC auf dem Validierungsblock schlechter ist als die des aktuell eingesetzten Modells, soll der `WalkForwardTrainer` das neue Modell ablehnen und das bestehende beibehalten.

**Validates: Requirements 5.6**

### Property 13: Classifier-Inferenz-Wertebereich

*For any* Feature-Vektor, soll die Ausgabe von `RegimeClassifier.predict()` ein `regime` aus {"trending", "choppy"} und eine `probability` im Bereich [0.0, 1.0] enthalten.

**Validates: Requirements 3.4**

### Property 14: Datenbank-Schema-Kompatibilität

*For any* Aufruf von `init_ml_tables()`, sollen die bestehenden Tabellen (`trades`, `signals`, `funding_events`, `consumed_entry_signals`, `trade_quantity_epochs`) unverändert bleiben und die neuen Tabellen (`ml_predictions`, `ml_model_metadata`) idempotent erstellt werden.

**Validates: Requirements 14.3, 14.4**

### Property 15: Label-Generierung ausschließlich aus Zukunftsdaten

*For any* Label bei Index i, soll das Label ausschließlich auf Preisbewegungen der Candles i+1 bis i+N basieren. Eine Änderung von Daten an Index i oder früheren Indizes darf das Label bei Index i nicht verändern (solange Candles i+1..i+N konstant bleiben).

**Validates: Requirements 5.7**

### Property 16: Exchange-Adapter Substitutionsprinzip

*For any* Implementierung von `ExchangeAdapter`, soll der `BitgetAdapter` alle abstrakten Methoden (`get_candles`, `get_current_price`, `place_order`, `close_order`, `get_account_info`, `get_top_symbols`) implementieren und bei Aufruf die entsprechenden Funktionen aus `bitget_client.py` delegieren.

**Validates: Requirements 10.1, 10.2**
