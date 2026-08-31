# ml_gate.py
"""
ML-Gate Orchestrator: Experimentelles Filter-System zwischen der
trend-breakout-v3-Strategie und der Orderausfuehrung.

Konfiguration ueber Umgebungsvariablen (Anforderung 9).
Gate-Entscheidungslogik (Anforderung 1, 2, 3.5, 4.3, 8.4, 12.4, 15).
"""

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration (Anforderung 9.1, 9.2)
# ---------------------------------------------------------------------------
ML_GATE_ENABLED: bool = os.getenv("ML_GATE_ENABLED", "false").lower() == "true"
ML_CONFIDENCE_THRESHOLD: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.5"))
ML_MIN_RISK_SCALE: float = float(os.getenv("ML_MIN_RISK_SCALE", "0.5"))
ML_GB_WEIGHT: float = float(os.getenv("ML_GB_WEIGHT", "0.7"))
ML_LLM_WEIGHT: float = float(os.getenv("ML_LLM_WEIGHT", "0.3"))
ML_LLM_ENABLED: bool = os.getenv("ML_LLM_ENABLED", "false").lower() == "true"
ML_MODEL_DIR: str = os.getenv("ML_MODEL_DIR", ".models/")
ML_RETRAIN_INTERVAL_DAYS: int = int(os.getenv("ML_RETRAIN_INTERVAL_DAYS", "30"))

_DISCLAIMER = (
    "ML-Gate ist ein experimentelles Filter-System. "
    "Vergangene Backtest-Ergebnisse garantieren keine zukuenftige Performance."
)


# ---------------------------------------------------------------------------
# Konfigurationsvalidierung (Anforderung 9.3)
# ---------------------------------------------------------------------------
def validate_config() -> None:
    """Prueft ML-Konfiguration beim Start. Wirft ValueError bei Ungueltigkeit."""
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
        raise ValueError("ML_GB_WEIGHT und ML_LLM_WEIGHT duerfen nicht negativ sein")
    if abs((ML_GB_WEIGHT + ML_LLM_WEIGHT) - 1.0) > 1e-9:
        raise ValueError(
            f"ML_GB_WEIGHT + ML_LLM_WEIGHT muessen 1.0 ergeben, "
            f"ist {ML_GB_WEIGHT + ML_LLM_WEIGHT}"
        )
    if ML_RETRAIN_INTERVAL_DAYS < 1:
        raise ValueError(
            f"ML_RETRAIN_INTERVAL_DAYS muss >= 1 sein, "
            f"ist {ML_RETRAIN_INTERVAL_DAYS}"
        )


# ---------------------------------------------------------------------------
# Config-Banner (Anforderung 9.4, 13.3)
# ---------------------------------------------------------------------------
def _log_config_banner() -> None:
    """ML-Konfiguration im Startbanner ausgeben."""
    logger.info("--- ML-Gate Konfiguration ---")
    logger.info(f"  ML_CONFIDENCE_THRESHOLD: {ML_CONFIDENCE_THRESHOLD}")
    logger.info(f"  ML_MIN_RISK_SCALE:       {ML_MIN_RISK_SCALE}")
    logger.info(f"  ML_GB_WEIGHT:            {ML_GB_WEIGHT}")
    logger.info(f"  ML_LLM_WEIGHT:           {ML_LLM_WEIGHT}")
    logger.info(f"  ML_LLM_ENABLED:          {ML_LLM_ENABLED}")
    logger.info(f"  ML_MODEL_DIR:            {ML_MODEL_DIR}")
    logger.info(f"  ML_RETRAIN_INTERVAL_DAYS:{ML_RETRAIN_INTERVAL_DAYS}")
    logger.info("-----------------------------")


# ---------------------------------------------------------------------------
# Datenmodell (Anforderung 1, 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateDecision:
    """Ergebnis einer ML-Gate-Evaluation."""

    action: str            # "allow" | "block" | "hold"
    risk_scale: float      # 0.0 - 1.0, Skalierungsfaktor fuer Positionsgroesse
    confidence: float      # 0.0 - 1.0, kombinierter Confidence_Score
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
# Classifier-Registry (pro Symbol)
# ---------------------------------------------------------------------------
_classifiers: dict = {}  # dict[str, RegimeClassifier]


# ---------------------------------------------------------------------------
# Kernlogik (Task 7.2)
# ---------------------------------------------------------------------------
def init(symbols: list[str]) -> None:
    """
    Laedt Modelle beim Bot-Start. Bei Fehler: Fallback-Modus.
    Wird aus main.py nach init_db() aufgerufen.

    Anforderung 12.3: Neuestes Modell pro Symbol laden.
    Anforderung 15.3: Fehler beim Laden -> Fallback-Modus.
    """
    if not ML_GATE_ENABLED:
        logger.info("ML-Gate: deaktiviert (ML_GATE_ENABLED=false)")
        return
    validate_config()
    logger.info(_DISCLAIMER)
    _log_config_banner()
    for symbol in symbols:
        try:
            from ml_trainer import RegimeClassifier, should_retrain

            classifier = RegimeClassifier.load_latest(symbol, ML_MODEL_DIR)
            _classifiers[symbol] = classifier
            logger.info(
                f"ML-Gate: Modell geladen fuer {symbol}: "
                f"version={classifier.metadata.get('version', 'unknown')} "
                f"path={classifier.model_path}"
            )

            # Anforderung 5.4, 12.1: Retraining-Check nach Laden
            if should_retrain(symbol, ML_MODEL_DIR):
                logger.warning(
                    f"ML-Gate: Retraining faellig fuer {symbol} "
                    f"(Intervall={ML_RETRAIN_INTERVAL_DAYS} Tage). "
                    f"Bitte manuell oder per Scheduler ausloesen."
                )
        except Exception as exc:
            logger.warning(
                f"ML-Gate: Modell fuer {symbol} nicht ladbar ({exc}); "
                "Fallback auf Default-Werte"
            )


def evaluate(symbol: str, analysis: dict) -> GateDecision:
    """
    Evaluiert ein Entry-Signal durch den ML-Layer.
    Faengt ALLE Exceptions ab und gibt im Fehlerfall _BYPASS zurueck.

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
            action="allow",
            risk_scale=1.0,
            confidence=0.5,
            regime="trending",
            filter_reason="ml_exception_fallback",
            gb_probability=0.5,
            llm_confidence=None,
            llm_regime=None,
            model_version="fallback",
            latency_ms=elapsed,
        )


def _evaluate_inner(
    symbol: str, analysis: dict, start_time: float
) -> GateDecision:
    """Interne Gate-Logik ohne Fallback-Wrapping."""
    from ml_features import build_features

    # 1) Features berechnen (OHLCV-basiert)
    features = build_features(symbol, analysis.get("indicators", {}))

    # 1b) Externe Sentiment-Features ergaenzen (Live-Only, nicht im Training)
    try:
        from market_sentiment import get_features as get_sentiment
        sentiment = get_sentiment(symbol)
        features.update(sentiment)
    except Exception as exc:
        logger.debug(f"Sentiment-Features nicht verfuegbar fuer {symbol}: {exc}")

    # 2) Gradient-Boosting-Vorhersage
    classifier = _classifiers.get(symbol)
    if classifier is None:
        # Anforderung 3.5: kein Modell -> Default
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
            # Anforderung 15.1: Inferenz-Exception -> Default
            logger.warning(
                f"ML-Gate: Classifier-Exception fuer {symbol}: {exc}"
            )
            gb_regime = "trending"
            gb_probability = 0.5
            model_version = "fallback"

    # 3) LLM-Regime-Advisor (optional)
    llm_confidence = None
    llm_regime = None
    if ML_LLM_ENABLED:
        try:
            llm_result = _query_llm_advisor(
                symbol, analysis.get("indicators", {})
            )
            llm_confidence = llm_result["confidence"]
            llm_regime = llm_result["regime"]
        except Exception as exc:
            # Anforderung 15.2 / 4.4: LLM-Fehler -> nur GB verwenden
            logger.warning(
                f"ML-Gate: LLM-Exception fuer {symbol}: {exc}"
            )

    # 4) Gewichteter Confidence_Score (Anforderung 4.3)
    if llm_confidence is not None:
        confidence = (
            ML_GB_WEIGHT * gb_probability + ML_LLM_WEIGHT * llm_confidence
        )
    else:
        confidence = gb_probability

    # 5) Gate-Entscheidung (Anforderung 1.2, 1.3)
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
    Gibt {"regime": str, "confidence": float} zurueck.
    """
    from ml_gate_llm import LLMRegimeAdvisor

    advisor = LLMRegimeAdvisor()
    return advisor.assess(symbol, indicators)


def rollback_model(symbol: str, version: str) -> bool:
    """
    Anforderung 12.4: Rollback auf eine fruehere Modellversion.

    Returns True bei Erfolg, False bei Fehler.
    """
    from ml_trainer import RegimeClassifier

    try:
        classifier = RegimeClassifier.load_version(
            symbol, version, ML_MODEL_DIR
        )
        _classifiers[symbol] = classifier
        logger.info(
            f"ML-Gate: Rollback fuer {symbol} auf Version {version}"
        )
        return True
    except Exception as exc:
        logger.error(
            f"ML-Gate: Rollback fehlgeschlagen fuer {symbol}: {exc}"
        )
        return False


def get_training_period(symbol: str) -> str | None:
    """
    Anforderung 7.4: Gibt den Trainingszeitraum des aktiven Modells
    fuer ein Symbol als JSON-String zurueck, oder None.
    """
    classifier = _classifiers.get(symbol)
    if classifier is None:
        return None
    period = classifier.metadata.get("training_period")
    if period:
        import json
        return json.dumps(period)
    return None
