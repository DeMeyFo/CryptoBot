# ml_prediction_log.py

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Outcome-Tracking Konfiguration (Anforderung 7.2)
# ---------------------------------------------------------------------------
ML_LABEL_HORIZON = int(os.getenv("ML_LABEL_HORIZON", "24"))
ML_LABEL_ATR_THRESHOLD = float(os.getenv("ML_LABEL_ATR_THRESHOLD", "1.5"))


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


def record(decision, symbol: str, top_features: dict | None = None,
           training_period: str | None = None) -> int:
    """
    Anforderung 7.1, 7.4: Persistiert eine ML-Vorhersage.
    Annotiert model_version und training_period (Anforderung 7.4).
    Returns: prediction_id für die Referenz in der signals-Tabelle.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as connection:
        cursor = connection.execute("""
            INSERT INTO ml_predictions
                (symbol, prediction_at, predicted_regime, confidence_score,
                 gb_probability, llm_confidence, llm_regime, top_features,
                 action, risk_scale, filter_reason, model_version,
                 training_period, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, now, decision.regime, decision.confidence,
            decision.gb_probability, decision.llm_confidence, decision.llm_regime,
            json.dumps(top_features) if top_features else None,
            decision.action, decision.risk_scale, decision.filter_reason,
            decision.model_version, training_period, decision.latency_ms,
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


def get_pending_outcomes(horizon_candles: int | None = None,
                         timeframe_seconds: int = 3600) -> list[dict]:
    """
    Anforderung 7.2: Finde Vorhersagen, deren N-Candle-Fenster abgelaufen
    ist und die noch kein Ergebnis haben.

    Returns: Liste von dicts mit id, symbol, prediction_at.
    """
    if horizon_candles is None:
        horizon_candles = ML_LABEL_HORIZON
    window_seconds = horizon_candles * timeframe_seconds
    cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
    cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()

    with _conn() as connection:
        rows = connection.execute("""
            SELECT id, symbol, prediction_at
            FROM ml_predictions
            WHERE outcome_updated_at IS NULL
              AND prediction_at <= ?
            ORDER BY prediction_at ASC
            LIMIT 50
        """, (cutoff_iso,)).fetchall()

    return [
        {"id": row[0], "symbol": row[1], "prediction_at": row[2]}
        for row in rows
    ]


def backfill_outcomes(
    get_candles_fn=None,
    horizon_candles: int | None = None,
    timeframe_seconds: int = 3600,
    atr_threshold: float | None = None,
) -> dict:
    """
    Anforderung 7.2: Periodisch aufgerufene Hilfsfunktion, die Vorhersagen
    nachtraeglich mit tatsaechlichem Regime und Performance annotiert.

    Wird aus der Trading-Loop aufgerufen. Ist non-blocking und wrapped
    in try/except, damit sie die Loop nie stoert.

    Args:
        get_candles_fn: Callable(symbol, granularity, limit) -> list[candle].
            Falls None, wird bitget_client.get_candles verwendet.
        horizon_candles: Anzahl Candles fuer das Evaluationsfenster.
            Default: ML_LABEL_HORIZON (24).
        timeframe_seconds: Dauer einer Candle in Sekunden (3600 fuer 1H).
        atr_threshold: ATR-Schwellenwert fuer Regime-Klassifikation.
            Default: ML_LABEL_ATR_THRESHOLD (1.5).

    Returns: dict mit updated (Anzahl aktualisiert), skipped, errors.
    """
    if horizon_candles is None:
        horizon_candles = ML_LABEL_HORIZON
    if atr_threshold is None:
        atr_threshold = ML_LABEL_ATR_THRESHOLD

    stats = {"updated": 0, "skipped": 0, "errors": 0}

    try:
        pending = get_pending_outcomes(horizon_candles, timeframe_seconds)
    except Exception as exc:
        logger.error(f"backfill_outcomes: Fehler beim Laden der pending predictions: {exc}")
        stats["errors"] += 1
        return stats

    if not pending:
        return stats

    if get_candles_fn is None:
        try:
            from bitget_client import get_candles
            get_candles_fn = get_candles
        except ImportError:
            logger.error("backfill_outcomes: bitget_client nicht verfuegbar")
            stats["errors"] += 1
            return stats

    for pred in pending:
        try:
            _backfill_single_prediction(
                pred, get_candles_fn, horizon_candles, atr_threshold
            )
            stats["updated"] += 1
        except Exception as exc:
            logger.warning(
                f"backfill_outcomes: Fehler bei prediction {pred['id']} "
                f"({pred['symbol']}): {exc}"
            )
            stats["errors"] += 1

    if stats["updated"] > 0:
        logger.info(
            f"Outcome-Tracking: {stats['updated']} predictions aktualisiert, "
            f"{stats['errors']} Fehler"
        )
    return stats


def _backfill_single_prediction(
    pred: dict,
    get_candles_fn,
    horizon_candles: int,
    atr_threshold: float,
) -> None:
    """
    Berechnet das tatsaechliche Regime und die Performance fuer eine
    einzelne Vorhersage und ruft update_outcome() auf.
    """
    symbol = pred["symbol"]

    # Wir brauchen genug Candles, um ATR und die folgenden N Candles
    # nach dem Vorhersagezeitpunkt zu berechnen.
    candles = get_candles_fn(symbol, "1H", horizon_candles + 50)
    if not candles or len(candles) < horizon_candles + 1:
        raise ValueError(
            f"Zu wenige Candles fuer {symbol}: {len(candles) if candles else 0}"
        )

    # Candles in Zeitreihen-Dicts umwandeln
    closes = [float(c[4]) if isinstance(c, (list, tuple)) else float(c.get("close", c.get("c", 0)))
              for c in candles]
    highs = [float(c[2]) if isinstance(c, (list, tuple)) else float(c.get("high", c.get("h", 0)))
             for c in candles]
    lows = [float(c[3]) if isinstance(c, (list, tuple)) else float(c.get("low", c.get("l", 0)))
            for c in candles]

    if len(closes) < horizon_candles + 1:
        raise ValueError(f"Nicht genug Preisdaten fuer {symbol}")

    # Verwende die letzten horizon_candles+1 Candles:
    # Index 0 = Vorhersage-Candle, Index 1..N = folgende Candles
    ref_close = closes[-(horizon_candles + 1)]
    window_closes = closes[-horizon_candles:]

    # Performance: max Preisbewegung im Fenster relativ zum Referenz-Close
    if ref_close <= 0:
        raise ValueError(f"Referenz-Close ist 0 fuer {symbol}")

    max_price = max(window_closes)
    min_price = min(window_closes)
    actual_performance = (max_price - min_price) / ref_close

    # ATR-Naeherung: Durchschnitt der True Range der letzten 14 Candles
    # vor dem Vorhersagezeitpunkt
    atr_window = min(14, len(closes) - horizon_candles - 1)
    if atr_window > 0:
        start_idx = len(closes) - horizon_candles - 1 - atr_window
        end_idx = len(closes) - horizon_candles - 1
        true_ranges = []
        for i in range(max(1, start_idx), end_idx + 1):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)
        atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    else:
        atr = 0.0

    # Regime bestimmen: trending wenn Performance > atr_threshold * ATR/close
    atr_normalized = atr / ref_close if ref_close > 0 else 0.0
    if actual_performance > atr_threshold * atr_normalized:
        actual_regime = "trending"
    else:
        actual_regime = "choppy"

    update_outcome(pred["id"], actual_regime, actual_performance)


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
