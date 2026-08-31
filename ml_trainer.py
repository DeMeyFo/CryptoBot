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
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retraining-Intervall (Anforderung 5.4)
# ---------------------------------------------------------------------------
ML_RETRAIN_INTERVAL_DAYS = int(os.getenv("ML_RETRAIN_INTERVAL_DAYS", "30"))


def should_retrain(symbol: str, model_dir: str) -> bool:
    """
    Anforderung 5.4, 12.1: Prueft ob seit dem letzten Training
    ML_RETRAIN_INTERVAL_DAYS vergangen sind.

    Laedt metadata.json des neuesten Modells fuer *symbol* und vergleicht
    das Feld ``trained_at`` mit der aktuellen UTC-Zeit.

    Returns True wenn Retraining faellig ist (oder kein Modell existiert).
    """
    base = Path(model_dir)
    if not base.exists():
        logger.info(
            f"should_retrain: Verzeichnis {model_dir} existiert nicht "
            f"fuer {symbol} -> Retraining empfohlen"
        )
        return True

    candidates = sorted(
        [d for d in base.iterdir()
         if d.is_dir() and d.name.startswith(f"{symbol}_")],
        key=lambda d: d.name,
        reverse=True,
    )
    if not candidates:
        logger.info(
            f"should_retrain: Kein Modell fuer {symbol} -> Retraining empfohlen"
        )
        return True

    meta_file = candidates[0] / "metadata.json"
    if not meta_file.exists():
        logger.warning(
            f"should_retrain: metadata.json fehlt in {candidates[0]} "
            f"-> Retraining empfohlen"
        )
        return True

    try:
        with open(meta_file) as f:
            metadata = json.load(f)
        trained_at_str = metadata.get("trained_at")
        if not trained_at_str:
            logger.warning(
                f"should_retrain: trained_at fehlt in {meta_file} "
                f"-> Retraining empfohlen"
            )
            return True

        trained_at = datetime.fromisoformat(trained_at_str)
        # Ensure timezone-aware comparison
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_since = (now - trained_at).days
        due = days_since >= ML_RETRAIN_INTERVAL_DAYS
        logger.info(
            f"should_retrain: {symbol} letztes Training vor {days_since} Tagen "
            f"(Intervall={ML_RETRAIN_INTERVAL_DAYS}d) -> "
            f"{'faellig' if due else 'nicht faellig'}"
        )
        return due
    except Exception as exc:
        logger.warning(
            f"should_retrain: Fehler beim Pruefen fuer {symbol}: {exc} "
            f"-> Retraining empfohlen"
        )
        return True


def persist_model_metadata(symbol: str, metadata: dict,
                           model_path: str) -> None:
    """
    Anforderung 12.2, 12.3: Persistiert Modell-Metadaten in der
    ml_model_metadata-Tabelle (SQLite).

    Setzt is_active=1 fuer das neue Modell und is_active=0 fuer alle
    aelteren Modelle desselben Symbols.
    """
    try:
        from ml_prediction_log import _conn
        version = metadata.get("version", "unknown")
        metrics = metadata.get("validation_metrics", {})
        training_period = metadata.get("training_period", {})

        with _conn() as connection:
            # Deaktiviere alle bestehenden aktiven Modelle fuer dieses Symbol
            connection.execute(
                "UPDATE ml_model_metadata SET is_active=0 WHERE symbol=?",
                (symbol,),
            )
            connection.execute("""
                INSERT OR REPLACE INTO ml_model_metadata
                    (symbol, version, model_path, training_start, training_end,
                     feature_columns, hyperparameters, accuracy,
                     precision_score, recall, f1_score, auc_roc,
                     data_hash, trained_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                symbol,
                version,
                model_path,
                training_period.get("start"),
                training_period.get("end"),
                json.dumps(metadata.get("feature_columns", [])),
                json.dumps(metadata.get("hyperparameters", {}), default=str),
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
                metrics.get("auc_roc"),
                metadata.get("data_hash"),
                metadata.get("trained_at",
                             datetime.now(timezone.utc).isoformat()),
            ))
            connection.commit()
        logger.info(
            f"persist_model_metadata: {symbol} Version {version} "
            f"als aktiv gespeichert"
        )
    except Exception as exc:
        logger.warning(
            f"persist_model_metadata: Fehler beim Speichern fuer {symbol}: {exc}"
        )


# ---------------------------------------------------------------------------
# Label-Generierung (Anforderung 5.7)
# ---------------------------------------------------------------------------
ML_LABEL_HORIZON = int(os.getenv("ML_LABEL_HORIZON", "12"))
ML_LABEL_ATR_THRESHOLD = float(os.getenv("ML_LABEL_ATR_THRESHOLD", "1.0"))


def generate_labels(
    df: pd.DataFrame,
    horizon: int = ML_LABEL_HORIZON,
    atr_threshold: float = ML_LABEL_ATR_THRESHOLD,
) -> pd.Series:
    """
    Label: trending (1) wenn die *gerichtete* Preisbewegung in den naechsten
    `horizon` Candles > atr_threshold * ATR ist, sonst choppy (0).

    Die gerichtete Bewegung ist definiert als:
        abs(close[i+horizon] - close[i]) / close[i]
    statt der alten max-min Spanne. Das ist strenger: eine volatile
    Seitwaertsphase mit grosser Spanne aber kleinem Nettoergebnis wird
    jetzt korrekt als choppy klassifiziert.

    Verwendet nur Daten der folgenden Candles -> nicht kausal fuer Features,
    aber korrekt als Label (supervised signal).
    """
    close = df["close"]
    atr = df["atr"]
    directional_move = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - horizon):
        end_close = close.iloc[i + horizon]
        directional_move.iloc[i] = abs(end_close - close.iloc[i]) / close.iloc[i]
    label = (directional_move > atr_threshold * atr / close).astype(float)
    return label


# ---------------------------------------------------------------------------
# Walk-Forward-Split (Anforderung 5.1, 5.2, 6.3)
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardSplit:
    train_start: int  # Index im DataFrame
    train_end: int
    val_start: int
    val_end: int


def walk_forward_splits(
    n_samples: int,
    train_ratio: float = 0.7,
    n_splits: int = 1,
) -> list[WalkForwardSplit]:
    """
    Erzeugt chronologisch geordnete Train/Val-Splits.
    Validierung folgt zeitlich auf Training, ohne Ueberlappung.
    """
    splits = []
    split_size = n_samples // n_splits
    for i in range(n_splits):
        start = i * split_size
        end = min(start + split_size, n_samples)
        boundary = start + int((end - start) * train_ratio)
        splits.append(
            WalkForwardSplit(
                train_start=start,
                train_end=boundary,
                val_start=boundary,
                val_end=end,
            )
        )
    return splits

# ---------------------------------------------------------------------------
# RegimeClassifier (Anforderung 3.1, 3.4, 3.5)
# ---------------------------------------------------------------------------
class RegimeClassifier:
    """Gradient-Boosting-Binaerklassifikator fuer Marktregime."""

    def __init__(self, model, feature_columns: list[str], metadata: dict):
        self.model = model
        self.feature_columns = feature_columns
        self.metadata = metadata
        self.model_path = metadata.get("path", "unknown")

    def predict(self, features: dict) -> dict:
        """
        Anforderung 3.4: Gibt Klassifikation und Wahrscheinlichkeit zurueck.
        """
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
        """Laedt das neueste Modell fuer ein Symbol aus model_dir."""
        base = Path(model_dir)
        if not base.exists():
            raise FileNotFoundError(
                f"Kein Modell-Verzeichnis: {model_dir}"
            )
        candidates = sorted(
            [d for d in base.iterdir()
             if d.is_dir() and d.name.startswith(f"{symbol}_")],
            key=lambda d: d.name,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"Kein Modell fuer {symbol} in {model_dir}"
            )
        return cls._load_from_dir(candidates[0])

    @classmethod
    def load_version(cls, symbol: str, version: str,
                     model_dir: str) -> "RegimeClassifier":
        """Laedt eine spezifische Modellversion (Anforderung 12.4)."""
        target = Path(model_dir) / f"{symbol}_{version}"
        if not target.is_dir():
            raise FileNotFoundError(
                f"Modellversion {version} nicht gefunden: {target}"
            )
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
ML_MODEL_DIR = os.getenv("ML_MODEL_DIR", ".models/")


class WalkForwardTrainer:
    """Walk-Forward-Training mit Modell-Persistenz und Validierung."""

    def __init__(self, model_dir: str = ML_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        symbol: str,
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict:
        """
        Fuehrt einen vollstaendigen Walk-Forward-Trainingszyklus durch.

        Anforderung 5.2: Hyperparameter-Suche nur auf Trainingsblock.
        Anforderung 5.5: Validierungsmetriken loggen.
        Anforderung 5.6: AUC-ROC Regression verhindern.
        Anforderung 12.1, 12.2: Modell + Metadaten persistieren.
        Anforderung 13.4: Metriken mit Praefix kennzeichnen.

        Returns: dict mit Validierungsmetriken und Modellpfad.
        """
        from xgboost import XGBClassifier
        from ml_features import build_feature_matrix

        # Feature-Matrix und Labels aufbauen
        frame = build_feature_matrix(df)
        labels = generate_labels(df)

        # NaN-Zeilen entfernen
        valid_mask = frame.notna().all(axis=1) & labels.notna()
        frame = frame[valid_mask]
        labels = labels[valid_mask]

        if len(frame) < 100:
            raise ValueError(
                f"Zu wenige gueltige Datenpunkte fuer Training: {len(frame)}"
            )

        # Walk-Forward-Split: letzten Split verwenden
        splits = walk_forward_splits(len(frame))
        split = splits[-1]

        X_train = frame.iloc[split.train_start : split.train_end]
        y_train = labels.iloc[split.train_start : split.train_end]
        X_val = frame.iloc[split.val_start : split.val_end]
        y_val = labels.iloc[split.val_start : split.val_end]

        # Anforderung 5.2: Training nur auf Trainingsblock
        # Class balancing: scale_pos_weight = n_negative / n_positive.
        # Without this the model memorises the majority class (trending)
        # and produces zero useful choppy predictions.
        n_pos = float((y_train == 1.0).sum())
        n_neg = float((y_train == 0.0).sum())
        spw = max(0.1, n_neg / n_pos) if n_pos > 0 else 1.0
        logger.info(
            f"  Klassenverteilung: {n_pos:.0f} trending / {n_neg:.0f} choppy "
            f"({n_neg / len(y_train) * 100:.1f}% Minderheit) "
            f"-> scale_pos_weight={spw:.3f}"
        )
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            scale_pos_weight=spw,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Anforderung 5.5: Validierungsmetriken berechnen
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(
                precision_score(y_val, y_pred, zero_division=0)
            ),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_val, y_proba)),
        }

        # Anforderung 5.6: Vergleich mit aktuellem Modell
        current_auc = self._current_model_auc(symbol)
        if current_auc is not None and metrics["auc_roc"] < current_auc:
            logger.warning(
                f"Historisch gemessen: Neues Modell fuer {symbol} "
                f"(AUC-ROC={metrics['auc_roc']:.4f}) schlechter als "
                f"aktuelles (AUC-ROC={current_auc:.4f}). "
                f"Bestehendes Modell beibehalten."
            )
            return {
                "status": "rejected",
                "metrics": metrics,
                "reason": "auc_regression",
            }

        # Anforderung 13.4: Metriken-Praefix
        logger.info(
            f"Historisch gemessen: {symbol} Walk-Forward-Validierung: "
            f"Accuracy={metrics['accuracy']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1_score']:.4f} "
            f"AUC-ROC={metrics['auc_roc']:.4f}"
        )

        # Anforderung 12.1, 12.2: Persistenz
        model_path = self._save_model(
            symbol, model, feature_columns, metrics, df
        )

        return {
            "status": "accepted",
            "metrics": metrics,
            "model_path": str(model_path),
        }

    def historical_predictions(
        self,
        symbol: str,
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> list[tuple]:
        """
        Anforderung 11.1: Historische Regime-Vorhersagen fuer Backtest.
        Returns: [(timestamp, regime, confidence_score), ...]

        Laedt das neueste Modell und wendet es auf alle Zeilen des
        Feature-DataFrames an, die keine NaN-Werte enthalten.
        """
        from ml_features import build_feature_matrix

        classifier = RegimeClassifier.load_latest(
            symbol, str(self.model_dir)
        )
        features = build_feature_matrix(df)

        predictions = []
        for i in range(len(features)):
            row = features.iloc[i]
            if row.isna().any():
                continue
            feature_dict = row.to_dict()
            result = classifier.predict(feature_dict)
            ts = df.iloc[i].get("timestamp", df.index[i])
            predictions.append((
                ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                result["regime"],
                result["probability"],
            ))
        return predictions

    def _save_model(
        self,
        symbol: str,
        model,
        feature_columns: list[str],
        metrics: dict,
        df: pd.DataFrame,
    ) -> Path:
        """
        Anforderung 12.1, 12.2: Modell + Metadaten speichern.
        Verzeichnisstruktur: {ML_MODEL_DIR}/{symbol}_{timestamp}/
          - model.pkl
          - metadata.json
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dir_name = f"{symbol}_{timestamp}"
        model_dir = self.model_dir / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Modell serialisieren
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Daten-Hash fuer Nachvollziehbarkeit
        data_hash = hashlib.sha256(
            df.to_csv(index=False).encode()
        ).hexdigest()[:16]

        # Trainings- und Validierungszeitraum ermitteln
        ts_col = "timestamp" if "timestamp" in df.columns else None
        training_start = str(df.iloc[0][ts_col]) if ts_col else str(df.index[0])
        training_end = str(df.iloc[-1][ts_col]) if ts_col else str(df.index[-1])

        metadata = {
            "symbol": symbol,
            "version": timestamp,
            "feature_columns": feature_columns,
            "training_period": {
                "start": training_start,
                "end": training_end,
            },
            "validation_metrics": metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_hash": data_hash,
            "hyperparameters": model.get_params(),
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(
            f"Backtest-Ergebnis: Modell gespeichert in {model_dir}"
        )
        return model_dir

    def _current_model_auc(self, symbol: str) -> float | None:
        """Laedt die AUC-ROC des aktuellen Modells fuer den Vergleich."""
        try:
            classifier = RegimeClassifier.load_latest(
                symbol, str(self.model_dir)
            )
            return classifier.metadata.get(
                "validation_metrics", {}
            ).get("auc_roc")
        except FileNotFoundError:
            return None
