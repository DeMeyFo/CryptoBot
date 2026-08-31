# tests/test_retraining.py
"""
Tests fuer Retraining-Logik in ml_trainer.py (Task 12.1)

Validates: Requirements 5.4, 12.1, 12.2, 12.3
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_trainer import should_retrain, persist_model_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_model_dir_with_metadata(
    base: str,
    symbol: str,
    version: str,
    trained_at: str | None = None,
) -> Path:
    """Create a mock model directory with metadata.json."""
    dir_path = Path(base) / f"{symbol}_{version}"
    dir_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "symbol": symbol,
        "version": version,
        "feature_columns": ["adx", "rsi", "atr"],
        "validation_metrics": {"auc_roc": 0.75, "accuracy": 0.80},
        "training_period": {"start": "2024-01-01", "end": "2024-06-30"},
        "data_hash": "abc123",
    }
    if trained_at is not None:
        metadata["trained_at"] = trained_at

    with open(dir_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    return dir_path


# ---------------------------------------------------------------------------
# Tests for should_retrain
# Validates: Requirement 5.4, 12.1
# ---------------------------------------------------------------------------

class TestShouldRetrain(unittest.TestCase):
    """
    **Validates: Requirements 5.4, 12.1**

    should_retrain() checks if ML_RETRAIN_INTERVAL_DAYS have passed
    since the last training timestamp in metadata.json.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_no_model_dir_returns_true(self):
        """When model directory doesn't exist, retraining is needed."""
        result = should_retrain("BTCUSDT", "/nonexistent/path/xyz")
        self.assertTrue(result)

    def test_empty_model_dir_returns_true(self):
        """When no model subdirs exist for the symbol, retraining is needed."""
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertTrue(result)

    def test_no_metadata_file_returns_true(self):
        """When metadata.json is missing, retraining is needed."""
        dir_path = Path(self.tmpdir) / "BTCUSDT_20250101T000000Z"
        dir_path.mkdir(parents=True)
        # No metadata.json created
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertTrue(result)

    def test_missing_trained_at_field_returns_true(self):
        """When trained_at is missing from metadata, retraining is needed."""
        _create_model_dir_with_metadata(
            self.tmpdir, "BTCUSDT", "20250101T000000Z",
            trained_at=None,
        )
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertTrue(result)

    @patch("ml_trainer.ML_RETRAIN_INTERVAL_DAYS", 30)
    def test_recent_training_returns_false(self):
        """Model trained 5 days ago with 30-day interval is not due."""
        recent = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        _create_model_dir_with_metadata(
            self.tmpdir, "BTCUSDT", "20250801T000000Z",
            trained_at=recent,
        )
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertFalse(result)

    @patch("ml_trainer.ML_RETRAIN_INTERVAL_DAYS", 30)
    def test_old_training_returns_true(self):
        """Model trained 60 days ago with 30-day interval is due."""
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        _create_model_dir_with_metadata(
            self.tmpdir, "BTCUSDT", "20250101T000000Z",
            trained_at=old,
        )
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertTrue(result)

    @patch("ml_trainer.ML_RETRAIN_INTERVAL_DAYS", 30)
    def test_exactly_at_interval_returns_true(self):
        """Model trained exactly 30 days ago should be due (>= check)."""
        exact = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        _create_model_dir_with_metadata(
            self.tmpdir, "BTCUSDT", "20250101T000000Z",
            trained_at=exact,
        )
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertTrue(result)

    @patch("ml_trainer.ML_RETRAIN_INTERVAL_DAYS", 7)
    def test_configurable_interval(self):
        """Retraining interval of 7 days: model trained 3 days ago is not due."""
        recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        _create_model_dir_with_metadata(
            self.tmpdir, "BTCUSDT", "20250801T000000Z",
            trained_at=recent,
        )
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertFalse(result)

    @patch("ml_trainer.ML_RETRAIN_INTERVAL_DAYS", 7)
    def test_configurable_interval_expired(self):
        """Retraining interval of 7 days: model trained 10 days ago is due."""
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        _create_model_dir_with_metadata(
            self.tmpdir, "BTCUSDT", "20250101T000000Z",
            trained_at=old,
        )
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertTrue(result)

    @patch("ml_trainer.ML_RETRAIN_INTERVAL_DAYS", 30)
    def test_picks_latest_model_dir(self):
        """should_retrain checks the most recent model directory (sorted last)."""
        # Old model
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        _create_model_dir_with_metadata(
            self.tmpdir, "BTCUSDT", "20250101T000000Z",
            trained_at=old,
        )
        # Recent model (sorted after the old one alphabetically)
        recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        _create_model_dir_with_metadata(
            self.tmpdir, "BTCUSDT", "20250801T000000Z",
            trained_at=recent,
        )
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertFalse(result)

    @patch("ml_trainer.ML_RETRAIN_INTERVAL_DAYS", 30)
    def test_ignores_other_symbols(self):
        """should_retrain for BTCUSDT ignores ETHUSDT model dirs."""
        recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        _create_model_dir_with_metadata(
            self.tmpdir, "ETHUSDT", "20250801T000000Z",
            trained_at=recent,
        )
        # No BTCUSDT model exists
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertTrue(result)

    @patch("ml_trainer.ML_RETRAIN_INTERVAL_DAYS", 30)
    def test_malformed_trained_at_returns_true(self):
        """Malformed trained_at string should trigger retraining."""
        dir_path = Path(self.tmpdir) / "BTCUSDT_20250101T000000Z"
        dir_path.mkdir(parents=True, exist_ok=True)
        metadata = {"trained_at": "not-a-date"}
        with open(dir_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertTrue(result)

    @patch("ml_trainer.ML_RETRAIN_INTERVAL_DAYS", 30)
    def test_naive_datetime_handled(self):
        """Naive datetime string (no timezone) should still be handled."""
        recent = (datetime.now(timezone.utc) - timedelta(days=5))
        # Write as naive ISO string (no +00:00)
        naive_str = recent.strftime("%Y-%m-%dT%H:%M:%S")
        _create_model_dir_with_metadata(
            self.tmpdir, "BTCUSDT", "20250801T000000Z",
            trained_at=naive_str,
        )
        result = should_retrain("BTCUSDT", self.tmpdir)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Tests for persist_model_metadata
# Validates: Requirements 12.2, 12.3
# ---------------------------------------------------------------------------

class TestPersistModelMetadata(unittest.TestCase):
    """
    **Validates: Requirements 12.2, 12.3**

    persist_model_metadata() stores model metadata in the
    ml_model_metadata SQLite table and sets is_active=1.
    """

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        # Create the ml_model_metadata table
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
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
        conn.commit()
        conn.close()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def _mock_conn(self):
        """Return a connection to the test database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _query_all(self):
        """Query all records from ml_model_metadata."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM ml_model_metadata").fetchall()
        conn.close()
        return rows

    @patch("ml_prediction_log._conn")
    def test_persist_creates_record(self, mock_conn):
        """persist_model_metadata should create a record in the DB."""
        mock_conn.side_effect = self._mock_conn

        metadata = {
            "symbol": "BTCUSDT",
            "version": "20250601T120000Z",
            "feature_columns": ["adx", "rsi"],
            "training_period": {"start": "2024-01-01", "end": "2024-06-30"},
            "validation_metrics": {
                "accuracy": 0.80,
                "precision": 0.75,
                "recall": 0.70,
                "f1_score": 0.72,
                "auc_roc": 0.85,
            },
            "trained_at": "2025-06-01T12:00:00+00:00",
            "data_hash": "abc123def456",
            "hyperparameters": {"n_estimators": 200, "max_depth": 4},
        }
        persist_model_metadata("BTCUSDT", metadata, "/models/BTCUSDT_20250601T120000Z")

        rows = self._query_all()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["symbol"], "BTCUSDT")
        self.assertEqual(row["version"], "20250601T120000Z")
        self.assertEqual(row["is_active"], 1)
        self.assertAlmostEqual(row["auc_roc"], 0.85)
        self.assertAlmostEqual(row["accuracy"], 0.80)

    @patch("ml_prediction_log._conn")
    def test_persist_deactivates_previous(self, mock_conn):
        """New model should set is_active=0 for previous models of the same symbol."""
        mock_conn.side_effect = self._mock_conn

        metadata_v1 = {
            "symbol": "BTCUSDT",
            "version": "20250101T000000Z",
            "feature_columns": ["adx"],
            "training_period": {"start": "2024-01-01", "end": "2024-03-31"},
            "validation_metrics": {"auc_roc": 0.70},
            "trained_at": "2025-01-01T00:00:00+00:00",
            "data_hash": "hash1",
        }
        persist_model_metadata("BTCUSDT", metadata_v1, "/models/v1")

        metadata_v2 = {
            "symbol": "BTCUSDT",
            "version": "20250601T000000Z",
            "feature_columns": ["adx", "rsi"],
            "training_period": {"start": "2024-04-01", "end": "2024-06-30"},
            "validation_metrics": {"auc_roc": 0.85},
            "trained_at": "2025-06-01T00:00:00+00:00",
            "data_hash": "hash2",
        }
        persist_model_metadata("BTCUSDT", metadata_v2, "/models/v2")

        rows = self._query_all()
        self.assertEqual(len(rows), 2)

        # Sort by version to get consistent order
        sorted_rows = sorted(rows, key=lambda r: r["version"])
        self.assertEqual(sorted_rows[0]["is_active"], 0)  # v1 deactivated
        self.assertEqual(sorted_rows[1]["is_active"], 1)  # v2 active

    @patch("ml_prediction_log._conn")
    def test_persist_different_symbols_independent(self, mock_conn):
        """Models for different symbols should not affect each other."""
        mock_conn.side_effect = self._mock_conn

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            metadata = {
                "symbol": symbol,
                "version": "20250601T000000Z",
                "feature_columns": ["adx"],
                "training_period": {},
                "validation_metrics": {},
                "trained_at": "2025-06-01T00:00:00+00:00",
                "data_hash": "hash",
            }
            persist_model_metadata(symbol, metadata, f"/models/{symbol}_v1")

        rows = self._query_all()
        active_rows = [r for r in rows if r["is_active"] == 1]
        self.assertEqual(len(active_rows), 2)

    @patch("ml_prediction_log._conn")
    def test_persist_stores_feature_columns_as_json(self, mock_conn):
        """feature_columns should be stored as JSON string."""
        mock_conn.side_effect = self._mock_conn

        cols = ["adx", "rsi", "atr", "di_spread"]
        metadata = {
            "symbol": "BTCUSDT",
            "version": "20250601T000000Z",
            "feature_columns": cols,
            "training_period": {},
            "validation_metrics": {},
            "trained_at": "2025-06-01T00:00:00+00:00",
            "data_hash": "hash",
        }
        persist_model_metadata("BTCUSDT", metadata, "/models/v1")

        rows = self._query_all()
        stored_cols = json.loads(rows[0]["feature_columns"])
        self.assertEqual(stored_cols, cols)


# ---------------------------------------------------------------------------
# Tests for ml_gate.init() retraining check
# Validates: Requirements 12.3
# ---------------------------------------------------------------------------

class TestMlGateInitRetrainingCheck(unittest.TestCase):
    """
    **Validates: Requirement 12.3**

    ml_gate.init() checks if retraining is due after loading each model.
    """

    @patch("ml_trainer.should_retrain", return_value=True)
    @patch("ml_trainer.RegimeClassifier.load_latest")
    def test_init_calls_should_retrain(self, mock_load, mock_retrain):
        """init() should call should_retrain after successfully loading a model."""
        import ml_gate

        mock_classifier = MagicMock()
        mock_classifier.metadata = {"version": "test_v1"}
        mock_classifier.model_path = "/models/test"
        mock_load.return_value = mock_classifier

        with patch.multiple(
            ml_gate,
            ML_GATE_ENABLED=True,
            ML_CONFIDENCE_THRESHOLD=0.5,
            ML_MIN_RISK_SCALE=0.5,
            ML_GB_WEIGHT=0.7,
            ML_LLM_WEIGHT=0.3,
            ML_RETRAIN_INTERVAL_DAYS=30,
        ):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                ml_gate.init(["BTCUSDT"])

        mock_retrain.assert_called_once_with("BTCUSDT", ml_gate.ML_MODEL_DIR)

    @patch("ml_trainer.should_retrain", return_value=False)
    @patch("ml_trainer.RegimeClassifier.load_latest")
    def test_init_no_warning_when_not_due(self, mock_load, mock_retrain):
        """When retraining is not due, init proceeds without warning."""
        import ml_gate

        mock_classifier = MagicMock()
        mock_classifier.metadata = {"version": "test_v1"}
        mock_classifier.model_path = "/models/test"
        mock_load.return_value = mock_classifier

        with patch.multiple(
            ml_gate,
            ML_GATE_ENABLED=True,
            ML_CONFIDENCE_THRESHOLD=0.5,
            ML_MIN_RISK_SCALE=0.5,
            ML_GB_WEIGHT=0.7,
            ML_LLM_WEIGHT=0.3,
            ML_RETRAIN_INTERVAL_DAYS=30,
        ):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                ml_gate.init(["BTCUSDT"])
                # The model should be loaded (check inside context)
                self.assertIn("BTCUSDT", ml_gate._classifiers)

        # should_retrain was called and returned False
        mock_retrain.assert_called_once_with("BTCUSDT", ml_gate.ML_MODEL_DIR)

    @patch("ml_trainer.RegimeClassifier.load_latest",
           side_effect=FileNotFoundError("No model"))
    def test_init_model_load_failure_skips_retrain_check(self, mock_load):
        """When model loading fails, should_retrain is not called."""
        import ml_gate

        with patch.multiple(
            ml_gate,
            ML_GATE_ENABLED=True,
            ML_CONFIDENCE_THRESHOLD=0.5,
            ML_MIN_RISK_SCALE=0.5,
            ML_GB_WEIGHT=0.7,
            ML_LLM_WEIGHT=0.3,
            ML_RETRAIN_INTERVAL_DAYS=30,
        ):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                with patch("ml_trainer.should_retrain") as mock_retrain:
                    ml_gate.init(["BTCUSDT"])
                    mock_retrain.assert_not_called()

    def test_init_disabled_skips_all(self):
        """When ML_GATE_ENABLED=false, init does nothing."""
        import ml_gate

        with patch.multiple(ml_gate, ML_GATE_ENABLED=False):
            with patch("ml_trainer.RegimeClassifier.load_latest") as mock_load:
                ml_gate.init(["BTCUSDT"])
                mock_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
