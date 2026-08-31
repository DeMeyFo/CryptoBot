# tests/test_ml_prediction_log.py
"""
Unit-Tests und Property-Tests fuer ml_prediction_log.py
Validates: Requirements 7.1, 7.4, 14.3, 14.4
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_decision(**overrides) -> SimpleNamespace:
    """Create a mock GateDecision-like object with sensible defaults."""
    defaults = dict(
        action="allow",
        risk_scale=0.75,
        confidence=0.82,
        regime="trending",
        filter_reason="",
        gb_probability=0.85,
        llm_confidence=None,
        llm_regime=None,
        model_version="20250101T120000Z",
        latency_ms=12.5,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _TempDBTestCase(unittest.TestCase):
    """Base class that redirects ml_prediction_log to a temporary SQLite DB."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = self._tmp.name

        # Monkey-patch the DB_PATH that ml_prediction_log reads from config
        import ml_prediction_log
        self._original_db_path = ml_prediction_log.DB_PATH
        ml_prediction_log.DB_PATH = self._db_path

        # Also patch config.DB_PATH so _conn() picks up the temp path
        import config
        self._original_config_db_path = config.DB_PATH
        config.DB_PATH = self._db_path

        # Reload _conn closure by re-importing won't help since _conn reads
        # DB_PATH at call time from the module-level variable. The module
        # uses `from config import DB_PATH` which binds at import time, so
        # we patched ml_prediction_log.DB_PATH directly above.

        # Initialize the ML tables
        ml_prediction_log.init_ml_tables()

    def tearDown(self):
        import ml_prediction_log
        import config
        ml_prediction_log.DB_PATH = self._original_db_path
        config.DB_PATH = self._original_config_db_path
        os.unlink(self._db_path)

    def _raw_conn(self) -> sqlite3.Connection:
        """Direct connection to the temp DB for verification queries."""
        return sqlite3.connect(self._db_path)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestRecord(_TempDBTestCase):
    """Test: record() creates correct record with all required fields.

    Validates: Requirement 7.1
    """

    def test_record_returns_positive_id(self):
        """record() returns a positive integer prediction_id."""
        import ml_prediction_log
        decision = _make_decision()
        pid = ml_prediction_log.record(decision, "BTCUSDT")
        self.assertIsInstance(pid, int)
        self.assertGreater(pid, 0)

    def test_record_persists_all_required_fields(self):
        """record() persists symbol, prediction_at, predicted_regime,
        confidence_score, gb_probability, action, and risk_scale."""
        import ml_prediction_log
        decision = _make_decision(
            regime="choppy",
            confidence=0.42,
            gb_probability=0.38,
            action="block",
            risk_scale=0.0,
            filter_reason="ml_regime_blocked",
            model_version="v2",
            latency_ms=5.1,
        )
        pid = ml_prediction_log.record(decision, "ETHUSDT")

        conn = self._raw_conn()
        row = conn.execute(
            "SELECT * FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row)
        columns = self._get_columns("ml_predictions")
        record = dict(zip(columns, row))

        # Required fields
        self.assertEqual(record["symbol"], "ETHUSDT")
        self.assertIsNotNone(record["prediction_at"])
        self.assertEqual(record["predicted_regime"], "choppy")
        self.assertAlmostEqual(record["confidence_score"], 0.42)
        self.assertAlmostEqual(record["gb_probability"], 0.38)
        self.assertEqual(record["action"], "block")
        self.assertAlmostEqual(record["risk_scale"], 0.0)
        self.assertEqual(record["filter_reason"], "ml_regime_blocked")
        self.assertEqual(record["model_version"], "v2")
        self.assertAlmostEqual(record["latency_ms"], 5.1)

    def test_record_with_top_features(self):
        """record() correctly serializes top_features as JSON."""
        import ml_prediction_log
        decision = _make_decision()
        features = {"adx": 45.2, "rsi": 62.1, "atr_pct": 1.8}
        pid = ml_prediction_log.record(decision, "BTCUSDT", top_features=features)

        conn = self._raw_conn()
        row = conn.execute(
            "SELECT top_features FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        stored = json.loads(row[0])
        self.assertEqual(stored, features)

    def test_record_without_top_features(self):
        """record() stores NULL when top_features is None."""
        import ml_prediction_log
        decision = _make_decision()
        pid = ml_prediction_log.record(decision, "BTCUSDT", top_features=None)

        conn = self._raw_conn()
        row = conn.execute(
            "SELECT top_features FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        self.assertIsNone(row[0])

    def test_record_with_llm_fields(self):
        """record() persists llm_confidence and llm_regime when provided."""
        import ml_prediction_log
        decision = _make_decision(
            llm_confidence=0.7,
            llm_regime="trending",
        )
        pid = ml_prediction_log.record(decision, "SOLUSDT")

        conn = self._raw_conn()
        columns = self._get_columns("ml_predictions")
        row = conn.execute(
            "SELECT * FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        record = dict(zip(columns, row))
        self.assertAlmostEqual(record["llm_confidence"], 0.7)
        self.assertEqual(record["llm_regime"], "trending")

    def test_record_multiple_predictions(self):
        """Multiple record() calls create distinct rows with incrementing IDs."""
        import ml_prediction_log
        decision = _make_decision()
        id1 = ml_prediction_log.record(decision, "BTCUSDT")
        id2 = ml_prediction_log.record(decision, "ETHUSDT")
        id3 = ml_prediction_log.record(decision, "BTCUSDT")

        self.assertNotEqual(id1, id2)
        self.assertNotEqual(id2, id3)
        self.assertGreater(id2, id1)
        self.assertGreater(id3, id2)

    def _get_columns(self, table: str) -> list[str]:
        conn = self._raw_conn()
        cols = [row[1] for row in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()]
        conn.close()
        return cols


class TestUpdateOutcome(_TempDBTestCase):
    """Test: update_outcome() adds actual_regime and actual_performance.

    Validates: Requirement 7.2
    """

    def test_update_outcome_sets_fields(self):
        """update_outcome() correctly sets actual_regime, actual_performance,
        and outcome_updated_at."""
        import ml_prediction_log
        decision = _make_decision(regime="trending")
        pid = ml_prediction_log.record(decision, "BTCUSDT")

        ml_prediction_log.update_outcome(pid, "trending", 0.035)

        conn = self._raw_conn()
        row = conn.execute(
            "SELECT actual_regime, actual_performance, outcome_updated_at "
            "FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        self.assertEqual(row[0], "trending")
        self.assertAlmostEqual(row[1], 0.035)
        self.assertIsNotNone(row[2])  # outcome_updated_at is set

    def test_update_outcome_with_negative_performance(self):
        """update_outcome() handles negative performance values."""
        import ml_prediction_log
        decision = _make_decision(regime="choppy", action="allow")
        pid = ml_prediction_log.record(decision, "ADAUSDT")

        ml_prediction_log.update_outcome(pid, "choppy", -0.02)

        conn = self._raw_conn()
        row = conn.execute(
            "SELECT actual_regime, actual_performance FROM ml_predictions WHERE id=?",
            (pid,)
        ).fetchone()
        conn.close()

        self.assertEqual(row[0], "choppy")
        self.assertAlmostEqual(row[1], -0.02)

    def test_update_outcome_only_affects_target_row(self):
        """update_outcome() only modifies the row with the given prediction_id."""
        import ml_prediction_log
        d1 = _make_decision(regime="trending")
        d2 = _make_decision(regime="choppy")
        pid1 = ml_prediction_log.record(d1, "BTCUSDT")
        pid2 = ml_prediction_log.record(d2, "ETHUSDT")

        ml_prediction_log.update_outcome(pid1, "trending", 0.05)

        conn = self._raw_conn()
        row2 = conn.execute(
            "SELECT actual_regime, actual_performance FROM ml_predictions WHERE id=?",
            (pid2,)
        ).fetchone()
        conn.close()

        self.assertIsNone(row2[0])
        self.assertIsNone(row2[1])


class TestQueryAccuracy(_TempDBTestCase):
    """Test: query_accuracy() computes correct accuracy.

    Validates: Requirement 7.3
    """

    def test_empty_range_returns_zero(self):
        """query_accuracy() returns zero accuracy when no records exist."""
        import ml_prediction_log
        result = ml_prediction_log.query_accuracy("2025-01-01", "2025-12-31")
        self.assertEqual(result["total"], 0)
        self.assertAlmostEqual(result["accuracy"], 0.0)
        self.assertAlmostEqual(result["block_rate"], 0.0)

    def test_accuracy_computation(self):
        """query_accuracy() correctly computes accuracy from predictions
        with known outcomes."""
        import ml_prediction_log

        # Insert 4 predictions: 3 correct, 1 incorrect
        decisions = [
            _make_decision(regime="trending", action="allow"),
            _make_decision(regime="choppy", action="block"),
            _make_decision(regime="trending", action="allow"),
            _make_decision(regime="trending", action="allow"),
        ]
        actual_regimes = ["trending", "choppy", "trending", "choppy"]

        pids = []
        for d in decisions:
            pids.append(ml_prediction_log.record(d, "BTCUSDT"))

        # Update outcomes
        for pid, actual in zip(pids, actual_regimes):
            ml_prediction_log.update_outcome(pid, actual, 0.01)

        result = ml_prediction_log.query_accuracy("2000-01-01", "2099-12-31")

        self.assertEqual(result["total"], 4)
        # 3 out of 4 correct
        self.assertAlmostEqual(result["accuracy"], 0.75)

    def test_block_rate_includes_pending(self):
        """query_accuracy() computes block_rate using total_with_pending
        (including predictions without outcomes)."""
        import ml_prediction_log

        # 2 predictions with outcomes, 1 without
        d_block = _make_decision(regime="choppy", action="block")
        d_allow = _make_decision(regime="trending", action="allow")
        d_pending = _make_decision(regime="trending", action="allow")

        pid1 = ml_prediction_log.record(d_block, "BTCUSDT")
        pid2 = ml_prediction_log.record(d_allow, "BTCUSDT")
        _pid3 = ml_prediction_log.record(d_pending, "BTCUSDT")  # no outcome

        ml_prediction_log.update_outcome(pid1, "choppy", -0.01)
        ml_prediction_log.update_outcome(pid2, "trending", 0.02)

        result = ml_prediction_log.query_accuracy("2000-01-01", "2099-12-31")

        # total = 2 (with outcomes), total_with_pending = 3
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["total_with_pending"], 3)
        # 1 block out of 3 total predictions
        self.assertAlmostEqual(result["block_rate"], 1.0 / 3.0, places=5)

    def test_accuracy_respects_date_range(self):
        """query_accuracy() only includes predictions within the date range."""
        import ml_prediction_log

        # We insert predictions and check they are filtered by date.
        # Since prediction_at is auto-set to now, all will be in the same
        # "today" range. We test that a far-future range returns 0.
        d = _make_decision(regime="trending", action="allow")
        pid = ml_prediction_log.record(d, "BTCUSDT")
        ml_prediction_log.update_outcome(pid, "trending", 0.01)

        # Date range far in the past should return nothing
        result = ml_prediction_log.query_accuracy("1990-01-01", "1990-12-31")
        self.assertEqual(result["total"], 0)


# ---------------------------------------------------------------------------
# Property 11: Vorhersage-Logging Vollstaendigkeit
# Validates: Requirements 7.1, 7.4
#
# For any GateDecision returned by ml_gate.evaluate(), the corresponding
# ml_prediction_log.record() call shall create a record in ml_predictions
# containing all required fields (symbol, prediction_at, predicted_regime,
# confidence_score, gb_probability, action, risk_scale).
# ---------------------------------------------------------------------------

class TestProperty11LoggingVollstaendigkeit(_TempDBTestCase):
    """
    **Validates: Requirements 7.1, 7.4**

    Property 11: Vorhersage-Logging Vollstaendigkeit --
    every GateDecision creates a complete record with all required fields.
    """

    REQUIRED_FIELDS = [
        "symbol", "prediction_at", "predicted_regime",
        "confidence_score", "gb_probability", "action", "risk_scale",
    ]

    def _get_columns(self, table: str) -> list[str]:
        conn = self._raw_conn()
        cols = [row[1] for row in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()]
        conn.close()
        return cols

    def _assert_record_complete(self, decision, symbol):
        """Helper: record a decision and assert all required fields are non-NULL."""
        import ml_prediction_log
        pid = ml_prediction_log.record(decision, symbol)

        conn = self._raw_conn()
        columns = self._get_columns("ml_predictions")
        row = conn.execute(
            "SELECT * FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row, f"No record found for prediction_id={pid}")
        record = dict(zip(columns, row))

        for field in self.REQUIRED_FIELDS:
            self.assertIsNotNone(
                record[field],
                f"Required field '{field}' is NULL for decision: {decision}"
            )

    def test_allow_decision_creates_complete_record(self):
        """An 'allow' GateDecision produces a record with all required fields."""
        decision = _make_decision(
            action="allow", regime="trending", confidence=0.9,
            gb_probability=0.88, risk_scale=0.95,
        )
        self._assert_record_complete(decision, "BTCUSDT")

    def test_block_decision_creates_complete_record(self):
        """A 'block' GateDecision produces a record with all required fields."""
        decision = _make_decision(
            action="block", regime="choppy", confidence=0.3,
            gb_probability=0.25, risk_scale=0.0,
            filter_reason="ml_regime_blocked",
        )
        self._assert_record_complete(decision, "ETHUSDT")

    def test_hold_decision_creates_complete_record(self):
        """A 'hold' GateDecision produces a record with all required fields."""
        decision = _make_decision(
            action="hold", regime="neutral", confidence=0.5,
            gb_probability=0.5, risk_scale=0.5,
            filter_reason="manual_hold",
        )
        self._assert_record_complete(decision, "SOLUSDT")

    def test_decision_with_llm_creates_complete_record(self):
        """A GateDecision with LLM data produces a complete record."""
        decision = _make_decision(
            action="allow", regime="trending", confidence=0.85,
            gb_probability=0.9, llm_confidence=0.7, llm_regime="trending",
            risk_scale=0.92,
        )
        self._assert_record_complete(decision, "ADAUSDT")

    def test_decision_with_zero_confidence_creates_complete_record(self):
        """A GateDecision with zero confidence still creates a complete record."""
        decision = _make_decision(
            action="block", regime="choppy", confidence=0.0,
            gb_probability=0.0, risk_scale=0.0,
        )
        self._assert_record_complete(decision, "BNBUSDT")

    def test_decision_with_model_version_annotated(self):
        """Every record includes the model_version annotation (Req 7.4)."""
        import ml_prediction_log
        decision = _make_decision(model_version="20250615T080000Z")
        pid = ml_prediction_log.record(decision, "BTCUSDT")

        conn = self._raw_conn()
        row = conn.execute(
            "SELECT model_version FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        self.assertEqual(row[0], "20250615T080000Z")


# ---------------------------------------------------------------------------
# Property 14: Datenbank-Schema-Kompatibilitaet
# Validates: Requirements 14.3, 14.4
#
# For any call to init_ml_tables(), existing tables (trades, signals,
# funding_events, consumed_entry_signals, trade_quantity_epochs) shall
# remain unchanged, and new tables (ml_predictions, ml_model_metadata)
# shall be created idempotently.
# ---------------------------------------------------------------------------

class TestProperty14SchemaKompatibilitaet(_TempDBTestCase):
    """
    **Validates: Requirements 14.3, 14.4**

    Property 14: Datenbank-Schema-Kompatibilitaet --
    existing tables remain unchanged, new tables are created idempotently.
    """

    def test_ml_predictions_table_exists(self):
        """init_ml_tables() creates the ml_predictions table."""
        conn = self._raw_conn()
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        self.assertIn("ml_predictions", tables)

    def test_ml_model_metadata_table_exists(self):
        """init_ml_tables() creates the ml_model_metadata table."""
        conn = self._raw_conn()
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        self.assertIn("ml_model_metadata", tables)

    def test_init_ml_tables_is_idempotent(self):
        """Calling init_ml_tables() twice does not raise and does not
        lose existing data."""
        import ml_prediction_log

        # Insert a record
        decision = _make_decision()
        pid = ml_prediction_log.record(decision, "BTCUSDT")

        # Call init_ml_tables() again (idempotent)
        ml_prediction_log.init_ml_tables()

        # Record still exists
        conn = self._raw_conn()
        row = conn.execute(
            "SELECT id FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)

    def test_existing_tables_not_modified(self):
        """init_ml_tables() does not create or modify the existing
        application tables (trades, signals, etc.)."""
        import ml_prediction_log

        # Before: record existing tables
        conn = self._raw_conn()
        tables_before = set(row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall())
        conn.close()

        # These are the application tables that should NOT be created
        # by init_ml_tables(). If they don't exist, init_ml_tables()
        # must not create them.
        existing_app_tables = {
            "trades", "signals", "funding_events",
            "consumed_entry_signals", "trade_quantity_epochs",
        }

        # Re-call init_ml_tables
        ml_prediction_log.init_ml_tables()

        conn = self._raw_conn()
        tables_after = set(row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall())
        conn.close()

        # The only new tables should be ml_predictions and ml_model_metadata
        newly_created = tables_after - tables_before
        for table in newly_created:
            self.assertNotIn(table, existing_app_tables,
                             f"init_ml_tables() should not create '{table}'")

    def test_ml_predictions_schema_has_required_columns(self):
        """ml_predictions table has all required columns from the design."""
        expected_columns = {
            "id", "symbol", "prediction_at", "predicted_regime",
            "confidence_score", "gb_probability", "llm_confidence",
            "llm_regime", "top_features", "action", "risk_scale",
            "filter_reason", "model_version", "training_period",
            "latency_ms", "actual_regime", "actual_performance",
            "outcome_updated_at",
        }
        conn = self._raw_conn()
        actual_columns = set(
            row[1] for row in conn.execute(
                "PRAGMA table_info(ml_predictions)"
            ).fetchall()
        )
        conn.close()
        self.assertTrue(
            expected_columns.issubset(actual_columns),
            f"Missing columns: {expected_columns - actual_columns}"
        )

    def test_ml_model_metadata_schema_has_required_columns(self):
        """ml_model_metadata table has all required columns from the design."""
        expected_columns = {
            "id", "symbol", "version", "model_path",
            "training_start", "training_end",
            "validation_start", "validation_end",
            "feature_columns", "hyperparameters",
            "accuracy", "precision_score", "recall",
            "f1_score", "auc_roc", "data_hash",
            "trained_at", "is_active",
        }
        conn = self._raw_conn()
        actual_columns = set(
            row[1] for row in conn.execute(
                "PRAGMA table_info(ml_model_metadata)"
            ).fetchall()
        )
        conn.close()
        self.assertTrue(
            expected_columns.issubset(actual_columns),
            f"Missing columns: {expected_columns - actual_columns}"
        )

    def test_indices_created(self):
        """init_ml_tables() creates the expected indices."""
        expected_indices = {
            "idx_ml_predictions_symbol_time",
            "idx_ml_predictions_outcome",
            "idx_ml_model_metadata_symbol",
        }
        conn = self._raw_conn()
        actual_indices = set(
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        )
        conn.close()
        self.assertTrue(
            expected_indices.issubset(actual_indices),
            f"Missing indices: {expected_indices - actual_indices}"
        )

    def test_ml_model_metadata_unique_constraint(self):
        """ml_model_metadata enforces UNIQUE(symbol, version)."""
        conn = self._raw_conn()
        conn.execute("""
            INSERT INTO ml_model_metadata
                (symbol, version, model_path, trained_at)
            VALUES ('BTCUSDT', 'v1', '/path/model.pkl', '2025-01-01T00:00:00')
        """)
        conn.commit()

        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO ml_model_metadata
                    (symbol, version, model_path, trained_at)
                VALUES ('BTCUSDT', 'v1', '/path/model2.pkl', '2025-06-01T00:00:00')
            """)
        conn.close()

    def test_ml_predictions_check_constraints(self):
        """ml_predictions enforces CHECK constraints on predicted_regime and action."""
        conn = self._raw_conn()

        # Invalid regime should fail
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO ml_predictions
                    (symbol, prediction_at, predicted_regime, confidence_score,
                     gb_probability, action, risk_scale)
                VALUES ('BTC', '2025-01-01', 'invalid_regime', 0.5,
                        0.5, 'allow', 0.5)
            """)

        # Invalid action should fail
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO ml_predictions
                    (symbol, prediction_at, predicted_regime, confidence_score,
                     gb_probability, action, risk_scale)
                VALUES ('BTC', '2025-01-01', 'trending', 0.5,
                        0.5, 'invalid_action', 0.5)
            """)
        conn.close()


# ---------------------------------------------------------------------------
# Tests for training_period annotation (Anforderung 7.4)
# ---------------------------------------------------------------------------

class TestRecordTrainingPeriod(_TempDBTestCase):
    """Test: record() annotates training_period when provided.

    Validates: Requirement 7.4
    """

    def _get_columns(self, table: str) -> list[str]:
        conn = self._raw_conn()
        cols = [row[1] for row in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()]
        conn.close()
        return cols

    def test_record_with_training_period(self):
        """record() persists training_period when provided."""
        import ml_prediction_log
        decision = _make_decision(model_version="20250601T120000Z")
        period = '{"start": "2024-01-01", "end": "2024-06-30"}'
        pid = ml_prediction_log.record(
            decision, "BTCUSDT", training_period=period
        )

        conn = self._raw_conn()
        columns = self._get_columns("ml_predictions")
        row = conn.execute(
            "SELECT * FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        record = dict(zip(columns, row))
        self.assertEqual(record["training_period"], period)
        self.assertEqual(record["model_version"], "20250601T120000Z")

    def test_record_without_training_period(self):
        """record() stores NULL when training_period is not provided."""
        import ml_prediction_log
        decision = _make_decision()
        pid = ml_prediction_log.record(decision, "BTCUSDT")

        conn = self._raw_conn()
        columns = self._get_columns("ml_predictions")
        row = conn.execute(
            "SELECT * FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        record = dict(zip(columns, row))
        self.assertIsNone(record["training_period"])


# ---------------------------------------------------------------------------
# Tests for outcome tracking (Anforderung 7.2)
# backfill_outcomes(), get_pending_outcomes(), _backfill_single_prediction()
# ---------------------------------------------------------------------------

class TestOutcomeTracking(_TempDBTestCase):
    """Test: Outcome tracking correctly updates predictions retroactively.

    Validates: Requirements 7.2, 7.4
    """

    def _get_columns(self, table: str) -> list[str]:
        conn = self._raw_conn()
        cols = [row[1] for row in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()]
        conn.close()
        return cols

    def _insert_old_prediction(self, symbol: str, hours_ago: float,
                               regime: str = "trending") -> int:
        """Insert a prediction with a timestamp in the past."""
        from datetime import timedelta
        import ml_prediction_log

        past_time = (
            datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        ).isoformat()

        conn = self._raw_conn()
        cursor = conn.execute("""
            INSERT INTO ml_predictions
                (symbol, prediction_at, predicted_regime, confidence_score,
                 gb_probability, action, risk_scale, model_version, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, past_time, regime, 0.8, 0.85, "allow", 0.9,
              "20250101T120000Z", 10.0))
        conn.commit()
        pid = cursor.lastrowid
        conn.close()
        return pid

    def test_get_pending_outcomes_returns_old_predictions(self):
        """get_pending_outcomes() finds predictions older than the horizon."""
        import ml_prediction_log

        # Insert a prediction from 48 hours ago (> 24h horizon)
        pid = self._insert_old_prediction("BTCUSDT", hours_ago=48)

        pending = ml_prediction_log.get_pending_outcomes(
            horizon_candles=24, timeframe_seconds=3600
        )
        ids = [p["id"] for p in pending]
        self.assertIn(pid, ids)

    def test_get_pending_outcomes_skips_recent(self):
        """get_pending_outcomes() does not return recent predictions."""
        import ml_prediction_log

        # Insert a prediction from 1 hour ago (< 24h horizon)
        self._insert_old_prediction("BTCUSDT", hours_ago=1)

        pending = ml_prediction_log.get_pending_outcomes(
            horizon_candles=24, timeframe_seconds=3600
        )
        self.assertEqual(len(pending), 0)

    def test_get_pending_outcomes_skips_already_updated(self):
        """get_pending_outcomes() does not return predictions with outcomes."""
        import ml_prediction_log

        pid = self._insert_old_prediction("BTCUSDT", hours_ago=48)
        ml_prediction_log.update_outcome(pid, "trending", 0.05)

        pending = ml_prediction_log.get_pending_outcomes(
            horizon_candles=24, timeframe_seconds=3600
        )
        ids = [p["id"] for p in pending]
        self.assertNotIn(pid, ids)

    def test_backfill_outcomes_updates_prediction(self):
        """backfill_outcomes() calls update_outcome() for eligible predictions."""
        import ml_prediction_log

        pid = self._insert_old_prediction("BTCUSDT", hours_ago=48)

        # Mock candle data: list of [ts, open, high, low, close, volume]
        # We need horizon_candles + 50 candles minimum.
        # Create candles with a clear trending pattern.
        base_price = 50000.0
        n_candles = 24 + 50
        candles = []
        for i in range(n_candles):
            price = base_price + i * 100  # trending upward
            candles.append([
                1700000000000 + i * 3600000,  # timestamp
                price - 50,                   # open
                price + 200,                  # high
                price - 200,                  # low
                price,                        # close
                1000,                         # volume
            ])

        def mock_get_candles(symbol, granularity, limit):
            return candles[:limit]

        stats = ml_prediction_log.backfill_outcomes(
            get_candles_fn=mock_get_candles,
            horizon_candles=24,
            timeframe_seconds=3600,
        )

        self.assertEqual(stats["updated"], 1)
        self.assertEqual(stats["errors"], 0)

        # Verify the prediction was updated
        conn = self._raw_conn()
        row = conn.execute(
            "SELECT actual_regime, actual_performance, outcome_updated_at "
            "FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row[0])  # actual_regime is set
        self.assertIsNotNone(row[1])  # actual_performance is set
        self.assertIsNotNone(row[2])  # outcome_updated_at is set
        self.assertIn(row[0], ("trending", "choppy"))
        self.assertGreater(row[1], 0.0)  # trending data -> positive performance

    def test_backfill_outcomes_no_pending(self):
        """backfill_outcomes() returns zeros when nothing is pending."""
        import ml_prediction_log

        stats = ml_prediction_log.backfill_outcomes(
            get_candles_fn=lambda s, g, l: [],
            horizon_candles=24,
            timeframe_seconds=3600,
        )
        self.assertEqual(stats["updated"], 0)
        self.assertEqual(stats["errors"], 0)

    def test_backfill_outcomes_handles_candle_errors(self):
        """backfill_outcomes() counts errors when candle fetching fails."""
        import ml_prediction_log

        self._insert_old_prediction("BTCUSDT", hours_ago=48)

        def failing_candles(symbol, granularity, limit):
            return []  # too few candles

        stats = ml_prediction_log.backfill_outcomes(
            get_candles_fn=failing_candles,
            horizon_candles=24,
            timeframe_seconds=3600,
        )
        self.assertEqual(stats["updated"], 0)
        self.assertGreater(stats["errors"], 0)

    def test_backfill_outcomes_never_raises(self):
        """backfill_outcomes() never raises, even when get_candles_fn throws."""
        import ml_prediction_log

        self._insert_old_prediction("BTCUSDT", hours_ago=48)

        def exploding_candles(symbol, granularity, limit):
            raise RuntimeError("API down")

        # Should not raise
        stats = ml_prediction_log.backfill_outcomes(
            get_candles_fn=exploding_candles,
            horizon_candles=24,
            timeframe_seconds=3600,
        )
        self.assertEqual(stats["updated"], 0)
        self.assertGreater(stats["errors"], 0)

    def test_update_outcome_correctly_updates_record(self):
        """update_outcome() correctly sets actual_regime, actual_performance,
        and outcome_updated_at for a previously recorded prediction."""
        import ml_prediction_log

        decision = _make_decision(
            regime="trending", confidence=0.85, model_version="v1"
        )
        pid = ml_prediction_log.record(
            decision, "BTCUSDT",
            training_period='{"start": "2024-01-01", "end": "2024-06-30"}',
        )

        # Before update: outcome fields should be NULL
        conn = self._raw_conn()
        columns = self._get_columns("ml_predictions")
        row = conn.execute(
            "SELECT * FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()
        record = dict(zip(columns, row))
        self.assertIsNone(record["actual_regime"])
        self.assertIsNone(record["actual_performance"])
        self.assertIsNone(record["outcome_updated_at"])

        # Update outcome
        ml_prediction_log.update_outcome(pid, "choppy", -0.015)

        # After update: outcome fields should be set
        conn = self._raw_conn()
        row = conn.execute(
            "SELECT * FROM ml_predictions WHERE id=?", (pid,)
        ).fetchone()
        conn.close()
        record = dict(zip(columns, row))
        self.assertEqual(record["actual_regime"], "choppy")
        self.assertAlmostEqual(record["actual_performance"], -0.015)
        self.assertIsNotNone(record["outcome_updated_at"])
        # model_version and training_period should still be present
        self.assertEqual(record["model_version"], "v1")
        self.assertEqual(
            record["training_period"],
            '{"start": "2024-01-01", "end": "2024-06-30"}',
        )


if __name__ == "__main__":
    unittest.main()
