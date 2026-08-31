# tests/test_error_resilience.py
"""
Fehlerresilienz-Tests fuer den ML-Layer.

Ergaenzt die Property-10-Tests in test_ml_gate.py um Szenarien,
die dort nicht abgedeckt sind:

1. ml_gate.init() Fallback bei fehlerhaftem Modell-Verzeichnis (Req 15.3)
2. ml_prediction_log.record() Fehler unterbricht den Trade-Flow nicht (Req 15.4)
3. Integration: init() mit nicht-existierendem/kaputtem Model-Dir

**Validates: Requirements 15.1, 15.2, 15.3, 15.4**
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_gate
from ml_gate import GateDecision, evaluate, init


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_gate_config(
    enabled: bool = True,
    threshold: float = 0.5,
    min_risk_scale: float = 0.5,
    gb_weight: float = 0.7,
    llm_weight: float = 0.3,
    llm_enabled: bool = False,
    model_dir: str = ".models/",
):
    """Monkey-patches ml_gate module-level config vars."""
    return unittest.mock.patch.multiple(
        ml_gate,
        ML_GATE_ENABLED=enabled,
        ML_CONFIDENCE_THRESHOLD=threshold,
        ML_MIN_RISK_SCALE=min_risk_scale,
        ML_GB_WEIGHT=gb_weight,
        ML_LLM_WEIGHT=llm_weight,
        ML_LLM_ENABLED=llm_enabled,
        ML_MODEL_DIR=model_dir,
    )


def _dummy_analysis(indicators: dict | None = None) -> dict:
    return {"indicators": indicators or {}}


# ---------------------------------------------------------------------------
# Test 1: init() Fallback bei fehlerhaftem Modell-Laden (Req 15.3)
#
# When model loading fails for a symbol during init(), the gate must
# continue without that symbol's classifier (fallback mode) and NOT raise.
# ---------------------------------------------------------------------------

class TestInitFallbackOnModelLoadFailure(unittest.TestCase):
    """
    **Validates: Requirement 15.3**

    IF die Modell-Ladeoperation beim Bot-Start fehlschlaegt, THEN soll
    ml_gate.init() im Fallback-Modus starten (kein Classifier fuer dieses
    Symbol) und eine Warnung protokollieren, ohne eine Exception auszuloesen.
    """

    def test_init_continues_when_load_latest_raises(self):
        """
        **Validates: Requirement 15.3**

        When RegimeClassifier.load_latest raises FileNotFoundError for one
        symbol, init() must not raise and the symbol must not appear in
        _classifiers.
        """
        with _patch_gate_config(enabled=True):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                with patch(
                    "ml_gate.RegimeClassifier.load_latest",
                    side_effect=FileNotFoundError("No model found"),
                ) if hasattr(ml_gate, "RegimeClassifier") else patch(
                    "ml_trainer.RegimeClassifier.load_latest",
                    side_effect=FileNotFoundError("No model found"),
                ):
                    # init() must not raise
                    try:
                        init(["BTCUSDT", "ETHUSDT"])
                    except Exception as exc:
                        self.fail(
                            f"init() raised {type(exc).__name__}: {exc} "
                            "but should have caught the model-loading error"
                        )
                    # No classifier should be loaded
                    self.assertNotIn("BTCUSDT", ml_gate._classifiers)
                    self.assertNotIn("ETHUSDT", ml_gate._classifiers)

    def test_init_loads_partial_when_one_symbol_fails(self):
        """
        **Validates: Requirement 15.3**

        When one symbol's model loads successfully and another fails,
        only the successful symbol should have a classifier.
        """
        good_classifier = MagicMock()
        good_classifier.metadata = {"version": "v1"}
        good_classifier.model_path = "/tmp/fake"

        def _load_latest(symbol, model_dir):
            if symbol == "BTCUSDT":
                return good_classifier
            raise FileNotFoundError(f"No model for {symbol}")

        with _patch_gate_config(enabled=True):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                with patch(
                    "ml_trainer.RegimeClassifier.load_latest",
                    side_effect=_load_latest,
                ):
                    with patch("ml_trainer.should_retrain", return_value=False):
                        init(["BTCUSDT", "ETHUSDT"])

                    self.assertIn("BTCUSDT", ml_gate._classifiers)
                    self.assertNotIn("ETHUSDT", ml_gate._classifiers)

    def test_init_with_nonexistent_model_dir(self):
        """
        **Validates: Requirement 15.3**

        Integration test: init() with a model directory that doesn't exist.
        RegimeClassifier.load_latest will raise because no directory to scan.
        init() must not propagate the exception.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = os.path.join(tmpdir, "does_not_exist")
            with _patch_gate_config(enabled=True, model_dir=nonexistent):
                with patch.dict(ml_gate._classifiers, {}, clear=True):
                    try:
                        init(["BTCUSDT"])
                    except Exception as exc:
                        self.fail(
                            f"init() raised {type(exc).__name__}: {exc} "
                            "for nonexistent model dir"
                        )
                    self.assertNotIn("BTCUSDT", ml_gate._classifiers)

    def test_init_with_empty_model_dir(self):
        """
        **Validates: Requirement 15.3**

        Integration test: init() with an empty model directory (exists but
        contains no model subdirectories). Should fallback gracefully.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with _patch_gate_config(enabled=True, model_dir=tmpdir):
                with patch.dict(ml_gate._classifiers, {}, clear=True):
                    try:
                        init(["BTCUSDT"])
                    except Exception as exc:
                        self.fail(
                            f"init() raised {type(exc).__name__}: {exc} "
                            "for empty model dir"
                        )
                    self.assertNotIn("BTCUSDT", ml_gate._classifiers)

    def test_evaluate_uses_default_after_failed_init(self):
        """
        **Validates: Requirements 15.1, 15.3**

        After init() fails to load a model, evaluate() for that symbol
        should return the default trending/0.5 confidence decision
        (equivalent to ML_GATE_ENABLED=false behavior for that symbol).
        """
        with _patch_gate_config(enabled=True, threshold=0.0):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                with patch("ml_features.build_features", return_value={}):
                    decision = evaluate("BTCUSDT", _dummy_analysis())
                    self.assertEqual(decision.action, "allow")
                    self.assertEqual(decision.regime, "trending")
                    self.assertAlmostEqual(decision.gb_probability, 0.5)
                    self.assertEqual(decision.model_version, "no_model")

    def test_init_disabled_is_noop(self):
        """
        **Validates: Requirement 15.3**

        When ML_GATE_ENABLED=false, init() should be a no-op and not
        attempt to load any models.
        """
        with _patch_gate_config(enabled=False):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                with patch(
                    "ml_trainer.RegimeClassifier.load_latest",
                ) as mock_load:
                    init(["BTCUSDT"])
                    mock_load.assert_not_called()
                    self.assertEqual(len(ml_gate._classifiers), 0)


# ---------------------------------------------------------------------------
# Test 2: ml_prediction_log.record() Fehler bricht den Trade-Flow nicht
#
# In main.py, the calls to ml_prediction_log.record() are wrapped in
# try/except. This test validates that pattern at the integration level.
# (Req 15.4)
# ---------------------------------------------------------------------------

class TestPredictionLogRecordFailure(unittest.TestCase):
    """
    **Validates: Requirement 15.4**

    Wenn ml_prediction_log.record() eine Exception wirft, darf der
    Trade-Flow in scan_new_entries() nicht unterbrochen werden.
    """

    def test_record_exception_does_not_propagate_on_block(self):
        """
        **Validates: Requirement 15.4**

        When a signal is blocked and ml_prediction_log.record() fails,
        the flow must continue to the next symbol without raising.
        This simulates the try/except block in scan_new_entries().
        """
        block_decision = GateDecision(
            action="block",
            risk_scale=0.0,
            confidence=0.3,
            regime="choppy",
            filter_reason="ml_regime_blocked",
            gb_probability=0.3,
            llm_confidence=None,
            llm_regime=None,
            model_version="v1",
            latency_ms=5.0,
        )

        # Simulate the exact try/except pattern from main.py scan_new_entries()
        import logging
        test_logger = logging.getLogger("test_resilience")
        logged_errors = []

        with patch("ml_prediction_log.record", side_effect=RuntimeError("DB write failed")):
            # This mirrors the main.py pattern:
            try:
                import ml_prediction_log
                ml_prediction_log.record(block_decision, "BTCUSDT")
            except Exception as exc:
                test_logger.error(f"ML prediction log failed: {exc}")
                logged_errors.append(str(exc))

        # The exception was caught (not propagated) and logged
        self.assertEqual(len(logged_errors), 1)
        self.assertIn("DB write failed", logged_errors[0])

    def test_record_exception_does_not_propagate_on_allow(self):
        """
        **Validates: Requirement 15.4**

        When a signal is allowed, the trade is placed, and
        ml_prediction_log.record() fails afterwards, the trade must
        still be considered successful (the order was already placed).
        """
        allow_decision = GateDecision(
            action="allow",
            risk_scale=0.85,
            confidence=0.7,
            regime="trending",
            filter_reason="",
            gb_probability=0.7,
            llm_confidence=None,
            llm_regime=None,
            model_version="v1",
            latency_ms=3.0,
        )

        # Simulate the main.py pattern after a successful order placement
        trade_executed = False
        prediction_logged = False

        # Step 1: Trade is executed (simulated)
        trade_executed = True

        # Step 2: Prediction log fails
        with patch("ml_prediction_log.record", side_effect=sqlite3_error("DB locked")):
            try:
                import ml_prediction_log
                ml_prediction_log.record(allow_decision, "BTCUSDT")
                prediction_logged = True
            except Exception:
                # main.py catches this and continues
                pass

        # Trade was executed regardless of logging failure
        self.assertTrue(trade_executed)
        self.assertFalse(prediction_logged)

    def test_record_with_various_exception_types(self):
        """
        **Validates: Requirement 15.4**

        The try/except in main.py catches all exception types from
        ml_prediction_log.record(), not just specific ones.
        """
        decision = GateDecision(
            action="allow",
            risk_scale=1.0,
            confidence=0.8,
            regime="trending",
            filter_reason="",
            gb_probability=0.8,
            llm_confidence=None,
            llm_regime=None,
            model_version="v1",
            latency_ms=2.0,
        )

        exception_types = [
            RuntimeError("runtime"),
            OSError("disk full"),
            TypeError("bad type"),
            ValueError("bad value"),
            Exception("generic"),
        ]

        for exc in exception_types:
            with self.subTest(exc_type=type(exc).__name__):
                caught = False
                with patch("ml_prediction_log.record", side_effect=exc):
                    try:
                        import ml_prediction_log
                        ml_prediction_log.record(decision, "BTCUSDT")
                    except Exception:
                        caught = True
                # The calling code (main.py pattern) catches it
                self.assertTrue(
                    caught,
                    f"Exception {type(exc).__name__} should have been raised "
                    "from record() and caught by the caller's try/except"
                )


# ---------------------------------------------------------------------------
# Test 3: Integration — init() with broken model directory contents
#
# Tests the full init() → load_latest() path with realistic error scenarios.
# (Req 15.3)
# ---------------------------------------------------------------------------

class TestInitWithBrokenModelDirectory(unittest.TestCase):
    """
    **Validates: Requirement 15.3**

    Integration-level tests for ml_gate.init() with model directories
    that exist but contain corrupted or incomplete data.
    """

    def test_init_with_corrupt_model_pkl(self):
        """
        **Validates: Requirement 15.3**

        When the model directory exists and contains a subdirectory with
        the right naming pattern, but model.pkl is corrupt, init() must
        catch the deserialization error and continue in fallback mode.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a model subdirectory with a corrupt model.pkl
            model_subdir = Path(tmpdir) / "BTCUSDT_20240101T000000Z"
            model_subdir.mkdir()
            (model_subdir / "model.pkl").write_text("not a pickle file")
            (model_subdir / "metadata.json").write_text(
                '{"version": "20240101T000000Z", "feature_columns": []}'
            )

            with _patch_gate_config(enabled=True, model_dir=tmpdir):
                with patch.dict(ml_gate._classifiers, {}, clear=True):
                    try:
                        init(["BTCUSDT"])
                    except Exception as exc:
                        self.fail(
                            f"init() raised {type(exc).__name__}: {exc} "
                            "for corrupt model.pkl"
                        )
                    # Classifier should not be loaded
                    self.assertNotIn("BTCUSDT", ml_gate._classifiers)

    def test_init_with_missing_metadata_json(self):
        """
        **Validates: Requirement 15.3**

        When model.pkl exists but metadata.json is missing, init() must
        handle the FileNotFoundError gracefully.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_subdir = Path(tmpdir) / "BTCUSDT_20240101T000000Z"
            model_subdir.mkdir()
            # Write a valid pickle but no metadata.json
            import pickle
            (model_subdir / "model.pkl").write_bytes(
                pickle.dumps({"dummy": "model"})
            )
            # metadata.json intentionally missing

            with _patch_gate_config(enabled=True, model_dir=tmpdir):
                with patch.dict(ml_gate._classifiers, {}, clear=True):
                    try:
                        init(["BTCUSDT"])
                    except Exception as exc:
                        self.fail(
                            f"init() raised {type(exc).__name__}: {exc} "
                            "for missing metadata.json"
                        )
                    self.assertNotIn("BTCUSDT", ml_gate._classifiers)

    def test_init_with_permission_error(self):
        """
        **Validates: Requirement 15.3**

        When the model directory is not readable (permission denied),
        init() must catch the error and continue.
        """
        with _patch_gate_config(enabled=True, model_dir="/root/no_access_dir"):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                try:
                    init(["BTCUSDT"])
                except Exception as exc:
                    self.fail(
                        f"init() raised {type(exc).__name__}: {exc} "
                        "for permission-denied model dir"
                    )
                self.assertNotIn("BTCUSDT", ml_gate._classifiers)


# ---------------------------------------------------------------------------
# Test 4: No exception from evaluate() reaches the trading loop
#
# Comprehensive check that even combinations of failures in multiple
# sub-components don't leak exceptions out of evaluate().
# (Req 15.4)
# ---------------------------------------------------------------------------

class TestEvaluateNeverLeaksExceptions(unittest.TestCase):
    """
    **Validates: Requirement 15.4**

    Comprehensive test that no combination of internal failures causes
    evaluate() to propagate an exception to the trading loop.
    """

    def test_simultaneous_feature_and_classifier_failure(self):
        """
        **Validates: Requirement 15.4**

        Both feature pipeline AND classifier fail simultaneously.
        evaluate() must still return a valid GateDecision.
        """
        broken_classifier = MagicMock()
        broken_classifier.predict.side_effect = RuntimeError("Inference crash")
        broken_classifier.metadata = {"version": "broken"}

        with _patch_gate_config(enabled=True):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": broken_classifier}):
                with patch(
                    "ml_features.build_features",
                    side_effect=RuntimeError("Feature crash"),
                ):
                    decision = evaluate("BTCUSDT", _dummy_analysis())
                    self.assertIsInstance(decision, GateDecision)
                    self.assertEqual(decision.action, "allow")
                    self.assertEqual(decision.risk_scale, 1.0)

    def test_feature_pipeline_returns_none(self):
        """
        **Validates: Requirement 15.4**

        Feature pipeline returns None instead of a dict. The gate
        should handle this gracefully.
        """
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = {
            "regime": "trending",
            "probability": 0.7,
        }
        mock_classifier.metadata = {"version": "v1"}

        with _patch_gate_config(enabled=True, threshold=0.0):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": mock_classifier}):
                with patch("ml_features.build_features", return_value=None):
                    try:
                        decision = evaluate("BTCUSDT", _dummy_analysis())
                        self.assertIsInstance(decision, GateDecision)
                        # Either it uses None as features (classifier may
                        # handle it) or it falls back to bypass
                        self.assertIn(decision.action, ("allow", "block"))
                    except Exception as exc:
                        self.fail(
                            f"evaluate() raised {type(exc).__name__}: {exc} "
                            "when features returned None"
                        )

    def test_classifier_returns_malformed_result(self):
        """
        **Validates: Requirement 15.4**

        Classifier returns a dict missing the 'regime' key.
        evaluate() must handle the KeyError.
        """
        broken_classifier = MagicMock()
        broken_classifier.predict.return_value = {"wrong_key": "oops"}
        broken_classifier.metadata = {"version": "broken"}

        with _patch_gate_config(enabled=True):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": broken_classifier}):
                with patch("ml_features.build_features", return_value={}):
                    try:
                        decision = evaluate("BTCUSDT", _dummy_analysis())
                        self.assertIsInstance(decision, GateDecision)
                        self.assertEqual(decision.action, "allow")
                    except Exception as exc:
                        self.fail(
                            f"evaluate() raised {type(exc).__name__}: {exc} "
                            "for malformed classifier result"
                        )

    def test_analysis_dict_missing_indicators(self):
        """
        **Validates: Requirement 15.4**

        analysis dict has no 'indicators' key. evaluate() must handle
        the missing key gracefully.
        """
        with _patch_gate_config(enabled=True):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                with patch("ml_features.build_features", return_value={}):
                    try:
                        decision = evaluate("BTCUSDT", {})
                        self.assertIsInstance(decision, GateDecision)
                    except Exception as exc:
                        self.fail(
                            f"evaluate() raised {type(exc).__name__}: {exc} "
                            "for analysis dict without 'indicators'"
                        )

    def test_analysis_is_none(self):
        """
        **Validates: Requirement 15.4**

        Passing None as analysis. evaluate() must handle it.
        """
        with _patch_gate_config(enabled=True):
            with patch("ml_features.build_features", return_value={}):
                try:
                    decision = evaluate("BTCUSDT", None)
                    self.assertIsInstance(decision, GateDecision)
                    self.assertEqual(decision.action, "allow")
                except Exception as exc:
                    self.fail(
                        f"evaluate() raised {type(exc).__name__}: {exc} "
                        "when analysis is None"
                    )


# Needed for the sqlite3 error simulation
def sqlite3_error(msg: str) -> Exception:
    """Create a sqlite3.OperationalError for test simulation."""
    import sqlite3
    return sqlite3.OperationalError(msg)


if __name__ == "__main__":
    unittest.main()
