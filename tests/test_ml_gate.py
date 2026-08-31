# tests/test_ml_gate.py
"""
Property-Tests fuer ml_gate.py — ML-Gate Orchestrator.

**Validates: Requirements 1.2, 1.3, 1.4, 2.1, 2.4, 4.3, 4.4, 9.3, 15.1, 15.2, 15.4**
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_gate
from ml_gate import (
    GateDecision,
    _BYPASS,
    _compute_risk_scale,
    evaluate,
    validate_config,
)

from hypothesis import given, settings, assume
from hypothesis import strategies as st


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
):
    """Context manager that monkey-patches ml_gate module-level config vars."""
    return unittest.mock.patch.multiple(
        ml_gate,
        ML_GATE_ENABLED=enabled,
        ML_CONFIDENCE_THRESHOLD=threshold,
        ML_MIN_RISK_SCALE=min_risk_scale,
        ML_GB_WEIGHT=gb_weight,
        ML_LLM_WEIGHT=llm_weight,
        ML_LLM_ENABLED=llm_enabled,
    )


def _make_mock_classifier(regime: str = "trending", probability: float = 0.8):
    """Create a mock classifier that returns a fixed regime and probability."""
    mock = MagicMock()
    mock.predict.return_value = {"regime": regime, "probability": probability}
    mock.metadata = {"version": "test_v1"}
    return mock


def _dummy_analysis(indicators: dict | None = None) -> dict:
    """Create a minimal analysis dict for evaluate()."""
    return {"indicators": indicators or {}}


# ---------------------------------------------------------------------------
# Property 1: ML-Gate-Bypass bei Deaktivierung
# Validates: Requirement 1.4
#
# When ML_GATE_ENABLED=false, evaluate() always returns action="allow"
# and risk_scale=1.0, regardless of symbol or analysis input.
# ---------------------------------------------------------------------------

class TestProperty1MLGateBypass(unittest.TestCase):
    """
    **Validates: Requirement 1.4**

    Property 1: ML-Gate-Bypass bei Deaktivierung — bei ML_GATE_ENABLED=false
    immer action='allow', risk_scale=1.0.
    """

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Nd")),
            min_size=3,
            max_size=12,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_disabled_gate_always_allows(self, symbol: str):
        """
        **Validates: Requirement 1.4**

        With ML_GATE_ENABLED=false, evaluate() must return the bypass decision
        for any symbol.
        """
        with _patch_gate_config(enabled=False):
            decision = evaluate(symbol, _dummy_analysis())
            self.assertEqual(decision.action, "allow")
            self.assertEqual(decision.risk_scale, 1.0)
            self.assertEqual(decision.confidence, 1.0)
            self.assertEqual(decision.regime, "trending")
            self.assertEqual(decision.filter_reason, "")
            self.assertEqual(decision.model_version, "none")

    def test_bypass_constant_values(self):
        """The _BYPASS constant must have the expected field values."""
        self.assertEqual(_BYPASS.action, "allow")
        self.assertEqual(_BYPASS.risk_scale, 1.0)
        self.assertEqual(_BYPASS.confidence, 1.0)
        self.assertEqual(_BYPASS.regime, "trending")
        self.assertEqual(_BYPASS.filter_reason, "")
        self.assertIsNone(_BYPASS.llm_confidence)
        self.assertIsNone(_BYPASS.llm_regime)


# ---------------------------------------------------------------------------
# Property 2: Choppy-Regime-Blockierung
# Validates: Requirement 1.2
#
# When the classifier predicts choppy and confidence < threshold,
# the gate must block the signal.
# ---------------------------------------------------------------------------

class TestProperty2ChoppyRegimeBlocking(unittest.TestCase):
    """
    **Validates: Requirement 1.2**

    Property 2: Choppy-Regime-Blockierung — choppy + niedrige Confidence -> block.
    """

    @given(
        gb_prob=st.floats(min_value=0.0, max_value=0.49),
        threshold=st.floats(min_value=0.5, max_value=1.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_choppy_low_confidence_blocks(self, gb_prob: float, threshold: float):
        """
        **Validates: Requirement 1.2**

        When regime is choppy and confidence < threshold, action must be 'block'.
        """
        assume(gb_prob < threshold)

        mock_classifier = _make_mock_classifier(regime="choppy", probability=gb_prob)

        with _patch_gate_config(enabled=True, threshold=threshold, llm_enabled=False):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": mock_classifier}):
                with patch("ml_features.build_features", return_value={}):
                    decision = evaluate("BTCUSDT", _dummy_analysis())
                    self.assertEqual(decision.action, "block")
                    self.assertEqual(decision.filter_reason, "ml_regime_blocked")
                    self.assertEqual(decision.risk_scale, 0.0)
                    self.assertEqual(decision.regime, "choppy")


# ---------------------------------------------------------------------------
# Property 3: Trending-Regime-Durchlass
# Validates: Requirement 1.3
#
# When the classifier predicts trending and confidence >= threshold,
# the gate must allow the signal.
# ---------------------------------------------------------------------------

class TestProperty3TrendingRegimeAllow(unittest.TestCase):
    """
    **Validates: Requirement 1.3**

    Property 3: Trending-Regime-Durchlass — trending + hohe Confidence -> allow.
    """

    @given(
        gb_prob=st.floats(min_value=0.5, max_value=1.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_trending_high_confidence_allows(self, gb_prob: float):
        """
        **Validates: Requirement 1.3**

        When regime is trending, action must be 'allow' regardless of
        confidence (trending never triggers the choppy block rule).
        """
        mock_classifier = _make_mock_classifier(regime="trending", probability=gb_prob)

        with _patch_gate_config(enabled=True, threshold=0.5, llm_enabled=False):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": mock_classifier}):
                with patch("ml_features.build_features", return_value={}):
                    decision = evaluate("BTCUSDT", _dummy_analysis())
                    self.assertEqual(decision.action, "allow")
                    self.assertEqual(decision.filter_reason, "")
                    self.assertGreater(decision.risk_scale, 0.0)

    @given(
        gb_prob=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_trending_always_allows_any_confidence(self, gb_prob: float):
        """
        **Validates: Requirement 1.3**

        The block condition only applies to choppy regime. Trending regime
        always passes through, even with low confidence.
        """
        mock_classifier = _make_mock_classifier(regime="trending", probability=gb_prob)

        with _patch_gate_config(enabled=True, threshold=0.5, llm_enabled=False):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": mock_classifier}):
                with patch("ml_features.build_features", return_value={}):
                    decision = evaluate("BTCUSDT", _dummy_analysis())
                    self.assertEqual(decision.action, "allow")


# ---------------------------------------------------------------------------
# Property 4: Risk-Scale-Bereichsinvariante
# Validates: Requirements 2.1, 2.4
#
# risk_scale is always in [ML_MIN_RISK_SCALE, 1.0] and monotonically
# increases with confidence.
# ---------------------------------------------------------------------------

class TestProperty4RiskScaleRange(unittest.TestCase):
    """
    **Validates: Requirements 2.1, 2.4**

    Property 4: Risk-Scale-Bereichsinvariante — risk_scale in
    [ML_MIN_RISK_SCALE, 1.0], monoton steigend mit Confidence.
    """

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0),
        min_risk_scale=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=200, deadline=None)
    def test_risk_scale_within_bounds(
        self, confidence: float, min_risk_scale: float
    ):
        """
        **Validates: Requirement 2.1**

        _compute_risk_scale always returns a value in [min_risk_scale, 1.0].
        """
        with _patch_gate_config(min_risk_scale=min_risk_scale):
            result = _compute_risk_scale(confidence)
            self.assertGreaterEqual(
                result,
                min_risk_scale,
                f"risk_scale {result} < min_risk_scale {min_risk_scale}",
            )
            self.assertLessEqual(
                result,
                1.0,
                f"risk_scale {result} > 1.0",
            )

    @given(
        c1=st.floats(min_value=0.0, max_value=1.0),
        c2=st.floats(min_value=0.0, max_value=1.0),
        min_risk_scale=st.floats(min_value=0.0, max_value=0.99),
    )
    @settings(max_examples=200, deadline=None)
    def test_risk_scale_monotonically_increasing(
        self, c1: float, c2: float, min_risk_scale: float
    ):
        """
        **Validates: Requirement 2.1**

        If c1 <= c2, then _compute_risk_scale(c1) <= _compute_risk_scale(c2).
        """
        assume(c1 <= c2)
        with _patch_gate_config(min_risk_scale=min_risk_scale):
            rs1 = _compute_risk_scale(c1)
            rs2 = _compute_risk_scale(c2)
            self.assertLessEqual(
                rs1,
                rs2 + 1e-9,  # floating point tolerance
                f"Monotonicity violated: scale({c1})={rs1} > scale({c2})={rs2}",
            )

    def test_risk_scale_at_extremes(self):
        """At confidence=0, risk_scale=min; at confidence=1, risk_scale=1.0."""
        with _patch_gate_config(min_risk_scale=0.5):
            self.assertAlmostEqual(_compute_risk_scale(0.0), 0.5, places=5)
            self.assertAlmostEqual(_compute_risk_scale(1.0), 1.0, places=5)

    def test_risk_scale_at_midpoint(self):
        """At confidence=0.5 with min=0.5, risk_scale should be 0.75."""
        with _patch_gate_config(min_risk_scale=0.5):
            result = _compute_risk_scale(0.5)
            self.assertAlmostEqual(result, 0.75, places=5)


# ---------------------------------------------------------------------------
# Property 8: Gewichtete Confidence-Score-Berechnung
# Validates: Requirements 4.3, 4.4
#
# When LLM is enabled and provides a result, the confidence is
# GB_WEIGHT * gb_prob + LLM_WEIGHT * llm_conf. When LLM fails,
# confidence falls back to gb_prob alone.
# ---------------------------------------------------------------------------

class TestProperty8WeightedConfidence(unittest.TestCase):
    """
    **Validates: Requirements 4.3, 4.4**

    Property 8: Gewichtete Confidence-Score-Berechnung —
    GB_WEIGHT * gb_prob + LLM_WEIGHT * llm_conf.
    """

    @given(
        gb_prob=st.floats(min_value=0.0, max_value=1.0),
        llm_conf=st.floats(min_value=0.0, max_value=1.0),
        gb_weight=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=200, deadline=None)
    def test_weighted_confidence_with_llm(
        self, gb_prob: float, llm_conf: float, gb_weight: float
    ):
        """
        **Validates: Requirement 4.3**

        With LLM enabled and returning a result, confidence =
        gb_weight * gb_prob + (1 - gb_weight) * llm_conf.
        """
        llm_weight = 1.0 - gb_weight

        mock_classifier = _make_mock_classifier(regime="trending", probability=gb_prob)
        mock_llm_result = {"regime": "trending", "confidence": llm_conf}

        with _patch_gate_config(
            enabled=True,
            threshold=0.0,  # ensure allow path
            gb_weight=gb_weight,
            llm_weight=llm_weight,
            llm_enabled=True,
        ):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": mock_classifier}):
                with patch("ml_features.build_features", return_value={}):
                    with patch("ml_gate._query_llm_advisor", return_value=mock_llm_result):
                        decision = evaluate("BTCUSDT", _dummy_analysis())
                        expected = gb_weight * gb_prob + llm_weight * llm_conf
                        self.assertAlmostEqual(
                            decision.confidence,
                            expected,
                            places=7,
                            msg=(
                                f"Expected confidence={expected}, got {decision.confidence} "
                                f"(gb_weight={gb_weight}, gb_prob={gb_prob}, "
                                f"llm_weight={llm_weight}, llm_conf={llm_conf})"
                            ),
                        )

    @given(
        gb_prob=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_confidence_fallback_on_llm_failure(self, gb_prob: float):
        """
        **Validates: Requirement 4.4**

        When LLM is enabled but raises an exception, confidence equals
        gb_prob alone (no LLM contribution).
        """
        mock_classifier = _make_mock_classifier(regime="trending", probability=gb_prob)

        with _patch_gate_config(
            enabled=True,
            threshold=0.0,
            llm_enabled=True,
        ):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": mock_classifier}):
                with patch("ml_features.build_features", return_value={}):
                    with patch(
                        "ml_gate._query_llm_advisor",
                        side_effect=RuntimeError("LLM unavailable"),
                    ):
                        decision = evaluate("BTCUSDT", _dummy_analysis())
                        self.assertAlmostEqual(
                            decision.confidence,
                            gb_prob,
                            places=7,
                            msg=(
                                f"On LLM failure, confidence should be {gb_prob}, "
                                f"got {decision.confidence}"
                            ),
                        )
                        self.assertIsNone(decision.llm_confidence)
                        self.assertIsNone(decision.llm_regime)

    @given(
        gb_prob=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_confidence_without_llm(self, gb_prob: float):
        """
        **Validates: Requirement 4.3**

        When LLM is disabled, confidence equals gb_prob.
        """
        mock_classifier = _make_mock_classifier(regime="trending", probability=gb_prob)

        with _patch_gate_config(
            enabled=True,
            threshold=0.0,
            llm_enabled=False,
        ):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": mock_classifier}):
                with patch("ml_features.build_features", return_value={}):
                    decision = evaluate("BTCUSDT", _dummy_analysis())
                    self.assertAlmostEqual(
                        decision.confidence,
                        gb_prob,
                        places=7,
                    )
                    self.assertIsNone(decision.llm_confidence)


# ---------------------------------------------------------------------------
# Property 9: Konfigurationsvalidierung
# Validates: Requirement 9.3
#
# validate_config() raises ValueError for invalid parameter combinations.
# ---------------------------------------------------------------------------

class TestProperty9ConfigValidation(unittest.TestCase):
    """
    **Validates: Requirement 9.3**

    Property 9: Konfigurationsvalidierung — ungueltige Werte -> ValueError.
    """

    @given(threshold=st.floats(min_value=-1e6, max_value=-0.01))
    @settings(max_examples=50, deadline=None)
    def test_negative_threshold_raises(self, threshold: float):
        """Negative ML_CONFIDENCE_THRESHOLD must raise ValueError."""
        with _patch_gate_config(threshold=threshold):
            with self.assertRaises(ValueError):
                validate_config()

    @given(threshold=st.floats(min_value=1.01, max_value=1e6))
    @settings(max_examples=50, deadline=None)
    def test_threshold_above_one_raises(self, threshold: float):
        """ML_CONFIDENCE_THRESHOLD > 1.0 must raise ValueError."""
        with _patch_gate_config(threshold=threshold):
            with self.assertRaises(ValueError):
                validate_config()

    @given(min_rs=st.floats(min_value=-1e6, max_value=-0.01))
    @settings(max_examples=50, deadline=None)
    def test_negative_min_risk_scale_raises(self, min_rs: float):
        """Negative ML_MIN_RISK_SCALE must raise ValueError."""
        with _patch_gate_config(min_risk_scale=min_rs):
            with self.assertRaises(ValueError):
                validate_config()

    @given(min_rs=st.floats(min_value=1.01, max_value=1e6))
    @settings(max_examples=50, deadline=None)
    def test_min_risk_scale_above_one_raises(self, min_rs: float):
        """ML_MIN_RISK_SCALE > 1.0 must raise ValueError."""
        with _patch_gate_config(min_risk_scale=min_rs):
            with self.assertRaises(ValueError):
                validate_config()

    @given(
        gb_weight=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_weights_not_summing_to_one_raises(self, gb_weight: float):
        """
        GB_WEIGHT + LLM_WEIGHT != 1.0 must raise ValueError.
        We intentionally make the sum wrong.
        """
        bad_llm_weight = 1.0 - gb_weight + 0.1  # sum = 1.1
        with _patch_gate_config(gb_weight=gb_weight, llm_weight=bad_llm_weight):
            with self.assertRaises(ValueError):
                validate_config()

    @given(
        gb_weight=st.floats(min_value=-1e6, max_value=-0.01),
    )
    @settings(max_examples=50, deadline=None)
    def test_negative_gb_weight_raises(self, gb_weight: float):
        """Negative ML_GB_WEIGHT must raise ValueError."""
        with _patch_gate_config(gb_weight=gb_weight, llm_weight=1.0 - gb_weight):
            with self.assertRaises(ValueError):
                validate_config()

    @given(
        threshold=st.floats(min_value=0.0, max_value=1.0),
        min_rs=st.floats(min_value=0.0, max_value=1.0),
        gb_weight=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_config_does_not_raise(
        self, threshold: float, min_rs: float, gb_weight: float
    ):
        """Valid configurations must NOT raise ValueError."""
        llm_weight = 1.0 - gb_weight
        with _patch_gate_config(
            threshold=threshold,
            min_risk_scale=min_rs,
            gb_weight=gb_weight,
            llm_weight=llm_weight,
        ):
            with patch.object(ml_gate, "ML_RETRAIN_INTERVAL_DAYS", 30):
                try:
                    validate_config()
                except ValueError:
                    self.fail(
                        f"validate_config() raised ValueError for valid config: "
                        f"threshold={threshold}, min_rs={min_rs}, "
                        f"gb_weight={gb_weight}, llm_weight={llm_weight}"
                    )

    def test_retrain_interval_zero_raises(self):
        """ML_RETRAIN_INTERVAL_DAYS < 1 must raise ValueError."""
        with _patch_gate_config():
            with patch.object(ml_gate, "ML_RETRAIN_INTERVAL_DAYS", 0):
                with self.assertRaises(ValueError):
                    validate_config()

    def test_retrain_interval_negative_raises(self):
        """Negative ML_RETRAIN_INTERVAL_DAYS must raise ValueError."""
        with _patch_gate_config():
            with patch.object(ml_gate, "ML_RETRAIN_INTERVAL_DAYS", -5):
                with self.assertRaises(ValueError):
                    validate_config()


# ---------------------------------------------------------------------------
# Property 10: Fehlerresilienz — Exception-Isolation
# Validates: Requirements 15.1, 15.2, 15.4
#
# No exception from the classifier, LLM advisor, or feature pipeline
# reaches the caller of evaluate().
# ---------------------------------------------------------------------------

class TestProperty10ExceptionIsolation(unittest.TestCase):
    """
    **Validates: Requirements 15.1, 15.2, 15.4**

    Property 10: Fehlerresilienz — Exception-Isolation — keine Exception
    erreicht den Aufrufer.
    """

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Nd")),
            min_size=3,
            max_size=10,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_feature_pipeline_exception_returns_bypass(self, symbol: str):
        """
        **Validates: Requirement 15.4**

        If build_features raises, evaluate() returns a valid GateDecision
        with action='allow' (fallback).
        """
        with _patch_gate_config(enabled=True):
            with patch(
                "ml_features.build_features",
                side_effect=RuntimeError("Feature pipeline crash"),
            ):
                decision = evaluate(symbol, _dummy_analysis())
                self.assertIsInstance(decision, GateDecision)
                self.assertEqual(decision.action, "allow")
                self.assertEqual(decision.risk_scale, 1.0)

    def test_classifier_exception_returns_fallback(self):
        """
        **Validates: Requirement 15.1**

        If the classifier's predict() raises, evaluate() returns a valid
        GateDecision without propagating the exception.
        """
        mock_classifier = MagicMock()
        mock_classifier.predict.side_effect = RuntimeError("Inference failed")
        mock_classifier.metadata = {"version": "broken"}

        with _patch_gate_config(enabled=True, threshold=0.0):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": mock_classifier}):
                with patch("ml_features.build_features", return_value={}):
                    decision = evaluate("BTCUSDT", _dummy_analysis())
                    self.assertIsInstance(decision, GateDecision)
                    # Classifier exception triggers default trending/0.5 inside
                    # _evaluate_inner, so action is still "allow"
                    self.assertEqual(decision.action, "allow")
                    self.assertIn(decision.model_version, ("fallback",))

    def test_llm_exception_uses_gb_only(self):
        """
        **Validates: Requirement 15.2**

        If the LLM advisor raises, evaluate() still succeeds using only
        the GB classifier result.
        """
        mock_classifier = _make_mock_classifier(regime="trending", probability=0.8)

        with _patch_gate_config(enabled=True, threshold=0.5, llm_enabled=True):
            with patch.dict(ml_gate._classifiers, {"BTCUSDT": mock_classifier}):
                with patch("ml_features.build_features", return_value={}):
                    with patch(
                        "ml_gate._query_llm_advisor",
                        side_effect=ConnectionError("LLM API down"),
                    ):
                        decision = evaluate("BTCUSDT", _dummy_analysis())
                        self.assertIsInstance(decision, GateDecision)
                        self.assertEqual(decision.action, "allow")
                        self.assertIsNone(decision.llm_confidence)
                        self.assertAlmostEqual(decision.confidence, 0.8)

    def test_no_model_returns_default_values(self):
        """
        **Validates: Requirements 15.1, 15.4**

        When no classifier is loaded for a symbol, evaluate() returns
        default trending/0.5 confidence without raising.
        """
        with _patch_gate_config(enabled=True, threshold=0.0):
            with patch.dict(ml_gate._classifiers, {}, clear=True):
                with patch("ml_features.build_features", return_value={}):
                    decision = evaluate("UNKNOWN", _dummy_analysis())
                    self.assertIsInstance(decision, GateDecision)
                    self.assertEqual(decision.action, "allow")
                    self.assertEqual(decision.regime, "trending")
                    self.assertAlmostEqual(decision.gb_probability, 0.5)
                    self.assertEqual(decision.model_version, "no_model")

    @given(
        exc_type=st.sampled_from([
            RuntimeError, ValueError, TypeError, KeyError,
            OSError, AttributeError, ZeroDivisionError,
        ]),
    )
    @settings(max_examples=20, deadline=None)
    def test_arbitrary_exception_never_propagates(self, exc_type: type):
        """
        **Validates: Requirement 15.4**

        evaluate() catches ALL exception types and never propagates them.
        """
        with _patch_gate_config(enabled=True):
            with patch(
                "ml_features.build_features",
                side_effect=exc_type("Arbitrary failure"),
            ):
                try:
                    decision = evaluate("BTCUSDT", _dummy_analysis())
                    self.assertIsInstance(decision, GateDecision)
                    self.assertEqual(decision.action, "allow")
                except Exception as exc:
                    self.fail(
                        f"evaluate() should never propagate exceptions, "
                        f"but raised {type(exc).__name__}: {exc}"
                    )


if __name__ == "__main__":
    unittest.main()
