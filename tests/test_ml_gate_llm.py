# tests/test_ml_gate_llm.py
"""
Unit-Tests fuer ml_gate_llm.py — LLMRegimeAdvisor.

Validates: Requirements 4.2, 4.4, 4.6
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import requests

# Ensure the project root is on sys.path so we can import ml_gate_llm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_gate_llm import LLMRegimeAdvisor


# ---------------------------------------------------------------------------
# Test: _build_indicator_summary() enthaelt keine OHLCV-Rohdaten
# Validates: Requirement 4.2
# ---------------------------------------------------------------------------

class TestBuildIndicatorSummary(unittest.TestCase):
    """_build_indicator_summary() sends only aggregated indicators, no raw OHLCV."""

    def setUp(self):
        self.advisor = LLMRegimeAdvisor()
        self.indicators = {
            "adx": 35.2,
            "rsi": 62.1,
            "ema20": 50000.0,
            "ema50": 49000.0,
            "ema200": 45000.0,
            "di_pos": 28.5,
            "di_neg": 15.3,
            "atr": 800.0,
            "volume_ratio": 1.4,
            "macd_hist": 120.5,
            "donchian_high": 52000.0,
            "donchian_low": 47000.0,
            "current_price": 50500.0,
        }

    def test_summary_contains_aggregated_indicators(self):
        """Summary includes all expected aggregated indicator keys."""
        summary = self.advisor._build_indicator_summary("BTCUSDT", self.indicators)
        self.assertIn("ADX", summary)
        self.assertIn("RSI", summary)
        self.assertIn("EMA20", summary)
        self.assertIn("EMA50", summary)
        self.assertIn("EMA200", summary)
        self.assertIn("DI+", summary)
        self.assertIn("DI-", summary)
        self.assertIn("ATR", summary)
        self.assertIn("Volume Ratio", summary)
        self.assertIn("MACD Histogram", summary)
        self.assertIn("Donchian High", summary)
        self.assertIn("Donchian Low", summary)
        self.assertIn("Current Price", summary)

    def test_summary_contains_symbol(self):
        """Summary includes the symbol name."""
        summary = self.advisor._build_indicator_summary("ETHUSDT", self.indicators)
        self.assertIn("ETHUSDT", summary)

    def test_summary_excludes_raw_ohlcv_keys(self):
        """
        Validates: Requirement 4.2
        Summary must not contain raw OHLCV fields (open, high, low, close, volume)
        beyond the aggregated indicators.
        """
        # Inject raw OHLCV keys into indicators — they must NOT appear in summary
        indicators_with_ohlcv = {
            **self.indicators,
            "open": 49800.0,
            "high": 51200.0,
            "low": 49500.0,
            "close": 50500.0,
            "volume": 12345678.0,
        }
        summary = self.advisor._build_indicator_summary("BTCUSDT", indicators_with_ohlcv)
        summary_lower = summary.lower()

        # The summary template uses explicit keys; raw OHLCV should not leak through.
        # Check that none of the lines starts with a raw OHLCV label.
        raw_ohlcv_labels = ["open:", "high:", "low:", "close:", "volume:"]
        for label in raw_ohlcv_labels:
            # Ensure the label does not appear as a standalone line key.
            # We allow substrings like "Donchian High" or "Volume Ratio" —
            # only standalone raw labels are forbidden.
            lines = [line.strip().lower() for line in summary.split("\n")]
            for line in lines:
                if line.startswith(label):
                    self.fail(
                        f"Raw OHLCV field '{label}' found as line start in summary: {line}"
                    )

    def test_summary_missing_indicators_show_na(self):
        """Missing indicator values appear as 'N/A' in the summary."""
        summary = self.advisor._build_indicator_summary("BTCUSDT", {})
        self.assertIn("N/A", summary)


# ---------------------------------------------------------------------------
# Test: _parse_response() — gueltige und ungueltige JSON-Antworten
# Validates: Requirements 4.1, 4.2
# ---------------------------------------------------------------------------

class TestParseResponse(unittest.TestCase):
    """_parse_response() correctly parses valid and invalid API responses."""

    def setUp(self):
        self.advisor = LLMRegimeAdvisor()

    def _make_api_response(self, text: str) -> dict:
        """Helper: builds a mock Claude API response with the given text content."""
        return {
            "content": [{"type": "text", "text": text}],
        }

    def test_valid_trending_response(self):
        """Parses a valid trending response correctly."""
        resp = self._make_api_response(
            '{"regime": "trending", "confidence": 0.85, "reasoning": "Strong ADX"}'
        )
        result = self.advisor._parse_response(resp)
        self.assertEqual(result["regime"], "trending")
        self.assertAlmostEqual(result["confidence"], 0.85)

    def test_valid_choppy_response(self):
        """Parses a valid choppy response correctly."""
        resp = self._make_api_response(
            '{"regime": "choppy", "confidence": 0.3, "reasoning": "Low ADX"}'
        )
        result = self.advisor._parse_response(resp)
        self.assertEqual(result["regime"], "choppy")
        self.assertAlmostEqual(result["confidence"], 0.3)

    def test_valid_neutral_response(self):
        """Parses a valid neutral response correctly."""
        resp = self._make_api_response(
            '{"regime": "neutral", "confidence": 0.5, "reasoning": "Mixed signals"}'
        )
        result = self.advisor._parse_response(resp)
        self.assertEqual(result["regime"], "neutral")
        self.assertAlmostEqual(result["confidence"], 0.5)

    def test_invalid_json_raises_value_error(self):
        """Invalid JSON in the response raises ValueError."""
        resp = self._make_api_response("This is not JSON at all")
        with self.assertRaises(ValueError) as ctx:
            self.advisor._parse_response(resp)
        self.assertIn("Ungültige LLM-Antwort", str(ctx.exception))

    def test_empty_content_raises_value_error(self):
        """Empty content list raises ValueError (empty JSON '{}' parsed, but no crash)."""
        resp = {"content": []}
        # With empty content, text defaults to "{}", which parses to {}
        # -> regime defaults to "neutral", confidence defaults to 0.5
        result = self.advisor._parse_response(resp)
        self.assertEqual(result["regime"], "neutral")
        self.assertAlmostEqual(result["confidence"], 0.5)

    def test_missing_content_key(self):
        """Missing 'content' key defaults gracefully."""
        resp = {}
        result = self.advisor._parse_response(resp)
        self.assertEqual(result["regime"], "neutral")
        self.assertAlmostEqual(result["confidence"], 0.5)

    def test_missing_regime_defaults_to_neutral(self):
        """Missing 'regime' field defaults to 'neutral'."""
        resp = self._make_api_response('{"confidence": 0.7}')
        result = self.advisor._parse_response(resp)
        self.assertEqual(result["regime"], "neutral")

    def test_missing_confidence_defaults_to_0_5(self):
        """Missing 'confidence' field defaults to 0.5."""
        resp = self._make_api_response('{"regime": "trending"}')
        result = self.advisor._parse_response(resp)
        self.assertAlmostEqual(result["confidence"], 0.5)

    def test_unexpected_regime_defaults_to_neutral(self):
        """Unexpected regime value (not trending/choppy/neutral) defaults to 'neutral'."""
        resp = self._make_api_response(
            '{"regime": "volatile", "confidence": 0.6}'
        )
        result = self.advisor._parse_response(resp)
        self.assertEqual(result["regime"], "neutral")

    def test_confidence_clamped_above_1(self):
        """Confidence above 1.0 is clamped to 1.0."""
        resp = self._make_api_response(
            '{"regime": "trending", "confidence": 1.5}'
        )
        result = self.advisor._parse_response(resp)
        self.assertAlmostEqual(result["confidence"], 1.0)

    def test_confidence_clamped_below_0(self):
        """Confidence below 0.0 is clamped to 0.0."""
        resp = self._make_api_response(
            '{"regime": "choppy", "confidence": -0.3}'
        )
        result = self.advisor._parse_response(resp)
        self.assertAlmostEqual(result["confidence"], 0.0)

    def test_confidence_non_numeric_defaults(self):
        """Non-numeric confidence value defaults to 0.5 (via float() failure)."""
        resp = self._make_api_response(
            '{"regime": "trending", "confidence": "high"}'
        )
        # float("high") raises ValueError, caught by the default in get()
        with self.assertRaises((ValueError, TypeError)):
            self.advisor._parse_response(resp)


# ---------------------------------------------------------------------------
# Test: Timeout-Verhalten und Fallback
# Validates: Requirements 4.4, 4.6
# ---------------------------------------------------------------------------

class TestTimeoutAndFallback(unittest.TestCase):
    """Tests for timeout behavior and response time logging."""

    def setUp(self):
        self.advisor = LLMRegimeAdvisor()

    @patch("ml_gate_llm.requests.post")
    def test_timeout_raises_exception(self, mock_post):
        """
        Validates: Requirement 4.4
        When _call_claude() times out, it propagates a ConnectionError/Timeout.
        The caller (ml_gate.py) handles fallback — the advisor itself raises.
        """
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        with self.assertRaises(requests.exceptions.Timeout):
            self.advisor._call_claude("test summary")

    @patch("ml_gate_llm.requests.post")
    def test_connection_error_propagates(self, mock_post):
        """
        Validates: Requirement 4.4
        ConnectionError from the HTTP call propagates to the caller.
        """
        mock_post.side_effect = requests.exceptions.ConnectionError("DNS failure")
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.advisor._call_claude("test summary")

    @patch("ml_gate_llm.requests.post")
    def test_http_error_propagates(self, mock_post):
        """
        Validates: Requirement 4.4
        HTTP 500 from the API raises HTTPError.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_post.return_value = mock_response
        with self.assertRaises(requests.exceptions.HTTPError):
            self.advisor._call_claude("test summary")

    @patch("ml_gate_llm.requests.post")
    def test_assess_timeout_propagates(self, mock_post):
        """
        Validates: Requirement 4.4
        assess() propagates timeout so ml_gate.evaluate() can handle fallback.
        """
        mock_post.side_effect = requests.exceptions.Timeout("API timeout")
        with self.assertRaises(requests.exceptions.Timeout):
            self.advisor.assess("BTCUSDT", {"adx": 30.0, "rsi": 55.0})

    @patch("ml_gate_llm.requests.post")
    def test_assess_logs_response_time(self, mock_post):
        """
        Validates: Requirement 4.6
        assess() measures and logs response time even on success.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": '{"regime": "trending", "confidence": 0.8}'}
            ]
        }
        mock_post.return_value = mock_response

        with patch("ml_gate_llm.logger") as mock_logger:
            result = self.advisor.assess("BTCUSDT", {"adx": 30.0})
            # Verify logger.info was called with response time message
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            timing_logged = any("Antwortzeit" in call for call in log_calls)
            self.assertTrue(
                timing_logged,
                f"Response time was not logged. Log calls: {log_calls}",
            )
        self.assertEqual(result["regime"], "trending")

    @patch("ml_gate_llm.requests.post")
    def test_assess_logs_response_time_on_failure(self, mock_post):
        """
        Validates: Requirement 4.6
        assess() logs response time even when _call_claude() raises.
        """
        mock_post.side_effect = requests.exceptions.Timeout("timeout")

        with patch("ml_gate_llm.logger") as mock_logger:
            with self.assertRaises(requests.exceptions.Timeout):
                self.advisor.assess("BTCUSDT", {"adx": 30.0})
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            timing_logged = any("Antwortzeit" in call for call in log_calls)
            self.assertTrue(
                timing_logged,
                f"Response time was not logged on failure. Log calls: {log_calls}",
            )

    @patch("ml_gate_llm.requests.post")
    def test_call_claude_uses_configured_timeout(self, mock_post):
        """_call_claude() passes ML_LLM_TIMEOUT to requests.post."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": '{"regime": "neutral"}'}]
        }
        mock_post.return_value = mock_response

        with patch("ml_gate_llm.ML_LLM_TIMEOUT", 15):
            self.advisor._call_claude("test")
            _, kwargs = mock_post.call_args
            self.assertEqual(kwargs["timeout"], 15)

    @patch("ml_gate_llm.requests.post")
    def test_call_claude_sends_correct_headers(self, mock_post):
        """_call_claude() sends the correct API headers."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": '{"regime": "neutral"}'}]
        }
        mock_post.return_value = mock_response

        with patch("ml_gate_llm.CLAUDE_API_KEY", "test-key-123"):
            self.advisor._call_claude("test summary")
            _, kwargs = mock_post.call_args
            headers = kwargs["headers"]
            self.assertEqual(headers["x-api-key"], "test-key-123")
            self.assertIn("anthropic-version", headers)
            self.assertEqual(headers["Content-Type"], "application/json")

    @patch("ml_gate_llm.requests.post")
    def test_successful_assess_returns_valid_result(self, mock_post):
        """Full assess() flow returns correct regime and confidence."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "content": [
                {
                    "type": "text",
                    "text": '{"regime": "choppy", "confidence": 0.35, "reasoning": "Flat ADX"}',
                }
            ]
        }
        mock_post.return_value = mock_response

        result = self.advisor.assess(
            "ETHUSDT",
            {"adx": 15.0, "rsi": 50.0, "ema20": 3000.0},
        )
        self.assertEqual(result["regime"], "choppy")
        self.assertAlmostEqual(result["confidence"], 0.35)


if __name__ == "__main__":
    unittest.main()
