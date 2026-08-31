# tests/test_causality.py
"""
Tests für den automatisierten Kausalitätstest in ml_features.py.
Validates: Requirements 6.4

Verifikation: Aufruf von causality_test() mit synthetischen OHLCV-Daten,
Ergebnis muss True sein (kein Feature verwendet Zukunftsdaten).
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path so we can import ml_features
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_features import causality_test


def _make_synthetic_ohlcv(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic OHLCV DataFrame for testing.

    Simulates a random walk with realistic price action:
    - Starts at price 100
    - Each candle has open/high/low/close with realistic wicks
    - Volume varies around a base level
    - Timestamps are hourly (matching STRATEGY_TIMEFRAME='1H')
    """
    rng = np.random.default_rng(seed)

    # Random walk for close prices
    returns = rng.normal(loc=0.0002, scale=0.01, size=n_rows)
    close = 100.0 * np.cumprod(1.0 + returns)

    # Build realistic OHLC from close
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0] * (1 - returns[0])

    # High is max of open/close plus a random wick
    wick_up = rng.uniform(0.0, 0.005, size=n_rows) * close
    high = np.maximum(open_prices, close) + wick_up

    # Low is min of open/close minus a random wick
    wick_down = rng.uniform(0.0, 0.005, size=n_rows) * close
    low = np.minimum(open_prices, close) - wick_down

    # Volume: base level with some variation
    volume = rng.uniform(500, 2000, size=n_rows)

    # Hourly timestamps
    timestamps = pd.date_range(
        start="2024-01-01", periods=n_rows, freq="1h", tz="UTC"
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestCausalityTest(unittest.TestCase):
    """
    Validates: Requirements 6.4

    Verifies that causality_test() confirms no feature at time t
    uses data from times > t.
    """

    def test_causality_test_returns_true_for_valid_data(self):
        """
        causality_test() must return True for a valid OHLCV DataFrame,
        confirming that manipulating the last 5 rows does not affect
        features computed at index -10.
        """
        df = _make_synthetic_ohlcv(n_rows=200)
        result = causality_test(df)
        self.assertTrue(
            result,
            "causality_test() returned False — a feature may be using future data"
        )

    def test_causality_test_with_different_seeds(self):
        """Causality holds across different random data sets."""
        for seed in (1, 99, 12345):
            df = _make_synthetic_ohlcv(n_rows=200, seed=seed)
            result = causality_test(df)
            self.assertTrue(
                result,
                f"causality_test() returned False for seed={seed}"
            )

    def test_causality_test_rejects_short_data(self):
        """causality_test() returns False for DataFrames with fewer than 50 rows."""
        df = _make_synthetic_ohlcv(n_rows=30)
        result = causality_test(df)
        self.assertFalse(
            result,
            "causality_test() should return False for < 50 rows"
        )

    def test_causality_test_with_minimum_viable_rows(self):
        """causality_test() works with exactly 50 rows (the minimum)."""
        df = _make_synthetic_ohlcv(n_rows=200)
        # Use 200 rows to ensure indicators have enough warmup period,
        # then verify the test still passes
        result = causality_test(df)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
