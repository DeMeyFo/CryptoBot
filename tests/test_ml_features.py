# tests/test_ml_features.py
"""
Property-Tests für ml_features.py — Feature-Kausalität und Korrektheit.

**Validates: Requirements 6.1, 6.2, 6.4**
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path so we can import ml_features
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_features import (
    FEATURE_COLUMNS,
    _add_derived_features,
    _empty_features,
    _extract_feature_row,
    build_feature_matrix,
    causality_test,
)
from technical_analysis import prepare_indicators


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_ohlcv(n_rows: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic OHLCV DataFrame that mimics realistic price data.

    Uses a random walk with drift so that EMA, ADX, RSI and other indicators
    compute without degenerate NaN-only columns.
    """
    rng = np.random.default_rng(seed)
    # Random walk starting at 100
    returns = rng.normal(loc=0.0002, scale=0.01, size=n_rows)
    close = 100.0 * np.cumprod(1.0 + returns)

    # Realistic high/low around close
    high = close * (1.0 + rng.uniform(0.001, 0.02, size=n_rows))
    low = close * (1.0 - rng.uniform(0.001, 0.02, size=n_rows))
    open_ = close * (1.0 + rng.normal(0, 0.005, size=n_rows))

    volume = rng.uniform(1_000, 100_000, size=n_rows)

    timestamps = pd.date_range(
        start="2024-01-01", periods=n_rows, freq="1h", tz="UTC"
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# Property 6: Feature-Kausalität
# Validates: Requirements 6.1, 6.2, 6.4
#
# For any DataFrame with OHLCV data, manipulation of data at timepoints > t
# must not change features at timepoint t.
# ---------------------------------------------------------------------------

class TestFeatureCausality(unittest.TestCase):
    """
    **Validates: Requirements 6.1, 6.2, 6.4**

    Property 6: Feature-Kausalität — Manipulation von Daten an Zeitpunkten > t
    darf Features bei Zeitpunkt t nicht verändern.
    """

    def setUp(self):
        """Create a synthetic OHLCV DataFrame with 250 rows."""
        self.df = _make_synthetic_ohlcv(n_rows=250)

    def test_future_data_manipulation_does_not_affect_past_features(self):
        """
        Build features at index -10, modify last 5 rows, rebuild.
        Features at index -10 must be identical.
        """
        # Original features
        frame_original = prepare_indicators(self.df.copy())
        frame_original = _add_derived_features(frame_original)
        original_features = _extract_feature_row(frame_original, index=-10)

        # Modify the last 5 rows (future data relative to index -10)
        df_modified = self.df.copy()
        df_modified.iloc[-5:, df_modified.columns.get_loc("close")] *= 2.0
        df_modified.iloc[-5:, df_modified.columns.get_loc("high")] *= 2.0
        df_modified.iloc[-5:, df_modified.columns.get_loc("volume")] *= 10.0

        frame_modified = prepare_indicators(df_modified)
        frame_modified = _add_derived_features(frame_modified)
        modified_features = _extract_feature_row(frame_modified, index=-10)

        # All features at index -10 must remain unchanged
        for col in FEATURE_COLUMNS:
            self.assertAlmostEqual(
                original_features[col],
                modified_features[col],
                places=10,
                msg=(
                    f"Feature '{col}' changed after future data manipulation: "
                    f"{original_features[col]} != {modified_features[col]}"
                ),
            )

    def test_causality_test_function_returns_true(self):
        """The built-in causality_test() must return True for valid causal features."""
        result = causality_test(self.df)
        self.assertTrue(
            result,
            "causality_test() returned False — features may contain look-ahead bias",
        )

    def test_causality_with_different_seed(self):
        """Causality holds across different synthetic datasets."""
        df2 = _make_synthetic_ohlcv(n_rows=200, seed=99)

        frame_orig = prepare_indicators(df2.copy())
        frame_orig = _add_derived_features(frame_orig)
        orig_features = _extract_feature_row(frame_orig, index=-10)

        df2_mod = df2.copy()
        df2_mod.iloc[-5:, df2_mod.columns.get_loc("close")] *= 3.0
        df2_mod.iloc[-5:, df2_mod.columns.get_loc("high")] *= 3.0
        df2_mod.iloc[-5:, df2_mod.columns.get_loc("low")] *= 0.5
        df2_mod.iloc[-5:, df2_mod.columns.get_loc("volume")] *= 5.0

        frame_mod = prepare_indicators(df2_mod)
        frame_mod = _add_derived_features(frame_mod)
        mod_features = _extract_feature_row(frame_mod, index=-10)

        for col in FEATURE_COLUMNS:
            self.assertAlmostEqual(
                orig_features[col],
                mod_features[col],
                places=10,
                msg=f"Feature '{col}' leaked future data (seed=99)",
            )


# ---------------------------------------------------------------------------
# build_feature_matrix: correct column count
# Validates: Requirements 6.1, 6.2
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix(unittest.TestCase):
    """
    **Validates: Requirements 6.1, 6.2**

    build_feature_matrix returns a DataFrame with exactly len(FEATURE_COLUMNS)
    columns (15 = 6 base + 9 derived).
    """

    def test_column_count_equals_feature_columns(self):
        """build_feature_matrix must return exactly 15 feature columns."""
        df = _make_synthetic_ohlcv(n_rows=250)
        result = build_feature_matrix(df)
        self.assertEqual(
            len(result.columns),
            len(FEATURE_COLUMNS),
            f"Expected {len(FEATURE_COLUMNS)} columns, got {len(result.columns)}: "
            f"{list(result.columns)}",
        )
        self.assertEqual(len(FEATURE_COLUMNS), 15)

    def test_columns_match_feature_columns(self):
        """build_feature_matrix columns must exactly match FEATURE_COLUMNS."""
        df = _make_synthetic_ohlcv(n_rows=250)
        result = build_feature_matrix(df)
        self.assertListEqual(list(result.columns), FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# _empty_features: correct structure
# Validates: Requirements 6.1, 6.2
# ---------------------------------------------------------------------------

class TestEmptyFeatures(unittest.TestCase):
    """
    **Validates: Requirements 6.1, 6.2**

    _empty_features returns a dict with all FEATURE_COLUMNS as keys and 0.0
    as values.
    """

    def test_empty_features_keys(self):
        """_empty_features dict has exactly the FEATURE_COLUMNS keys."""
        result = _empty_features()
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(FEATURE_COLUMNS))

    def test_empty_features_all_zero(self):
        """Every value in _empty_features must be 0.0."""
        result = _empty_features()
        for col in FEATURE_COLUMNS:
            self.assertEqual(
                result[col],
                0.0,
                f"_empty_features()['{col}'] should be 0.0, got {result[col]}",
            )

    def test_empty_features_count(self):
        """_empty_features must have exactly len(FEATURE_COLUMNS) entries."""
        result = _empty_features()
        self.assertEqual(len(result), len(FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
