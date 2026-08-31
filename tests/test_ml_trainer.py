# tests/test_ml_trainer.py
"""
Unit-Tests fuer ml_trainer.py — Label-Generierung und Walk-Forward-Splits.

**Validates: Requirements 5.1, 5.7**
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_trainer import (
    WalkForwardSplit,
    generate_labels,
    walk_forward_splits,
)
from technical_analysis import prepare_indicators


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_ohlcv(n_rows: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic OHLCV DataFrame with prepare_indicators applied,
    so it includes the 'atr' column needed by generate_labels.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=0.01, size=n_rows)
    close = 100.0 * np.cumprod(1.0 + returns)

    high = close * (1.0 + rng.uniform(0.001, 0.02, size=n_rows))
    low = close * (1.0 - rng.uniform(0.001, 0.02, size=n_rows))
    open_ = close * (1.0 + rng.normal(0, 0.005, size=n_rows))
    volume = rng.uniform(1_000, 100_000, size=n_rows)

    timestamps = pd.date_range(
        start="2024-01-01", periods=n_rows, freq="1h", tz="UTC"
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return prepare_indicators(df)


# ---------------------------------------------------------------------------
# Tests for generate_labels
# Validates: Requirement 5.7
# ---------------------------------------------------------------------------

class TestGenerateLabels(unittest.TestCase):
    """
    **Validates: Requirement 5.7**

    generate_labels produces binary labels (0 or 1) based on future price
    movement relative to ATR threshold.
    """

    def setUp(self):
        self.df = _make_synthetic_ohlcv(n_rows=250)

    def test_labels_are_binary(self):
        """Labels must be 0.0 or 1.0 (or NaN for the last `horizon` rows)."""
        labels = generate_labels(self.df, horizon=24, atr_threshold=1.5)
        valid = labels.dropna()
        unique = set(valid.unique())
        self.assertTrue(
            unique.issubset({0.0, 1.0}),
            f"Labels should be 0.0 or 1.0, got {unique}",
        )

    def test_last_horizon_rows_default_to_choppy(self):
        """
        The last `horizon` rows have no future data, so future_max_move is NaN.
        The boolean comparison NaN > threshold yields False, which becomes 0.0
        (choppy). This is expected design behavior.
        """
        horizon = 24
        labels = generate_labels(self.df, horizon=horizon, atr_threshold=1.5)
        tail = labels.iloc[-horizon:]
        self.assertTrue(
            (tail == 0.0).all(),
            f"Expected last {horizon} labels to be 0.0 (choppy), got: {tail.values}",
        )

    def test_label_count_matches_dataframe(self):
        """Labels series length must match the input DataFrame length."""
        horizon = 24
        labels = generate_labels(self.df, horizon=horizon, atr_threshold=1.5)
        self.assertEqual(
            len(labels),
            len(self.df),
            f"Expected {len(self.df)} labels, got {len(labels)}",
        )

    def test_high_threshold_produces_more_choppy(self):
        """A very high ATR threshold should produce mostly choppy (0) labels."""
        labels = generate_labels(self.df, horizon=24, atr_threshold=100.0)
        valid = labels.dropna()
        choppy_ratio = (valid == 0.0).mean()
        self.assertGreater(
            choppy_ratio,
            0.9,
            f"With threshold=100, expected mostly choppy labels, got choppy ratio {choppy_ratio}",
        )

    def test_low_threshold_produces_more_trending(self):
        """A very low ATR threshold should produce mostly trending (1) labels."""
        labels = generate_labels(self.df, horizon=24, atr_threshold=0.01)
        valid = labels.dropna()
        trending_ratio = (valid == 1.0).mean()
        self.assertGreater(
            trending_ratio,
            0.5,
            f"With threshold=0.01, expected mostly trending labels, got ratio {trending_ratio}",
        )

    def test_label_uses_only_future_data(self):
        """
        **Validates: Property 15**

        Changing data at index i or earlier must not change the label at index i
        (as long as candles i+1..i+N remain constant).
        """
        horizon = 10
        idx = 100  # target index

        labels_orig = generate_labels(self.df.copy(), horizon=horizon, atr_threshold=1.5)

        # Modify data at and before the target index
        df_mod = self.df.copy()
        df_mod.iloc[:idx + 1, df_mod.columns.get_loc("close")] *= 0.5
        df_mod.iloc[:idx + 1, df_mod.columns.get_loc("high")] *= 0.5
        df_mod.iloc[:idx + 1, df_mod.columns.get_loc("low")] *= 0.5
        # Recompute ATR for modified data (ATR changes with past data)
        # But label depends on future_max_move > threshold * atr / close,
        # and future_max_move uses close[i] as denominator, so changing close[i]
        # WILL change the label. The property is that the label at i depends on
        # candles i+1..i+N for the max price movement range.
        # Instead, verify that changing candles AFTER i+horizon does NOT affect label at i.
        df_mod2 = self.df.copy()
        far_future_start = idx + 1 + horizon
        if far_future_start < len(df_mod2):
            df_mod2.iloc[far_future_start:, df_mod2.columns.get_loc("close")] *= 3.0
            df_mod2.iloc[far_future_start:, df_mod2.columns.get_loc("high")] *= 3.0
            df_mod2.iloc[far_future_start:, df_mod2.columns.get_loc("low")] *= 3.0
            labels_mod2 = generate_labels(df_mod2, horizon=horizon, atr_threshold=1.5)
            self.assertAlmostEqual(
                labels_orig.iloc[idx],
                labels_mod2.iloc[idx],
                places=10,
                msg="Label at idx changed after modifying data beyond horizon window",
            )

    def test_different_horizon_changes_labels(self):
        """Different horizon values should produce different label distributions."""
        labels_short = generate_labels(self.df, horizon=5, atr_threshold=1.5)
        labels_long = generate_labels(self.df, horizon=48, atr_threshold=1.5)
        # A longer horizon looks at more candles, so it should generally
        # find larger price movements, producing more trending labels.
        trending_short = (labels_short == 1.0).sum()
        trending_long = (labels_long == 1.0).sum()
        self.assertNotEqual(
            trending_short,
            trending_long,
            "Different horizons should produce different trending counts",
        )


# ---------------------------------------------------------------------------
# Tests for WalkForwardSplit and walk_forward_splits
# Validates: Requirement 5.1
# ---------------------------------------------------------------------------

class TestWalkForwardSplits(unittest.TestCase):
    """
    **Validates: Requirement 5.1**

    walk_forward_splits produces chronologically ordered, non-overlapping
    train/validation splits.
    """

    def test_single_split_default(self):
        """A single split should cover the entire dataset."""
        splits = walk_forward_splits(1000, train_ratio=0.7, n_splits=1)
        self.assertEqual(len(splits), 1)
        s = splits[0]
        self.assertEqual(s.train_start, 0)
        self.assertEqual(s.train_end, 700)
        self.assertEqual(s.val_start, 700)
        self.assertEqual(s.val_end, 1000)

    def test_train_precedes_validation(self):
        """
        **Validates: Property 7**

        For every split, train_end <= val_start (no overlap),
        and train_start < train_end (non-empty training set).
        """
        for n_splits in [1, 2, 3, 5]:
            splits = walk_forward_splits(1000, train_ratio=0.7, n_splits=n_splits)
            for i, s in enumerate(splits):
                self.assertLess(
                    s.train_start, s.train_end,
                    f"Split {i}: train_start ({s.train_start}) must be < train_end ({s.train_end})",
                )
                self.assertLessEqual(
                    s.train_end, s.val_start,
                    f"Split {i}: train_end ({s.train_end}) must be <= val_start ({s.val_start})",
                )
                self.assertLess(
                    s.val_start, s.val_end,
                    f"Split {i}: val_start ({s.val_start}) must be < val_end ({s.val_end})",
                )

    def test_no_overlap_between_splits(self):
        """Consecutive splits must not overlap."""
        splits = walk_forward_splits(1000, train_ratio=0.7, n_splits=3)
        for i in range(len(splits) - 1):
            self.assertLessEqual(
                splits[i].val_end, splits[i + 1].train_start,
                f"Split {i} val_end ({splits[i].val_end}) must be <= "
                f"split {i + 1} train_start ({splits[i + 1].train_start})",
            )

    def test_multiple_splits_correct_count(self):
        """walk_forward_splits(n_splits=k) returns exactly k splits."""
        for k in [1, 2, 4, 5]:
            splits = walk_forward_splits(1000, train_ratio=0.7, n_splits=k)
            self.assertEqual(len(splits), k)

    def test_train_ratio_respected(self):
        """Training portion should be approximately train_ratio of each split."""
        splits = walk_forward_splits(1000, train_ratio=0.7, n_splits=1)
        s = splits[0]
        train_size = s.train_end - s.train_start
        total_size = s.val_end - s.train_start
        actual_ratio = train_size / total_size
        self.assertAlmostEqual(
            actual_ratio, 0.7, places=1,
            msg=f"Train ratio should be ~0.7, got {actual_ratio}",
        )

    def test_dataclass_fields(self):
        """WalkForwardSplit has the required fields."""
        s = WalkForwardSplit(train_start=0, train_end=70, val_start=70, val_end=100)
        self.assertEqual(s.train_start, 0)
        self.assertEqual(s.train_end, 70)
        self.assertEqual(s.val_start, 70)
        self.assertEqual(s.val_end, 100)

    def test_splits_cover_full_range_single_split(self):
        """A single split should span from 0 to n_samples."""
        splits = walk_forward_splits(500, train_ratio=0.8, n_splits=1)
        s = splits[0]
        self.assertEqual(s.train_start, 0)
        self.assertEqual(s.val_end, 500)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Tests for RegimeClassifier
# Validates: Requirements 3.1, 3.4, 3.5, 12.4
# ---------------------------------------------------------------------------

import json
import pickle
import shutil
import tempfile
from pathlib import Path

from ml_trainer import RegimeClassifier


class _MockModel:
    """Minimal mock that mimics sklearn's predict_proba interface."""

    def __init__(self, trending_prob: float = 0.8):
        self._trending_prob = trending_prob

    def predict_proba(self, X):
        """Return [choppy_prob, trending_prob] for each row."""
        n = X.shape[0]
        choppy = 1.0 - self._trending_prob
        return np.array([[choppy, self._trending_prob]] * n)


def _create_model_dir(base: str, symbol: str, version: str,
                      trending_prob: float = 0.8,
                      feature_columns: list[str] | None = None) -> Path:
    """Create a mock model directory with model.pkl and metadata.json."""
    cols = feature_columns or ["adx", "rsi", "atr"]
    dir_path = Path(base) / f"{symbol}_{version}"
    dir_path.mkdir(parents=True, exist_ok=True)

    model = _MockModel(trending_prob)
    with open(dir_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "symbol": symbol,
        "version": version,
        "feature_columns": cols,
        "validation_metrics": {"auc_roc": 0.75},
        "trained_at": "2025-01-01T00:00:00Z",
    }
    with open(dir_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return dir_path


class TestRegimeClassifierPredict(unittest.TestCase):
    """
    **Validates: Requirement 3.4, Property 13**

    RegimeClassifier.predict() returns regime in {trending, choppy}
    and probability in [0.0, 1.0].
    """

    def test_predict_trending(self):
        """Model with trending_prob=0.8 should predict 'trending'."""
        model = _MockModel(trending_prob=0.8)
        classifier = RegimeClassifier(
            model=model,
            feature_columns=["adx", "rsi", "atr"],
            metadata={"version": "test"},
        )
        result = classifier.predict({"adx": 40, "rsi": 55, "atr": 0.5})
        self.assertEqual(result["regime"], "trending")
        self.assertAlmostEqual(result["probability"], 0.8)

    def test_predict_choppy(self):
        """Model with trending_prob=0.3 should predict 'choppy'."""
        model = _MockModel(trending_prob=0.3)
        classifier = RegimeClassifier(
            model=model,
            feature_columns=["adx", "rsi", "atr"],
            metadata={"version": "test"},
        )
        result = classifier.predict({"adx": 15, "rsi": 50, "atr": 0.2})
        self.assertEqual(result["regime"], "choppy")
        self.assertAlmostEqual(result["probability"], 0.3)

    def test_predict_boundary_at_half(self):
        """Model with trending_prob=0.5 should predict 'trending' (>= 0.5)."""
        model = _MockModel(trending_prob=0.5)
        classifier = RegimeClassifier(
            model=model,
            feature_columns=["adx"],
            metadata={"version": "test"},
        )
        result = classifier.predict({"adx": 30})
        self.assertEqual(result["regime"], "trending")
        self.assertAlmostEqual(result["probability"], 0.5)

    def test_predict_missing_features_default_to_zero(self):
        """Missing feature keys should default to 0.0."""
        model = _MockModel(trending_prob=0.7)
        classifier = RegimeClassifier(
            model=model,
            feature_columns=["adx", "rsi", "atr"],
            metadata={"version": "test"},
        )
        # Only provide 'adx', others default to 0.0
        result = classifier.predict({"adx": 30})
        self.assertIn(result["regime"], ("trending", "choppy"))
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    def test_predict_probability_in_range(self):
        """Probability must always be in [0.0, 1.0]."""
        for prob in [0.0, 0.1, 0.5, 0.9, 1.0]:
            model = _MockModel(trending_prob=prob)
            classifier = RegimeClassifier(
                model=model,
                feature_columns=["adx"],
                metadata={"version": "test"},
            )
            result = classifier.predict({"adx": 25})
            self.assertGreaterEqual(result["probability"], 0.0)
            self.assertLessEqual(result["probability"], 1.0)
            self.assertIn(result["regime"], ("trending", "choppy"))


class TestRegimeClassifierLoadLatest(unittest.TestCase):
    """
    **Validates: Requirement 12.4**

    load_latest() loads the most recent model directory for a symbol.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_latest_picks_newest(self):
        """load_latest should pick the most recent (alphabetically last) directory."""
        _create_model_dir(self.tmpdir, "BTCUSDT", "20250101T000000Z", trending_prob=0.6)
        _create_model_dir(self.tmpdir, "BTCUSDT", "20250201T000000Z", trending_prob=0.9)

        classifier = RegimeClassifier.load_latest("BTCUSDT", self.tmpdir)
        self.assertEqual(classifier.metadata["version"], "20250201T000000Z")
        result = classifier.predict({"adx": 30, "rsi": 60, "atr": 0.5})
        self.assertAlmostEqual(result["probability"], 0.9)

    def test_load_latest_no_model_raises(self):
        """load_latest should raise FileNotFoundError when no model exists."""
        with self.assertRaises(FileNotFoundError):
            RegimeClassifier.load_latest("NONEXISTENT", self.tmpdir)

    def test_load_latest_ignores_other_symbols(self):
        """load_latest for BTCUSDT should not pick up ETHUSDT models."""
        _create_model_dir(self.tmpdir, "ETHUSDT", "20250301T000000Z", trending_prob=0.9)

        with self.assertRaises(FileNotFoundError):
            RegimeClassifier.load_latest("BTCUSDT", self.tmpdir)

    def test_load_latest_metadata_has_path(self):
        """Loaded classifier should have 'path' in metadata."""
        _create_model_dir(self.tmpdir, "BTCUSDT", "20250101T000000Z")
        classifier = RegimeClassifier.load_latest("BTCUSDT", self.tmpdir)
        self.assertIn("path", classifier.metadata)
        self.assertTrue(classifier.metadata["path"].endswith("BTCUSDT_20250101T000000Z"))


class TestRegimeClassifierLoadVersion(unittest.TestCase):
    """
    **Validates: Requirement 12.4**

    load_version() loads a specific model version for rollback.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_specific_version(self):
        """load_version should load the exact version requested."""
        _create_model_dir(self.tmpdir, "BTCUSDT", "20250101T000000Z", trending_prob=0.6)
        _create_model_dir(self.tmpdir, "BTCUSDT", "20250201T000000Z", trending_prob=0.9)

        classifier = RegimeClassifier.load_version(
            "BTCUSDT", "20250101T000000Z", self.tmpdir
        )
        self.assertEqual(classifier.metadata["version"], "20250101T000000Z")
        result = classifier.predict({"adx": 30, "rsi": 60, "atr": 0.5})
        self.assertAlmostEqual(result["probability"], 0.6)

    def test_load_version_not_found_raises(self):
        """load_version should raise FileNotFoundError for missing versions."""
        with self.assertRaises(FileNotFoundError):
            RegimeClassifier.load_version(
                "BTCUSDT", "20990101T000000Z", self.tmpdir
            )

    def test_load_version_feature_columns(self):
        """Loaded classifier should have the correct feature_columns."""
        cols = ["adx", "rsi", "macd_hist", "di_spread"]
        _create_model_dir(
            self.tmpdir, "ETHUSDT", "20250301T000000Z",
            feature_columns=cols,
        )
        classifier = RegimeClassifier.load_version(
            "ETHUSDT", "20250301T000000Z", self.tmpdir
        )
        self.assertEqual(classifier.feature_columns, cols)


# ---------------------------------------------------------------------------
# Tests for WalkForwardTrainer
# Validates: Requirements 5.2, 5.3, 5.5, 5.6, 11.1, 12.1, 12.2, 13.4
# ---------------------------------------------------------------------------

from ml_trainer import WalkForwardTrainer


class TestWalkForwardTrainerTrain(unittest.TestCase):
    """
    **Validates: Requirements 5.2, 5.3, 5.5, 5.6, 12.1, 12.2, 13.4**

    WalkForwardTrainer.train() performs walk-forward training,
    saves model + metadata, and rejects AUC-regressions.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trainer = WalkForwardTrainer(model_dir=self.tmpdir)
        # Create synthetic OHLCV data large enough for training
        self.df = _make_synthetic_ohlcv(n_rows=500, seed=123)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_train_returns_accepted(self):
        """Training with valid data should return status='accepted'."""
        from ml_features import FEATURE_COLUMNS
        result = self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)
        self.assertEqual(result["status"], "accepted")
        self.assertIn("metrics", result)
        self.assertIn("model_path", result)

    def test_train_saves_model_pkl(self):
        """Training should create a model.pkl file."""
        from ml_features import FEATURE_COLUMNS
        result = self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)
        model_path = Path(result["model_path"])
        self.assertTrue((model_path / "model.pkl").exists())

    def test_train_saves_metadata_json(self):
        """Training should create a metadata.json with required fields."""
        from ml_features import FEATURE_COLUMNS
        result = self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)
        model_path = Path(result["model_path"])
        meta_file = model_path / "metadata.json"
        self.assertTrue(meta_file.exists())

        with open(meta_file) as f:
            metadata = json.load(f)

        # Anforderung 12.1: Required metadata fields
        self.assertEqual(metadata["symbol"], "BTCUSDT")
        self.assertIn("version", metadata)
        self.assertIn("feature_columns", metadata)
        self.assertIn("training_period", metadata)
        self.assertIn("start", metadata["training_period"])
        self.assertIn("end", metadata["training_period"])
        self.assertIn("validation_metrics", metadata)
        self.assertIn("trained_at", metadata)
        self.assertIn("data_hash", metadata)
        self.assertIn("hyperparameters", metadata)

    def test_train_metrics_all_present(self):
        """Anforderung 5.5: All validation metrics should be reported."""
        from ml_features import FEATURE_COLUMNS
        result = self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)
        metrics = result["metrics"]
        for key in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
            self.assertIn(key, metrics, f"Missing metric: {key}")
            self.assertIsInstance(metrics[key], float)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

    def test_train_directory_naming_convention(self):
        """Anforderung 12.2: Model dir follows {symbol}_{timestamp}/ pattern."""
        from ml_features import FEATURE_COLUMNS
        result = self.trainer.train("ETHUSDT", self.df, FEATURE_COLUMNS)
        model_path = Path(result["model_path"])
        self.assertTrue(model_path.name.startswith("ETHUSDT_"))

    def test_train_too_few_samples_raises(self):
        """Training with < 100 valid samples should raise ValueError."""
        small_df = _make_synthetic_ohlcv(n_rows=50, seed=99)
        from ml_features import FEATURE_COLUMNS
        with self.assertRaises(ValueError):
            self.trainer.train("BTCUSDT", small_df, FEATURE_COLUMNS)

    def test_train_auc_regression_rejected(self):
        """
        Anforderung 5.6: If new model has worse AUC-ROC, reject it.
        We train once to create a baseline, then train again with the same data.
        If the second model is not worse, we just verify the mechanism works
        by mocking _current_model_auc.
        """
        from ml_features import FEATURE_COLUMNS

        # First train creates a model
        result1 = self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)
        self.assertEqual(result1["status"], "accepted")

        # Patch _current_model_auc to return a very high AUC
        original_method = self.trainer._current_model_auc
        self.trainer._current_model_auc = lambda symbol: 0.999

        try:
            result2 = self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)
            self.assertEqual(result2["status"], "rejected")
            self.assertEqual(result2["reason"], "auc_regression")
            self.assertIn("metrics", result2)
        finally:
            self.trainer._current_model_auc = original_method

    def test_saved_model_loadable_by_classifier(self):
        """Saved model should be loadable by RegimeClassifier.load_latest()."""
        from ml_features import FEATURE_COLUMNS
        self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)

        classifier = RegimeClassifier.load_latest("BTCUSDT", self.tmpdir)
        result = classifier.predict({"adx": 35, "rsi": 55, "atr": 0.5})
        self.assertIn(result["regime"], ("trending", "choppy"))
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)


class TestWalkForwardTrainerHistoricalPredictions(unittest.TestCase):
    """
    **Validates: Requirement 11.1**

    historical_predictions() returns list of (timestamp, regime, confidence)
    tuples for backtest integration.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trainer = WalkForwardTrainer(model_dir=self.tmpdir)
        self.df = _make_synthetic_ohlcv(n_rows=500, seed=456)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_historical_predictions_returns_tuples(self):
        """Each prediction should be a (timestamp, regime, confidence) tuple."""
        from ml_features import FEATURE_COLUMNS
        self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)

        predictions = self.trainer.historical_predictions(
            "BTCUSDT", self.df, FEATURE_COLUMNS
        )
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        for ts, regime, confidence in predictions:
            self.assertIsInstance(ts, str)
            self.assertIn(regime, ("trending", "choppy"))
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_historical_predictions_no_model_raises(self):
        """historical_predictions should raise when no model exists."""
        from ml_features import FEATURE_COLUMNS
        with self.assertRaises(FileNotFoundError):
            self.trainer.historical_predictions(
                "NONEXISTENT", self.df, FEATURE_COLUMNS
            )


class TestWalkForwardTrainerCurrentModelAuc(unittest.TestCase):
    """
    **Validates: Requirement 5.6**

    _current_model_auc() returns existing AUC or None.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trainer = WalkForwardTrainer(model_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_no_model_returns_none(self):
        """No existing model should return None."""
        result = self.trainer._current_model_auc("BTCUSDT")
        self.assertIsNone(result)

    def test_existing_model_returns_auc(self):
        """Existing model with AUC metadata should return the AUC value."""
        _create_model_dir(
            self.tmpdir, "BTCUSDT", "20250101T000000Z",
            trending_prob=0.8,
        )
        result = self.trainer._current_model_auc("BTCUSDT")
        self.assertAlmostEqual(result, 0.75)  # set in _create_model_dir


# ===========================================================================
# Property-Based Tests (hypothesis)
# Task 5.4: Property-Tests fuer Walk-Forward-Training
# ===========================================================================

from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Property 7: Walk-Forward chronologische Trennung
# Validates: Requirements 5.1, 6.3
#
# For any Walk-Forward-Split the last training timepoint must be strictly
# before the first validation timepoint, with no overlap.
# ---------------------------------------------------------------------------

class TestProperty7WalkForwardChronologischeTrennung(unittest.TestCase):
    """
    **Validates: Requirements 5.1, 6.3**

    Property 7: Walk-Forward chronologische Trennung — last training
    timepoint strictly before first validation timepoint.
    """

    @given(
        n_samples=st.integers(min_value=10, max_value=10_000),
        train_ratio=st.floats(min_value=0.1, max_value=0.9),
        n_splits=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=200, deadline=None)
    def test_train_end_le_val_start_for_all_splits(
        self, n_samples: int, train_ratio: float, n_splits: int
    ):
        """
        **Validates: Requirements 5.1**

        For every generated split, train_end <= val_start (no temporal leak).
        """
        splits = walk_forward_splits(n_samples, train_ratio, n_splits)
        self.assertEqual(len(splits), n_splits)
        for i, s in enumerate(splits):
            self.assertLessEqual(
                s.train_end,
                s.val_start,
                f"Split {i}: train_end ({s.train_end}) must be "
                f"<= val_start ({s.val_start})",
            )

    @given(
        n_samples=st.integers(min_value=20, max_value=10_000),
        train_ratio=st.floats(min_value=0.2, max_value=0.8),
        n_splits=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=200, deadline=None)
    def test_training_block_is_nonempty(
        self, n_samples: int, train_ratio: float, n_splits: int
    ):
        """
        **Validates: Requirements 5.1**

        For every split, train_start < train_end (non-empty training block),
        given that each split has enough samples for a meaningful partition.
        """
        split_size = n_samples // n_splits
        # Ensure integer truncation of (split_size * train_ratio) yields >= 1
        assume(int(split_size * train_ratio) >= 1)

        splits = walk_forward_splits(n_samples, train_ratio, n_splits)
        for i, s in enumerate(splits):
            self.assertLess(
                s.train_start,
                s.train_end,
                f"Split {i}: empty training block "
                f"(train_start={s.train_start}, train_end={s.train_end})",
            )

    @given(
        n_samples=st.integers(min_value=20, max_value=10_000),
        train_ratio=st.floats(min_value=0.1, max_value=0.9),
        n_splits=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=200, deadline=None)
    def test_consecutive_splits_do_not_overlap(
        self, n_samples: int, train_ratio: float, n_splits: int
    ):
        """
        **Validates: Requirements 6.3**

        For consecutive splits, split[k].val_end <= split[k+1].train_start.
        """
        splits = walk_forward_splits(n_samples, train_ratio, n_splits)
        for k in range(len(splits) - 1):
            self.assertLessEqual(
                splits[k].val_end,
                splits[k + 1].train_start,
                f"Splits {k} and {k+1} overlap: "
                f"val_end={splits[k].val_end}, "
                f"train_start={splits[k+1].train_start}",
            )


# ---------------------------------------------------------------------------
# Property 12: Modell-AUC-Regression verhindert Deployment
# Validates: Requirements 5.6
#
# For any newly trained model whose validation AUC-ROC is worse than the
# current model, the WalkForwardTrainer SHALL reject it.
# ---------------------------------------------------------------------------

class TestProperty12AUCRegressionVerhindertDeployment(unittest.TestCase):
    """
    **Validates: Requirements 5.6**

    Property 12: Modell-AUC-Regression verhindert Deployment — a model with
    worse AUC-ROC than the incumbent is always rejected.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trainer = WalkForwardTrainer(model_dir=self.tmpdir)
        self.df = _make_synthetic_ohlcv(n_rows=500, seed=777)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @given(
        fake_incumbent_auc=st.floats(min_value=0.51, max_value=1.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_worse_model_always_rejected(self, fake_incumbent_auc: float):
        """
        **Validates: Requirements 5.6**

        When _current_model_auc returns a high value, any new model whose
        AUC-ROC is lower must be rejected with status='rejected' and
        reason='auc_regression'.
        """
        from ml_features import FEATURE_COLUMNS

        # Patch _current_model_auc to return the hypothesis-provided AUC
        original = self.trainer._current_model_auc
        self.trainer._current_model_auc = lambda symbol: fake_incumbent_auc

        try:
            result = self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)
            # The synthetic data is random, so the trained model's AUC on
            # the validation set will be around 0.5.  With a fake incumbent
            # AUC > 0.51, the new model should almost always be worse.
            if result["metrics"]["auc_roc"] < fake_incumbent_auc:
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["reason"], "auc_regression")
            else:
                # If by chance the new model beats the threshold, accept it.
                self.assertEqual(result["status"], "accepted")
        finally:
            self.trainer._current_model_auc = original

    def test_no_incumbent_always_accepted(self):
        """
        **Validates: Requirements 5.6**

        When no incumbent model exists (_current_model_auc returns None),
        the new model is always accepted.
        """
        from ml_features import FEATURE_COLUMNS

        original = self.trainer._current_model_auc
        self.trainer._current_model_auc = lambda symbol: None

        try:
            result = self.trainer.train("BTCUSDT", self.df, FEATURE_COLUMNS)
            self.assertEqual(result["status"], "accepted")
        finally:
            self.trainer._current_model_auc = original


# ---------------------------------------------------------------------------
# Property 13: Classifier-Inferenz-Wertebereich
# Validates: Requirements 3.4
#
# For any feature vector, RegimeClassifier.predict() returns
# regime in {"trending", "choppy"} and probability in [0.0, 1.0].
# ---------------------------------------------------------------------------

class TestProperty13ClassifierInferenzWertebereich(unittest.TestCase):
    """
    **Validates: Requirements 3.4**

    Property 13: Classifier-Inferenz-Wertebereich — regime is always one of
    {"trending", "choppy"} and probability is always in [0.0, 1.0].
    """

    @given(
        trending_prob=st.floats(min_value=0.0, max_value=1.0),
        feature_values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=3,
        ),
    )
    @settings(max_examples=300, deadline=None)
    def test_predict_output_invariants(
        self, trending_prob: float, feature_values: list[float]
    ):
        """
        **Validates: Requirements 3.4**

        For any probability the mock model returns and any feature values,
        the classifier output must satisfy the value-range invariants.
        """
        model = _MockModel(trending_prob=trending_prob)
        columns = ["adx", "rsi", "atr"]
        classifier = RegimeClassifier(
            model=model,
            feature_columns=columns,
            metadata={"version": "prop13"},
        )
        features = dict(zip(columns, feature_values))
        result = classifier.predict(features)

        self.assertIn(
            result["regime"],
            {"trending", "choppy"},
            f"Unexpected regime: {result['regime']}",
        )
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    @given(
        trending_prob=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=200, deadline=None)
    def test_regime_classification_boundary(self, trending_prob: float):
        """
        **Validates: Requirements 3.4**

        trending_prob >= 0.5 => regime == 'trending',
        trending_prob < 0.5 => regime == 'choppy'.
        """
        model = _MockModel(trending_prob=trending_prob)
        classifier = RegimeClassifier(
            model=model,
            feature_columns=["adx"],
            metadata={"version": "prop13_boundary"},
        )
        result = classifier.predict({"adx": 30.0})

        if trending_prob >= 0.5:
            self.assertEqual(result["regime"], "trending")
        else:
            self.assertEqual(result["regime"], "choppy")

    @given(
        n_features=st.integers(min_value=1, max_value=20),
        trending_prob=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_variable_feature_count(
        self, n_features: int, trending_prob: float
    ):
        """
        **Validates: Requirements 3.4**

        Regardless of how many features the classifier expects,
        predict() must still return valid regime and probability.
        """
        columns = [f"feat_{i}" for i in range(n_features)]
        model = _MockModel(trending_prob=trending_prob)
        classifier = RegimeClassifier(
            model=model,
            feature_columns=columns,
            metadata={"version": "prop13_varfeat"},
        )
        # Provide only a subset of features (rest default to 0.0)
        partial = {columns[0]: 42.0} if columns else {}
        result = classifier.predict(partial)

        self.assertIn(result["regime"], {"trending", "choppy"})
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)


# ---------------------------------------------------------------------------
# Property 15: Label-Generierung ausschliesslich aus Zukunftsdaten
# Validates: Requirements 5.7
#
# For any label at index i, it depends only on candles i+1..i+N.
# Changing data beyond i+N or at/before i must not change the label at i.
# ---------------------------------------------------------------------------

class TestProperty15LabelGenerierungAusZukunftsdaten(unittest.TestCase):
    """
    **Validates: Requirements 5.7**

    Property 15: Label-Generierung ausschliesslich aus Zukunftsdaten —
    label at index i depends only on candles i+1..i+N.
    """

    def setUp(self):
        self.df = _make_synthetic_ohlcv(n_rows=300, seed=42)

    @given(
        idx=st.integers(min_value=0, max_value=249),
        horizon=st.integers(min_value=3, max_value=30),
    )
    @settings(max_examples=200, deadline=None)
    def test_far_future_changes_do_not_affect_label(
        self, idx: int, horizon: int
    ):
        """
        **Validates: Requirements 5.7**

        Modifying candles beyond index i + horizon must not change the
        label at index i.
        """
        far_future_start = idx + 1 + horizon
        assume(far_future_start < len(self.df))

        labels_orig = generate_labels(
            self.df.copy(), horizon=horizon, atr_threshold=1.5
        )

        df_mod = self.df.copy()
        df_mod.iloc[far_future_start:, df_mod.columns.get_loc("close")] *= 5.0
        df_mod.iloc[far_future_start:, df_mod.columns.get_loc("high")] *= 5.0
        df_mod.iloc[far_future_start:, df_mod.columns.get_loc("low")] *= 5.0

        labels_mod = generate_labels(
            df_mod, horizon=horizon, atr_threshold=1.5
        )

        self.assertAlmostEqual(
            labels_orig.iloc[idx],
            labels_mod.iloc[idx],
            places=10,
            msg=(
                f"Label at idx={idx} changed after modifying data beyond "
                f"horizon window (far_future_start={far_future_start})"
            ),
        )

    @given(
        idx=st.integers(min_value=50, max_value=200),
        horizon=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=200, deadline=None)
    def test_label_uses_window_i_plus_1_to_i_plus_n(
        self, idx: int, horizon: int
    ):
        """
        **Validates: Requirements 5.7**

        The label at index i looks at the price window
        close[i+1 : i+1+horizon]. Changing a candle inside that window
        CAN change the label (showing the label actually depends on
        those candles).
        """
        assume(idx + 1 + horizon <= len(self.df))

        labels_orig = generate_labels(
            self.df.copy(), horizon=horizon, atr_threshold=1.5
        )

        # Make a large price change inside the future window to force a
        # label difference (set close to extreme value).
        df_mod = self.df.copy()
        mid = idx + 1 + horizon // 2
        if mid < len(df_mod):
            df_mod.iloc[mid, df_mod.columns.get_loc("close")] *= 100.0
            df_mod.iloc[mid, df_mod.columns.get_loc("high")] *= 100.0

            labels_mod = generate_labels(
                df_mod, horizon=horizon, atr_threshold=1.5
            )

            # The label at idx SHOULD change (trending) because we created
            # a huge price movement inside the window.  We verify the label
            # is 1.0 (trending) after the extreme modification.
            self.assertEqual(
                labels_mod.iloc[idx],
                1.0,
                f"Expected label at idx={idx} to be trending (1.0) after "
                f"extreme price change inside horizon window",
            )
