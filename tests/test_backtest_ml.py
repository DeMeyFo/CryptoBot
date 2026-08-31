# tests/test_backtest_ml.py
"""
Deterministic tests for the ML-Gate integration in portfolio_backtest.py.

**Validates: Requirements 11.2, 11.3, 11.4**

Tests:
  - Backtest without ML gate produces identical results as before (regression).
  - Backtest with ML gate blocks entries when choppy regime is predicted.
  - Comparison output (_print_ml_comparison) contains both metric sets.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path so imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_backtest import (
    SignalFeatures,
    SimParams,
    SymbolData,
    _print_ml_comparison,
    build_timeline,
    performance,
    simulate,
)


# ---------------------------------------------------------------------------
# Helpers: build a small synthetic dataset with deterministic signals
# ---------------------------------------------------------------------------

def _make_bar_ms(n_bars: int, start_ms: int = 1_700_000_000_000,
                 bar_interval_ms: int = 3_600_000) -> np.ndarray:
    """Create an array of bar timestamps in milliseconds."""
    return np.arange(start_ms, start_ms + n_bars * bar_interval_ms,
                     bar_interval_ms, dtype="int64")


def _make_synthetic_symbol_data(
    symbol: str = "TESTUSDT",
    n_bars: int = 200,
    loop_start: int = 50,
    price_base: float = 100.0,
    atr_value: float = 2.0,
    signal_indices: list[int] | None = None,
    signal_side: str = "long",
) -> SymbolData:
    """
    Build a minimal SymbolData with a price pattern that reliably closes
    long trades.

    After each signal bar the price rises for ~15 bars (past 1R activation)
    then drops back sharply to trigger the stop. With ATR=2.0 and the
    default stop multiplier of 2.5, initial stop = entry - 5.0 and the
    trailing activation needs +5.0 from entry (1R).
    """
    bar_ms = _make_bar_ms(n_bars)

    # Build a price pattern with pumps that drop back to trigger stops.
    # Each 40-bar cycle: 15 bars up (exceed 1R), then 25 bars down past stop.
    close = np.empty(n_bars)
    for i in range(n_bars):
        base = price_base + i * 0.05  # very gentle overall uptrend
        cycle_pos = i % 40
        if cycle_pos < 15:
            # Rise phase: +0.6 per bar (9.0 total in 15 bars, well past 1R=5.0)
            close[i] = base + cycle_pos * 0.6
        else:
            # Drop phase: fall back below trailing stop
            close[i] = base + 9.0 - (cycle_pos - 15) * 0.8
    open_ = close * 0.999
    high = close * 1.003
    low = close * 0.997
    atr = np.full(n_bars, atr_value)

    # Signals: place entry signals at the given indices.
    signals: list[SignalFeatures | None] = [None] * n_bars
    if signal_indices is None:
        # Default: one signal roughly every 40 bars starting after loop_start.
        signal_indices = list(range(loop_start + 5, n_bars - 10, 40))

    for idx in signal_indices:
        if 0 <= idx < n_bars:
            signals[idx] = SignalFeatures(
                side=signal_side,
                atr=atr_value,
                score=7.5,
                adx=45.0,
                atr_pct=atr_value / close[idx],
                volume_ratio=1.5,
                rsi=55.0,
                macd_norm=0.3,
                di_spread=15.0,
            )

    # No funding events — keeps the test deterministic and avoids requiring
    # real settlement data.
    funding_events: list[dict] = []

    return SymbolData(
        symbol=symbol,
        source="Binance Futures",
        bar_ms=bar_ms,
        open=open_,
        high=high,
        low=low,
        close=close,
        atr=atr,
        signals=signals,
        funding_events=funding_events,
        loop_start=loop_start,
        adx_series=np.full(n_bars, 45.0),
        rsi_series=np.full(n_bars, 55.0),
        bb_upper=close * 1.04,
        bb_lower=close * 0.96,
        bb_mid=close,
        funding_rate_at_bar=np.full(n_bars, 0.0),
        trade_start=0,
    )


def _run_simulate(datasets: list[SymbolData],
                  params: SimParams) -> dict:
    """Run simulate with a freshly built timeline."""
    timeline = build_timeline(datasets)
    return simulate(datasets, timeline, params)


def _make_mock_trainer_with_predictions(
    predictions: list[tuple[str, str, float]],
) -> MagicMock:
    """Create a mock WalkForwardTrainer that returns the given predictions."""
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.historical_predictions.return_value = predictions
    return mock_trainer_instance


def _ml_gate_patches(mock_trainer_instance: MagicMock):
    """
    Context manager stack that patches the imports inside simulate()'s
    ML gate precomputation block.

    Inside simulate(), the ML gate path does:
        from ml_trainer import WalkForwardTrainer
        from backtest import _load_ohlcv
        from technical_analysis import closed_candles as _cc, prepare_indicators as _pi

    We need to mock ml_trainer as a sys.modules entry (since it's imported
    locally via `from ml_trainer import ...`), and patch _load_ohlcv,
    closed_candles, prepare_indicators at their source modules.
    """
    dummy_frame = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
        "close": [100.0] * 10,
    })

    mock_ml_trainer_mod = MagicMock()
    mock_ml_trainer_mod.WalkForwardTrainer = MagicMock(
        return_value=mock_trainer_instance,
    )

    class _PatchStack:
        def __init__(self):
            self._patches = [
                patch.dict("sys.modules", {"ml_trainer": mock_ml_trainer_mod}),
                patch("backtest._load_ohlcv",
                      return_value=(dummy_frame, "Binance Futures")),
                patch("technical_analysis.closed_candles",
                      return_value=dummy_frame),
                patch("technical_analysis.prepare_indicators",
                      return_value=dummy_frame),
            ]

        def __enter__(self):
            for p in self._patches:
                p.__enter__()
            return self

        def __exit__(self, *args):
            for p in reversed(self._patches):
                p.__exit__(*args)

    return _PatchStack()


# ---------------------------------------------------------------------------
# Test: Backtest without ML gate produces identical results (regression)
# ---------------------------------------------------------------------------

class TestBacktestWithoutMLGateRegression(unittest.TestCase):
    """
    **Validates: Requirement 11.2**

    Verifies that running the backtest twice with ml_gate_enabled=False
    produces bit-identical results, and that the ML gate counter stays zero.
    """

    def test_no_ml_gate_deterministic(self):
        """Two runs with identical inputs and ml_gate_enabled=False must match."""
        data = _make_synthetic_symbol_data(
            signal_indices=[60, 100, 140, 180],
        )
        params = SimParams(
            start_equity=1000.0,
            max_open_positions=2,
            fixed_notional_usdt=100.0,
            apply_margin_ceiling=False,
            ml_gate_enabled=False,
            fee_rate=0.0006,
            slippage_rate=0.0005,
        )

        result1 = _run_simulate([data], params)
        result2 = _run_simulate([data], params)

        # Equity curves must be identical.
        np.testing.assert_array_equal(
            result1["curve_equity"],
            result2["curve_equity"],
            err_msg="Equity curves differ between two identical runs",
        )
        np.testing.assert_array_equal(
            result1["curve_ms"],
            result2["curve_ms"],
        )
        # Trade counts must match.
        self.assertEqual(
            len(result1["closed_trades"]),
            len(result2["closed_trades"]),
        )
        # The ML gate counter must be zero when the gate is off.
        self.assertEqual(result1["skipped_ml_gate"], 0)
        self.assertEqual(result2["skipped_ml_gate"], 0)

    def test_no_ml_gate_default_params(self):
        """ml_gate_enabled defaults to False and has no effect on simulation."""
        data = _make_synthetic_symbol_data(signal_indices=[60, 100])
        params = SimParams(
            start_equity=1000.0,
            max_open_positions=2,
            fixed_notional_usdt=100.0,
            apply_margin_ceiling=False,
        )
        self.assertFalse(params.ml_gate_enabled)
        result = _run_simulate([data], params)
        self.assertEqual(result["skipped_ml_gate"], 0)


# ---------------------------------------------------------------------------
# Test: Backtest with ML gate blocks entries at choppy regime
# ---------------------------------------------------------------------------

class TestBacktestMLGateBlocksChoppyEntries(unittest.TestCase):
    """
    **Validates: Requirements 11.2, 11.3**

    When the ML gate is active and the prediction for a bar is (choppy,
    low confidence), that entry must be blocked. The skipped_ml_gate
    counter must increase. When the regime is trending the entry proceeds
    and risk_scale is applied.
    """

    def _build_datasets_and_params(
        self,
        signal_indices: list[int],
        ml_gate_enabled: bool = True,
    ) -> tuple[list[SymbolData], SimParams]:
        data = _make_synthetic_symbol_data(
            signal_indices=signal_indices,
        )
        params = SimParams(
            start_equity=1000.0,
            max_open_positions=4,
            fixed_notional_usdt=100.0,
            apply_margin_ceiling=False,
            ml_gate_enabled=ml_gate_enabled,
            ml_confidence_threshold=0.5,
            ml_min_risk_scale=0.5,
            fee_rate=0.0006,
            slippage_rate=0.0005,
        )
        return [data], params

    def test_all_choppy_blocks_all_entries(self):
        """
        **Validates: Requirement 11.2**

        When every signal bar is predicted as choppy with low confidence,
        all entries are blocked and no trades are opened.
        """
        signal_indices = [60, 100, 140, 180]
        datasets, params = self._build_datasets_and_params(signal_indices)
        data = datasets[0]

        predictions = []
        for idx in signal_indices:
            ts_ms = int(data.bar_ms[idx])
            ts_str = pd.Timestamp(ts_ms, unit="ms").isoformat()
            predictions.append((ts_str, "choppy", 0.2))

        mock_trainer = _make_mock_trainer_with_predictions(predictions)

        with _ml_gate_patches(mock_trainer):
            result = _run_simulate(datasets, params)

        self.assertEqual(
            result["skipped_ml_gate"],
            len(signal_indices),
            f"Expected {len(signal_indices)} ML-gate blocks, "
            f"got {result['skipped_ml_gate']}",
        )
        self.assertEqual(
            len(result["closed_trades"]),
            0,
            "No trades should be opened when all entries are blocked",
        )

    def test_trending_entries_pass_through(self):
        """
        **Validates: Requirement 11.2**

        When predictions are trending with high confidence, entries are
        NOT blocked. The trade count should match the non-ML-gate run.
        """
        signal_indices = [60, 100, 140]
        datasets, params = self._build_datasets_and_params(signal_indices)
        data = datasets[0]

        predictions = []
        for idx in signal_indices:
            ts_ms = int(data.bar_ms[idx])
            ts_str = pd.Timestamp(ts_ms, unit="ms").isoformat()
            predictions.append((ts_str, "trending", 0.9))

        mock_trainer = _make_mock_trainer_with_predictions(predictions)

        with _ml_gate_patches(mock_trainer):
            result_ml = _run_simulate(datasets, params)

        # Run without ML gate for comparison.
        params_no_ml = replace(params, ml_gate_enabled=False)
        result_no_ml = _run_simulate(datasets, params_no_ml)

        self.assertEqual(result_ml["skipped_ml_gate"], 0)
        # Both runs should produce the same number of trades because
        # trending predictions do not block anything.
        self.assertEqual(
            len(result_ml["closed_trades"]),
            len(result_no_ml["closed_trades"]),
        )

    def test_mixed_regime_blocks_only_choppy(self):
        """
        **Validates: Requirements 11.2, 11.3**

        With mixed predictions (some choppy, some trending), only the
        choppy entries are blocked. Trending entries proceed.
        """
        signal_indices = [60, 100, 140, 180]
        datasets, params = self._build_datasets_and_params(signal_indices)
        data = datasets[0]

        # Bars 60 and 140: choppy (should be blocked)
        # Bars 100 and 180: trending (should pass)
        predictions = []
        for i, idx in enumerate(signal_indices):
            ts_ms = int(data.bar_ms[idx])
            ts_str = pd.Timestamp(ts_ms, unit="ms").isoformat()
            if i % 2 == 0:
                predictions.append((ts_str, "choppy", 0.2))
            else:
                predictions.append((ts_str, "trending", 0.8))

        mock_trainer = _make_mock_trainer_with_predictions(predictions)

        with _ml_gate_patches(mock_trainer):
            result = _run_simulate(datasets, params)

        # Exactly 2 choppy entries should be blocked.
        self.assertEqual(result["skipped_ml_gate"], 2)

    def test_risk_scale_applied_on_trending(self):
        """
        **Validates: Requirement 11.3**

        When a trending prediction with moderate confidence passes through,
        the risk_scale should reduce the position size compared to a run
        without ML gate (which uses risk_scale=1.0).
        """
        signal_indices = [60]
        datasets, params_ml = self._build_datasets_and_params(
            signal_indices, ml_gate_enabled=True,
        )
        # Use risk-based sizing so risk_scale actually affects the notional.
        params_ml = replace(
            params_ml,
            fixed_notional_usdt=None,
            risk_per_trade_pct=0.02,
            notional_cap_usdt=500.0,
            min_notional_usdt=1.0,
            ml_min_risk_scale=0.5,
        )
        data = datasets[0]

        # Trending with confidence=0.6 -> risk_scale = 0.5 + 0.5*0.6 = 0.8
        predictions = []
        ts_ms = int(data.bar_ms[60])
        ts_str = pd.Timestamp(ts_ms, unit="ms").isoformat()
        predictions.append((ts_str, "trending", 0.6))

        mock_trainer = _make_mock_trainer_with_predictions(predictions)

        with _ml_gate_patches(mock_trainer):
            result_ml = _run_simulate(datasets, params_ml)

        # Run without ML gate (risk_scale=1.0).
        params_no_ml = replace(params_ml, ml_gate_enabled=False)
        result_no_ml = _run_simulate(datasets, params_no_ml)

        # Both should have exactly one trade.
        self.assertEqual(len(result_ml["closed_trades"]), 1)
        self.assertEqual(len(result_no_ml["closed_trades"]), 1)

        ml_trade = result_ml["closed_trades"][0]
        no_ml_trade = result_no_ml["closed_trades"][0]

        # The ML-gated trade should have a smaller notional than the baseline.
        self.assertLess(
            ml_trade["size_usdt"],
            no_ml_trade["size_usdt"],
            "ML-gated trade should be smaller due to risk_scale < 1.0",
        )


# ---------------------------------------------------------------------------
# Test: _print_ml_comparison output contains both metric sets
# ---------------------------------------------------------------------------

class TestMLComparisonOutput(unittest.TestCase):
    """
    **Validates: Requirement 11.4**

    The _print_ml_comparison function must produce a side-by-side display
    containing the key performance metrics for both the base and ML-gated runs.
    """

    def _make_stats(self, total_return: float = 50.0,
                    max_dd: float = 10.0,
                    profit_factor: float = 2.0,
                    trades: int = 20,
                    skipped_ml_gate: int = 0) -> dict:
        """Build a minimal stats dict mimicking performance() output."""
        months_index = pd.date_range("2024-01-31", periods=6, freq="ME")
        monthly_pct = pd.Series(
            [5.0, -2.0, 8.0, 3.0, -1.0, 4.0],
            index=months_index,
        )
        return {
            "start_equity": 1000.0,
            "final_equity": 1000.0 * (1.0 + total_return / 100.0),
            "total_return_pct": total_return,
            "geometric_monthly_pct": total_return / 6.0,
            "mean_monthly_pct": float(monthly_pct.mean()),
            "median_monthly_pct": float(monthly_pct.median()),
            "cagr_pct": total_return * 2.0,
            "max_drawdown_pct": max_dd,
            "months": 6,
            "positive_months": 4,
            "worst_month_pct": -2.0,
            "best_month_pct": 8.0,
            "worst_losing_streak_months": 1,
            "positive_months_pct": 66.7,
            "rolling": {},
            "longest_underwater_months": 1,
            "monthly_sortino": 1.5,
            "trades": trades,
            "win_rate_pct": 60.0,
            "profit_factor": profit_factor,
            "net_pnl": total_return * 10,
            "funding": 0.0,
            "fees": 10.0,
            "accounting_residual": 0.0,
            "skipped_no_slot": 0,
            "skipped_no_size": 0,
            "skipped_day_stop": 0,
            "skipped_filtered": 0,
            "skipped_ml_gate": skipped_ml_gate,
            "pyramid_adds": 0,
            "partial_takes": 0,
            "peak_gross_exposure": 1.0,
            "peak_symbol_exposure": 0.5,
            "still_open": 0,
            "ruined": False,
            "monthly_pct": monthly_pct,
        }

    def test_comparison_contains_key_metrics(self):
        """
        **Validates: Requirement 11.4**

        The comparison output must mention Sharpe, Sortino, Max Drawdown,
        Profit Factor, and the labels for both columns.
        """
        base_stats = self._make_stats(
            total_return=50.0, max_dd=10.0, profit_factor=2.0, trades=20,
        )
        ml_stats = self._make_stats(
            total_return=45.0, max_dd=8.0, profit_factor=2.5, trades=15,
            skipped_ml_gate=5,
        )

        captured = io.StringIO()
        with redirect_stdout(captured):
            _print_ml_comparison(base_stats, ml_stats)

        output = captured.getvalue()

        # Both column headers must be present.
        self.assertIn("Ohne ML-Gate", output)
        self.assertIn("Mit ML-Gate", output)

        # Key metrics from the comparison table must appear.
        self.assertIn("Sharpe", output)
        self.assertIn("Sortino", output)
        self.assertIn("Max Drawdown", output)
        self.assertIn("Profit Factor", output)
        self.assertIn("Total Return", output)
        self.assertIn("Monthly Return", output)
        self.assertIn("CAGR", output)
        self.assertIn("Win Rate", output)
        self.assertIn("Trades", output)

        # ML-gate blocked count must appear.
        self.assertIn("ML-Gate blocked entries", output)
        self.assertIn("5", output)

        # The disclaimer must be present.
        self.assertIn("Backtest-Ergebnis", output)

    def test_comparison_contains_monthly_breakdown(self):
        """
        **Validates: Requirement 11.4**

        When both stats have monthly_pct, the output should include a
        monthly side-by-side breakdown with Delta column.
        """
        base_stats = self._make_stats(total_return=50.0)
        ml_stats = self._make_stats(total_return=40.0)

        captured = io.StringIO()
        with redirect_stdout(captured):
            _print_ml_comparison(base_stats, ml_stats)

        output = captured.getvalue()

        # The monthly breakdown header must be present.
        self.assertIn("Month", output)
        self.assertIn("Ohne ML", output)
        self.assertIn("Mit ML", output)
        self.assertIn("Delta", output)

        # At least one month should appear (e.g. 2024-01).
        self.assertIn("2024-01", output)

    def test_comparison_handles_identical_stats(self):
        """When both runs are identical, the comparison still renders."""
        stats = self._make_stats()
        captured = io.StringIO()
        with redirect_stdout(captured):
            _print_ml_comparison(stats, stats)
        output = captured.getvalue()
        self.assertIn("ML-GATE COMPARISON", output)


# ---------------------------------------------------------------------------
# Test: performance() includes skipped_ml_gate metric
# ---------------------------------------------------------------------------

class TestPerformanceIncludesMLGateMetric(unittest.TestCase):
    """
    **Validates: Requirement 11.4**

    The performance() function must include skipped_ml_gate in its output
    so it can be displayed in the comparison.
    """

    def test_performance_has_ml_gate_key(self):
        """performance() output includes skipped_ml_gate."""
        data = _make_synthetic_symbol_data(signal_indices=[60])
        params = SimParams(
            start_equity=1000.0,
            max_open_positions=2,
            fixed_notional_usdt=100.0,
            apply_margin_ceiling=False,
            ml_gate_enabled=False,
        )
        result = _run_simulate([data], params)
        stats = performance(result)
        self.assertIn("skipped_ml_gate", stats)
        self.assertEqual(stats["skipped_ml_gate"], 0)


if __name__ == "__main__":
    unittest.main()
