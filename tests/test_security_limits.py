# tests/test_security_limits.py
"""
Property-Test fuer bestehende Sicherheits-Limits in risk_manager.py.

**Property 5: Bestehende Exposure-Limits als harte Obergrenze** —
calculate_position_notional() ueberschreitet nie die Caps unabhaengig
vom risk_scale-Wert.

**Validates: Requirements 2.3, 8.1**
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from risk_manager import (
    calculate_position_notional,
    exposure_headroom,
    position_notional_cap,
    position_notional_floor,
)


# ---------------------------------------------------------------------------
# Strategies for generating realistic trading parameters
# ---------------------------------------------------------------------------

# Equity: positive values representing account balance in USDT
equity_st = st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)

# Entry prices: positive, representing crypto prices
entry_price_st = st.floats(min_value=0.01, max_value=100_000.0, allow_nan=False, allow_infinity=False)

# Stop loss distance as a fraction of entry price (0.1% to 20%)
sl_fraction_st = st.floats(min_value=0.001, max_value=0.20, allow_nan=False, allow_infinity=False)

# risk_scale: the ML-layer scaling factor, including values > 1.0
risk_scale_st = st.sampled_from([0.1, 0.5, 1.0, 2.0, 10.0]) | st.floats(
    min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
)

# Exposure values: non-negative, representing current exposure in USDT
exposure_st = st.floats(min_value=0.0, max_value=500_000.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 5: Bestehende Exposure-Limits als harte Obergrenze
# Validates: Requirements 2.3, 8.1
# ---------------------------------------------------------------------------

class TestProperty5ExposureLimitsHardCap(unittest.TestCase):
    """
    **Validates: Requirements 2.3, 8.1**

    Property 5: Bestehende Exposure-Limits als harte Obergrenze —
    calculate_position_notional() never exceeds the configured caps
    regardless of risk_scale.
    """

    @given(
        equity=equity_st,
        entry_price=entry_price_st,
        sl_fraction=sl_fraction_st,
        risk_scale=risk_scale_st,
        symbol_exposure=exposure_st,
        gross_exposure=exposure_st,
    )
    @settings(max_examples=500, deadline=None)
    def test_notional_never_exceeds_position_cap(
        self,
        equity: float,
        entry_price: float,
        sl_fraction: float,
        risk_scale: float,
        symbol_exposure: float,
        gross_exposure: float,
    ):
        """
        **Validates: Requirement 2.3**

        The result of calculate_position_notional() must never exceed
        position_notional_cap(equity), which is the configured ceiling
        for one entry (MAX_POSITION_PCT * equity in percent mode, or
        min(POSITION_SIZE_USDT, MAX_POSITION_USDT) in absolute mode).
        """
        side = "long"
        stop_loss = entry_price * (1 - sl_fraction)
        max_notional = position_notional_cap(equity)

        result = calculate_position_notional(
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            max_notional=max_notional,
            symbol_exposure_usdt=symbol_exposure,
            gross_exposure_usdt=gross_exposure,
            risk_scale=risk_scale,
        )

        self.assertGreaterEqual(result, 0.0, "Position notional must be non-negative")
        self.assertLessEqual(
            result,
            max_notional + 0.01,  # tolerance for rounding
            f"Position notional {result} exceeded max_notional cap {max_notional} "
            f"with risk_scale={risk_scale}",
        )

    @given(
        equity=equity_st,
        entry_price=entry_price_st,
        sl_fraction=sl_fraction_st,
        risk_scale=risk_scale_st,
        symbol_exposure=exposure_st,
        gross_exposure=exposure_st,
    )
    @settings(max_examples=500, deadline=None)
    def test_notional_never_exceeds_symbol_exposure_headroom(
        self,
        equity: float,
        entry_price: float,
        sl_fraction: float,
        risk_scale: float,
        symbol_exposure: float,
        gross_exposure: float,
    ):
        """
        **Validates: Requirement 2.3**

        The result must never exceed the per-symbol exposure headroom
        (MAX_SYMBOL_EXPOSURE_PCT * equity - symbol_exposure_usdt).
        """
        side = "long"
        stop_loss = entry_price * (1 - sl_fraction)
        max_notional = position_notional_cap(equity)

        result = calculate_position_notional(
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            max_notional=max_notional,
            symbol_exposure_usdt=symbol_exposure,
            gross_exposure_usdt=gross_exposure,
            risk_scale=risk_scale,
        )

        headroom = exposure_headroom(equity, symbol_exposure, gross_exposure)
        if result > 0:
            self.assertLessEqual(
                result,
                headroom + 0.01,  # tolerance for rounding
                f"Position notional {result} exceeded exposure headroom {headroom} "
                f"with risk_scale={risk_scale}",
            )

    @given(
        equity=equity_st,
        entry_price=entry_price_st,
        sl_fraction=sl_fraction_st,
        risk_scale=risk_scale_st,
        symbol_exposure=exposure_st,
        gross_exposure=exposure_st,
    )
    @settings(max_examples=500, deadline=None)
    def test_notional_never_exceeds_gross_exposure_limit(
        self,
        equity: float,
        entry_price: float,
        sl_fraction: float,
        risk_scale: float,
        symbol_exposure: float,
        gross_exposure: float,
    ):
        """
        **Validates: Requirement 8.1**

        The result must never exceed MAX_GROSS_EXPOSURE_PCT * equity -
        gross_exposure_usdt, which is the gross exposure ceiling.
        """
        import config

        side = "long"
        stop_loss = entry_price * (1 - sl_fraction)
        max_notional = position_notional_cap(equity)

        result = calculate_position_notional(
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            max_notional=max_notional,
            symbol_exposure_usdt=symbol_exposure,
            gross_exposure_usdt=gross_exposure,
            risk_scale=risk_scale,
        )

        gross_limit = equity * config.MAX_GROSS_EXPOSURE_PCT - max(0.0, gross_exposure)
        if result > 0:
            self.assertLessEqual(
                result,
                gross_limit + 0.01,  # tolerance for rounding
                f"Position notional {result} exceeded gross exposure limit {gross_limit} "
                f"with risk_scale={risk_scale}",
            )

    @given(
        equity=equity_st,
        entry_price=entry_price_st,
        sl_fraction=sl_fraction_st,
    )
    @settings(max_examples=200, deadline=None)
    def test_high_risk_scale_does_not_increase_beyond_normal_limits(
        self,
        equity: float,
        entry_price: float,
        sl_fraction: float,
    ):
        """
        **Validates: Requirements 2.3, 8.1**

        A risk_scale > 1.0 must NOT produce a larger position than the
        hard caps allow. Compare risk_scale=1.0 with risk_scale=10.0:
        the result at scale=10.0 must not exceed the exposure headroom
        or position cap, even though the risk budget is scaled up.
        """
        side = "long"
        stop_loss = entry_price * (1 - sl_fraction)
        max_notional = position_notional_cap(equity)

        result_normal = calculate_position_notional(
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            max_notional=max_notional,
            symbol_exposure_usdt=0.0,
            gross_exposure_usdt=0.0,
            risk_scale=1.0,
        )

        result_high = calculate_position_notional(
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            max_notional=max_notional,
            symbol_exposure_usdt=0.0,
            gross_exposure_usdt=0.0,
            risk_scale=10.0,
        )

        # Both must respect the same hard caps
        ceiling = min(
            max_notional,
            exposure_headroom(equity, 0.0, 0.0),
        )

        self.assertLessEqual(
            result_high,
            ceiling + 0.01,
            f"risk_scale=10.0 produced {result_high} exceeding ceiling {ceiling}",
        )

        # The high-scale result cannot exceed the ceiling even if it is
        # larger than the normal-scale result
        self.assertLessEqual(
            result_high,
            ceiling + 0.01,
            f"risk_scale=10.0 exceeded the hard cap: {result_high} > {ceiling}",
        )

    @given(
        equity=equity_st,
        entry_price=entry_price_st,
        sl_fraction=sl_fraction_st,
    )
    @settings(max_examples=200, deadline=None)
    def test_specific_risk_scale_values_respect_caps(
        self,
        equity: float,
        entry_price: float,
        sl_fraction: float,
    ):
        """
        **Validates: Requirements 2.3, 8.1**

        Verify all the specified risk_scale values (0.1, 0.5, 1.0, 2.0, 10.0)
        never produce a result exceeding the caps.
        """
        stop_loss = entry_price * (1 - sl_fraction)
        max_notional = position_notional_cap(equity)
        ceiling = min(
            max_notional,
            exposure_headroom(equity, 0.0, 0.0),
        )

        for scale in [0.1, 0.5, 1.0, 2.0, 10.0]:
            result = calculate_position_notional(
                equity=equity,
                entry_price=entry_price,
                stop_loss=stop_loss,
                max_notional=max_notional,
                symbol_exposure_usdt=0.0,
                gross_exposure_usdt=0.0,
                risk_scale=scale,
            )

            self.assertGreaterEqual(result, 0.0, f"Negative notional at scale={scale}")
            self.assertLessEqual(
                result,
                ceiling + 0.01,
                f"At risk_scale={scale}, notional {result} exceeded ceiling {ceiling}",
            )

    @given(
        equity=equity_st,
        entry_price=entry_price_st,
        sl_fraction=sl_fraction_st,
        risk_scale=risk_scale_st,
    )
    @settings(max_examples=300, deadline=None)
    def test_result_zero_or_above_floor(
        self,
        equity: float,
        entry_price: float,
        sl_fraction: float,
        risk_scale: float,
    ):
        """
        **Validates: Requirement 2.3**

        The result is either 0.0 (entry skipped) or >= position_notional_floor(equity).
        There is no middle ground: positions too small to be meaningful are not opened.
        """
        stop_loss = entry_price * (1 - sl_fraction)
        max_notional = position_notional_cap(equity)

        result = calculate_position_notional(
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            max_notional=max_notional,
            symbol_exposure_usdt=0.0,
            gross_exposure_usdt=0.0,
            risk_scale=risk_scale,
        )

        floor = position_notional_floor(equity)
        if result > 0:
            self.assertGreaterEqual(
                result,
                floor - 0.01,  # tolerance for rounding
                f"Non-zero result {result} is below floor {floor}",
            )

    def test_zero_equity_returns_zero(self):
        """Edge case: zero equity always returns zero regardless of risk_scale."""
        for scale in [0.1, 0.5, 1.0, 2.0, 10.0]:
            result = calculate_position_notional(
                equity=0.0,
                entry_price=50000.0,
                stop_loss=49000.0,
                max_notional=1000.0,
                risk_scale=scale,
            )
            self.assertEqual(result, 0.0)

    def test_zero_entry_price_returns_zero(self):
        """Edge case: zero entry price always returns zero."""
        for scale in [0.1, 0.5, 1.0, 2.0, 10.0]:
            result = calculate_position_notional(
                equity=10000.0,
                entry_price=0.0,
                stop_loss=0.0,
                max_notional=1000.0,
                risk_scale=scale,
            )
            self.assertEqual(result, 0.0)

    def test_zero_max_notional_returns_zero(self):
        """Edge case: zero max_notional always returns zero."""
        for scale in [0.1, 0.5, 1.0, 2.0, 10.0]:
            result = calculate_position_notional(
                equity=10000.0,
                entry_price=50000.0,
                stop_loss=49000.0,
                max_notional=0.0,
                risk_scale=scale,
            )
            self.assertEqual(result, 0.0)

    def test_negative_risk_scale_treated_as_zero(self):
        """
        **Validates: Requirement 8.1**

        Negative risk_scale is clamped to 0 via max(0.0, risk_scale),
        producing zero risk budget and thus zero notional.
        """
        result = calculate_position_notional(
            equity=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            max_notional=5000.0,
            risk_scale=-1.0,
        )
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
