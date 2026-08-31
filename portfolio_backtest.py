"""
Portfolio-level, cost-aware backtest with one shared, compounding equity curve.

Why this exists next to backtest.py
-----------------------------------
backtest.py answers "does the signal core have an edge?". It trades a fixed
notional, keeps one position slot per symbol, starts from cash 0.0 and reports
`roi_pct = pnl / POSITION_SIZE_USDT`. That is profit relative to a single
position size, not the return of an account, so it cannot express a target like
"10% per month".

This module simulates the live portfolio contract instead:
  * one shared equity that compounds across all symbols,
  * MAX_OPEN_POSITIONS as a global slot limit (not one slot per symbol),
  * the live risk-based sizing function including its notional cap and floor,
  * a leverage/margin ceiling equivalent to main._available_notional_cap(),
  * funding settled per symbol before stops and before next-open entries,
  * ruin detection, because leveraged compounding can end at zero.

Per symbol and bar the event order matches backtest.py exactly:
funding -> next-open entry -> stop/trail -> mark-to-market -> signal at close.

Costs are parameters here so cost sensitivity can be measured without editing
config. With --replicate the engine reproduces the backtest.py contract
(fixed notional, one slot per symbol, no margin ceiling) so this
reimplementation can be checked against the already reviewed baseline.
"""

import argparse
import pickle
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import (
    WARMUP,
    FundingDataError,
    _align_funding_events_to_bars,
    _load_ohlcv,
    fetch_binance_funding_history,
    funding_payment,
)
from bitget_client import get_top_symbols
from config import (
    ADX_ENTRY_THRESHOLD,
    DAILY_LOSS_LIMIT_USDT,
    DONCHIAN_PERIOD,
    EMA_FAST_PERIOD,
    EMA_MID_PERIOD,
    EMA_SLOW_PERIOD,
    LEVERAGE,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_USDT,
    MIN_POSITION_USDT,
    POSITION_SIZE_USDT,
    RISK_PER_TRADE_PCT,
    SLIPPAGE_RATE,
    STRATEGY_TIMEFRAME,
    TAKER_FEE_RATE,
    TRAILING_ACTIVATION_R,
    TREND_STOP_ATR_MULTIPLIER,
    TREND_TRAIL_ATR_MULTIPLIER,
)
from technical_analysis import closed_candles, evaluate_signal, prepare_indicators


class DataUnavailable(RuntimeError):
    """Required history for one symbol could not be loaded completely."""


def to_epoch_ms(timestamps) -> np.ndarray:
    """
    Convert a timestamp column to integer epoch milliseconds.

    pandas may hand back datetime64 in second, millisecond, microsecond or
    nanosecond resolution depending on how the frame was built, so the integer
    view cannot be scaled by a hard-coded factor.
    """
    index = pd.DatetimeIndex(timestamps)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return (
        index.tz_localize(None).astype("datetime64[ms]").astype("int64").to_numpy()
    )


@dataclass
class SignalFeatures:
    """
    Indicator readings captured at the moment a setup completed.

    Storing them lets entry-quality filters be swept at simulation time. Every
    filter can only remove signals the configured entry rules already accepted,
    so filtering here is exact and needs no recomputation.
    """

    side: str
    atr: float
    score: float
    adx: float
    atr_pct: float
    volume_ratio: float
    rsi: float
    macd_norm: float
    di_spread: float


@dataclass
class SymbolData:
    """Immutable, pre-computed inputs for one symbol. Reused across sweep runs."""

    symbol: str
    source: str
    bar_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    signals: list
    funding_events: list
    loop_start: int
    # Series needed by the alternative return sources. They are stored rather
    # than folded into `signals` so their thresholds can be swept for free.
    adx_series: np.ndarray | None = None
    rsi_series: np.ndarray | None = None
    bb_upper: np.ndarray | None = None
    bb_lower: np.ndarray | None = None
    bb_mid: np.ndarray | None = None
    funding_rate_at_bar: np.ndarray | None = None
    trade_start: int = 0


@dataclass
class SimParams:
    """One configuration under test."""

    start_equity: float = 1000.0
    risk_per_trade_pct: float = RISK_PER_TRADE_PCT
    max_open_positions: int = MAX_OPEN_POSITIONS
    # Notional cap: either a fixed USDT amount or a fraction of current equity.
    notional_cap_usdt: float | None = None
    notional_cap_pct: float | None = None
    min_notional_usdt: float = MIN_POSITION_USDT
    fee_rate: float = TAKER_FEE_RATE
    slippage_rate: float = SLIPPAGE_RATE
    leverage: float = LEVERAGE
    apply_margin_ceiling: bool = True
    fixed_notional_usdt: float | None = None
    # main._daily_loss_limit_hit() blocks new entries for the rest of the UTC day.
    # Absolute wins if both are set; None on both disables the block.
    daily_loss_limit_usdt: float | None = None
    daily_loss_limit_pct: float | None = None
    # Research knobs. Defaults reproduce the live risk_manager contract exactly.
    allowed_sides: tuple = ("long", "short")
    stop_atr_multiplier: float = TREND_STOP_ATR_MULTIPLIER
    trail_atr_multiplier: float = TREND_TRAIL_ATR_MULTIPLIER
    trailing_activation_r: float = TRAILING_ACTIVATION_R
    # Pyramiding: add to a winner every `step_r` of favourable excursion,
    # measured in units of the original entry risk. 0 adds keeps it disabled.
    pyramid_max_adds: int = 0
    pyramid_step_r: float = 1.0
    pyramid_risk_fraction: float = 1.0
    # Exposure ceilings as a multiple of current equity. Risk-based sizing only
    # bounds the loss when the stop fills at its price; these bound the loss
    # when it does not (gap, liquidity hole, dead process). None disables.
    max_symbol_exposure: float | None = None
    max_gross_exposure: float | None = None
    # Partial profit taking: realise `fraction` of the position once the trade
    # is `at_r` in profit, and let the rest run on the same trailing stop. This
    # trades tail size for consistency, so it is judged on the monthly
    # distribution rather than on total return alone. 0 disables.
    partial_take_at_r: float = 0.0
    partial_take_fraction: float = 0.0
    # Entry-quality filters. Losing months come from many small stop-outs in
    # chop, so these attack the loss side instead of trimming the winners.
    min_adx: float = 0.0
    min_atr_pct: float = 0.0
    max_atr_pct: float = 0.0
    min_volume_ratio: float = 0.0
    min_rsi: float = 0.0
    max_rsi: float = 100.0
    min_macd_norm: float = float("-inf")
    min_di_spread: float = 0.0

    # ── Alternative return sources ────────────────────────────────────────
    # "trend"   : the validated Donchian breakout with an ATR trailing stop.
    # "meanrev" : Bollinger/RSI reversion inside a non-trending regime. Aims to
    #             earn in exactly the chop where the trend sleeve bleeds.
    # "funding" : contrarian positioning signal from an extreme funding rate.
    #             A very positive rate means crowded longs.
    strategy: str = "trend"
    # Mean reversion. Regime is deliberately disjoint from the trend sleeve.
    mr_max_adx: float = 20.0
    mr_rsi_long: float = 30.0
    mr_rsi_short: float = 70.0
    mr_stop_atr: float = 2.0
    mr_time_stop_bars: int = 48
    # Funding extreme. Rates are per settlement, not annualised.
    fund_short_rate: float = 0.0005
    fund_long_rate: float = -0.0003
    fund_hold_bars: int = 24
    fund_stop_atr: float = 2.5
    fund_target_atr: float = 0.0
    # "contrarian" fades crowded positioning: a very positive rate means longs
    # are crowded, so it sells. "momentum" does the opposite and treats a high
    # rate as confirmation that the crowd is being paid to stay.
    fund_direction: str = "contrarian"

    # ── ML-Gate (Anforderung 11) ──────────────────────────────────────────
    # When enabled the backtest queries historical regime predictions from
    # WalkForwardTrainer and blocks entries that the ML model considers choppy
    # with low confidence.  Disabled by default so the existing behaviour is
    # unchanged.
    ml_gate_enabled: bool = False
    ml_model_dir: str = ".models/"
    ml_confidence_threshold: float = 0.5
    ml_min_risk_scale: float = 0.5

    def label(self) -> str:
        if self.fixed_notional_usdt is not None:
            sizing = f"fixed {self.fixed_notional_usdt:.0f} USDT"
        else:
            sizing = f"risk {self.risk_per_trade_pct * 100:.2f}%"
            if self.notional_cap_pct is not None:
                sizing += f", cap {self.notional_cap_pct * 100:.0f}% eq"
            else:
                sizing += f", cap {self.notional_cap_usdt:.0f} USDT"
        if self.daily_loss_limit_usdt is not None:
            limit = f", day-stop {self.daily_loss_limit_usdt:.0f} USDT"
        elif self.daily_loss_limit_pct is not None:
            limit = f", day-stop {self.daily_loss_limit_pct * 100:.1f}% eq"
        else:
            limit = ", no day-stop"
        extra = ""
        if tuple(self.allowed_sides) != ("long", "short"):
            extra += f", {'/'.join(self.allowed_sides)}-only"
        if (self.stop_atr_multiplier != TREND_STOP_ATR_MULTIPLIER
                or self.trail_atr_multiplier != TREND_TRAIL_ATR_MULTIPLIER):
            extra += (
                f", stop {self.stop_atr_multiplier:g}atr"
                f"/trail {self.trail_atr_multiplier:g}atr"
            )
        if self.trailing_activation_r != TRAILING_ACTIVATION_R:
            extra += f", act {self.trailing_activation_r:g}R"
        if self.pyramid_max_adds > 0:
            extra += (
                f", pyramid {self.pyramid_max_adds}x every "
                f"{self.pyramid_step_r:g}R @{self.pyramid_risk_fraction:g}risk"
            )
        if self.max_symbol_exposure is not None:
            extra += f", sym<={self.max_symbol_exposure:g}x"
        if self.max_gross_exposure is not None:
            extra += f", gross<={self.max_gross_exposure:g}x"
        if self.partial_take_at_r > 0 and self.partial_take_fraction > 0:
            extra += (
                f", take {self.partial_take_fraction * 100:.0f}% @"
                f"{self.partial_take_at_r:g}R"
            )
        if self.min_adx > 0:
            extra += f", ADX>={self.min_adx:g}"
        if self.min_di_spread > 0:
            extra += f", DIspread>={self.min_di_spread:g}"
        if self.min_volume_ratio > 0:
            extra += f", vol>={self.min_volume_ratio:g}x"
        if self.min_rsi > 0:
            extra += f", RSI>={self.min_rsi:g}"
        if self.max_rsi < 100:
            extra += f", RSI<={self.max_rsi:g}"
        if self.min_macd_norm > float("-inf"):
            extra += f", MACD>={self.min_macd_norm:g}atr"
        if self.min_atr_pct > 0:
            extra += f", ATR>={self.min_atr_pct * 100:g}%"
        if self.max_atr_pct > 0:
            extra += f", ATR<={self.max_atr_pct * 100:g}%"
        if self.strategy == "meanrev":
            extra += (
                f", meanrev ADX<{self.mr_max_adx:g} "
                f"RSI{self.mr_rsi_long:g}/{self.mr_rsi_short:g} "
                f"stop {self.mr_stop_atr:g}atr time {self.mr_time_stop_bars}b"
            )
        elif self.strategy == "funding":
            extra += (
                f", funding {self.fund_direction} "
                f"|rate|>{self.fund_short_rate * 100:g}%/"
                f"{self.fund_long_rate * 100:g}% "
                f"hold {self.fund_hold_bars}b stop {self.fund_stop_atr:g}atr"
            )
        if self.ml_gate_enabled:
            extra += ", ML-Gate"
        return (
            f"[{self.strategy}] {sizing}, max {self.max_open_positions} "
            f"pos{limit}{extra}"
        )


# ── Cost-parametrised fill and close (mirrors backtest.py formulas) ───────────

def _adverse_fill(raw_price: float, side: str, entry: bool,
                  slippage_rate: float) -> float:
    if entry:
        factor = 1 + slippage_rate if side == "long" else 1 - slippage_rate
    else:
        factor = 1 - slippage_rate if side == "long" else 1 + slippage_rate
    return raw_price * factor


def _close_position(trade: dict, raw_exit: float, result: str,
                    exit_index: int, params: SimParams) -> dict:
    exit_price = _adverse_fill(raw_exit, trade["side"], False, params.slippage_rate)
    direction = 1.0 if trade["side"] == "long" else -1.0
    gross_pnl = (exit_price - trade["entry"]) * trade["quantity"] * direction
    exit_fee = exit_price * trade["quantity"] * params.fee_rate
    total_fees = trade["entry_fee"] + exit_fee
    # A partial take was already credited to cash when it happened, so
    # `price_fee_pnl` must stay the cash effect of this final leg only.
    # `pnl` is the statistic for the whole position and therefore includes it.
    realised_partial = float(trade.get("realized_partial") or 0.0)
    partial_fees = float(trade.get("partial_fees") or 0.0)
    price_fee_pnl = gross_pnl - total_fees
    funding = float(trade.get("funding") or 0.0)
    trade.update({
        "exit_price": exit_price,
        "result": "trail" if result == "sl" and gross_pnl > 0 else result,
        "fees": total_fees + partial_fees,
        "price_fee_pnl": price_fee_pnl,
        "realized_partial": realised_partial,
        "funding": funding,
        "pnl": price_fee_pnl + funding + realised_partial,
        "duration": exit_index - trade["entry_idx"],
    })
    return trade


def _unrealised_price_fee_pnl(trade: dict, close_price: float,
                              params: SimParams) -> float:
    """Mark-to-market excluding funding, which is already booked to cash."""
    hypothetical_exit = _adverse_fill(
        close_price, trade["side"], False, params.slippage_rate
    )
    direction = 1.0 if trade["side"] == "long" else -1.0
    gross = (hypothetical_exit - trade["entry"]) * trade["quantity"] * direction
    exit_fee = hypothetical_exit * trade["quantity"] * params.fee_rate
    return gross - trade["entry_fee"] - exit_fee


def _stop_from_atr(side: str, entry_price: float, atr: float,
                   multiplier: float) -> float:
    offset = atr * multiplier
    if side == "long":
        return round(entry_price - offset, 8)
    return round(entry_price + offset, 8)


def _initial_stop(side: str, entry_price: float, atr: float,
                  params: SimParams) -> float:
    """Mirror of risk_manager.calculate_atr_sl_tp with a tunable multiplier."""
    return _stop_from_atr(side, entry_price, atr, params.stop_atr_multiplier)


def _entry_candidate(data: SymbolData, index: int, bar_close: float,
                     params: SimParams) -> dict | None:
    """
    Entry decision for the configured return source, judged on a closed bar.

    Returns the side plus the exit contract the position will be managed under,
    because the sleeves exit differently: the trend sleeve rides an ATR trailing
    stop, the others aim at a fixed target with a time stop.
    """
    if params.strategy == "trend":
        signal = data.signals[index]
        if signal is None or signal.side not in params.allowed_sides:
            return None
        accepted = (
            signal.adx >= params.min_adx
            and signal.di_spread >= params.min_di_spread
            and signal.volume_ratio >= params.min_volume_ratio
            and params.min_rsi <= signal.rsi <= params.max_rsi
            and signal.macd_norm >= params.min_macd_norm
            and (params.min_atr_pct <= 0 or signal.atr_pct >= params.min_atr_pct)
            and (params.max_atr_pct <= 0 or signal.atr_pct <= params.max_atr_pct)
        )
        if not accepted:
            return {"rejected": True}
        return {
            "side": signal.side, "atr": signal.atr, "score": signal.score,
            "exit_mode": "trail", "stop_atr": params.stop_atr_multiplier,
            "target": None, "time_stop_bars": 0,
        }

    atr = float(data.atr[index])
    if not np.isfinite(atr) or atr <= 0:
        return None

    if params.strategy == "meanrev":
        adx = float(data.adx_series[index])
        rsi = float(data.rsi_series[index])
        upper = float(data.bb_upper[index])
        lower = float(data.bb_lower[index])
        middle = float(data.bb_mid[index])
        if not all(np.isfinite(value) for value in (adx, rsi, upper, lower, middle)):
            return None
        # Only inside a non-trending regime, which is disjoint from the trend
        # sleeve's ADX gate, so the two sources cannot fire on the same bar.
        if adx >= params.mr_max_adx:
            return None
        side = None
        if bar_close < lower and rsi <= params.mr_rsi_long:
            side = "long"
        elif bar_close > upper and rsi >= params.mr_rsi_short:
            side = "short"
        if side is None or side not in params.allowed_sides:
            return None
        # The target must sit in front of the entry, otherwise the trade has no
        # room to work.
        if (side == "long" and middle <= bar_close) or (
                side == "short" and middle >= bar_close):
            return None
        return {
            "side": side, "atr": atr, "score": 0.0, "exit_mode": "target",
            "stop_atr": params.mr_stop_atr, "target": middle,
            "time_stop_bars": params.mr_time_stop_bars,
        }

    if params.strategy == "funding":
        rate = float(data.funding_rate_at_bar[index])
        if not np.isfinite(rate):
            return None
        side = None
        if rate >= params.fund_short_rate:
            side = "short" if params.fund_direction == "contrarian" else "long"
        elif rate <= params.fund_long_rate:
            side = "long" if params.fund_direction == "contrarian" else "short"
        if side is None or side not in params.allowed_sides:
            return None
        target = None
        if params.fund_target_atr > 0:
            offset = atr * params.fund_target_atr
            target = bar_close + offset if side == "long" else bar_close - offset
        return {
            "side": side, "atr": atr, "score": 0.0, "exit_mode": "target",
            "stop_atr": params.fund_stop_atr, "target": target,
            "time_stop_bars": params.fund_hold_bars,
        }

    raise ValueError(f"unknown strategy {params.strategy!r}")


def _next_trailing_stop(trade: dict, candle_close: float, candle_high: float,
                        candle_low: float, atr: float,
                        params: SimParams) -> tuple[float, float, bool]:
    """Mirror of risk_manager.next_atr_trailing_stop with tunable multipliers."""
    side = trade["side"]
    entry = float(trade["entry_price"])
    current_stop = float(trade.get("stop_loss") or 0)
    initial_stop = float(trade.get("initial_stop_loss") or current_stop)
    initial_risk = abs(entry - initial_stop)
    if initial_risk <= 0 or atr <= 0:
        return current_stop, float(trade.get("best_price") or entry), False

    previous_best = float(trade.get("best_price") or entry)
    if side == "long":
        best_price = max(previous_best, candle_high)
        reached_r = (best_price - entry) / initial_risk
        if reached_r < params.trailing_activation_r:
            return current_stop, best_price, False
        candidate = candle_close - atr * params.trail_atr_multiplier
        new_stop = max(current_stop, candidate)
    else:
        best_price = min(previous_best, candle_low)
        reached_r = (entry - best_price) / initial_risk
        if reached_r < params.trailing_activation_r:
            return current_stop, best_price, False
        candidate = candle_close + atr * params.trail_atr_multiplier
        new_stop = min(current_stop, candidate) if current_stop else candidate
    return round(new_stop, 8), round(best_price, 8), new_stop != current_stop


def _size_notional(equity: float, entry_price: float, stop_loss: float,
                   max_notional: float, params: SimParams,
                   risk_scale: float = 1.0) -> float:
    """Mirror of risk_manager.calculate_position_notional with tunable risk."""
    if equity <= 0 or entry_price <= 0 or max_notional <= 0:
        return 0.0
    stop_fraction = abs(entry_price - stop_loss) / entry_price
    roundtrip_cost_fraction = 2 * (params.fee_rate + params.slippage_rate)
    loss_fraction = stop_fraction + roundtrip_cost_fraction
    if loss_fraction <= 0:
        return 0.0
    risk_budget = equity * params.risk_per_trade_pct * risk_scale
    notional = min(max_notional, risk_budget / loss_fraction)
    if notional < params.min_notional_usdt:
        return 0.0
    return round(notional, 2)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_symbol_data(symbol: str, granularity: str, limit: int,
                     months: int) -> SymbolData:
    raw_frame, source = _load_ohlcv(symbol, granularity, limit, months)
    raw_frame = closed_candles(raw_frame, granularity)
    if raw_frame.empty or len(raw_frame) < WARMUP + 10:
        raise DataUnavailable(f"{symbol}: only {len(raw_frame)} candles")

    frame = prepare_indicators(raw_frame)
    if months > 0:
        evaluation_start = (
            frame["timestamp"].iloc[-1] - pd.Timedelta(days=months * 30.44)
        ).floor("h")
        evaluation_index = int(frame["timestamp"].searchsorted(evaluation_start))
    else:
        evaluation_index = WARMUP
    loop_start = max(WARMUP, evaluation_index - 1)
    if loop_start >= len(frame) - 2:
        raise DataUnavailable(f"{symbol}: evaluation window is empty")

    bar_ms = to_epoch_ms(frame["timestamp"])

    # Funding fails closed, exactly like the per-symbol study. A percentage
    # return that silently ignores funding cashflow would be misleading.
    if source != "Binance Futures":
        raise DataUnavailable(
            f"{symbol}: funding history requires Binance Futures data "
            "(use --months to select that source)"
        )
    events = fetch_binance_funding_history(
        symbol, int(bar_ms[loop_start]), int(bar_ms[-1])
    )
    events, unaligned = _align_funding_events_to_bars(
        events, [int(value) for value in bar_ms[loop_start:]], tolerance_ms=1000
    )
    if unaligned:
        raise DataUnavailable(
            f"{symbol}: {len(unaligned)} funding settlement(s) fall inside "
            f"{granularity} candles; ordering cannot be resolved"
        )

    # ADX and the ATR/price ratio are stored with each signal so trend-strength
    # and volatility filters can be swept at simulation time. Both only remove
    # signals that the configured entry rules already accepted, so filtering
    # here is exact and needs no recomputation.
    signals: list = [None] * len(frame)
    for index in range(loop_start, len(frame)):
        signal = evaluate_signal(frame, index)
        if signal.get("entry_signal") in ("long", "short"):
            indicators = signal.get("indicators") or {}
            price = float(indicators.get("current_price") or 0.0)
            atr = float(signal["atr"])
            macd_hist = float(indicators.get("macd_hist") or 0.0)
            signals[index] = SignalFeatures(
                side=signal["entry_signal"],
                atr=atr,
                score=float(signal["score"]),
                adx=float(indicators.get("adx") or 0.0),
                atr_pct=(atr / price) if price > 0 else 0.0,
                volume_ratio=float(indicators.get("volume_ratio") or 0.0),
                rsi=float(indicators.get("rsi") or 50.0),
                # Normalised by ATR so the value is comparable across symbols
                # and price levels.
                macd_norm=(macd_hist / atr) if atr > 0 else 0.0,
                di_spread=abs(
                    float(indicators.get("di_pos") or 0.0)
                    - float(indicators.get("di_neg") or 0.0)
                ),
            )

    # Bollinger bands for the mean-reversion sleeve. Computed here rather than
    # in technical_analysis.py so an unvalidated research idea never touches the
    # indicator set the live bot depends on.
    close_series = frame["close"]
    bb_mid = close_series.rolling(20).mean()
    bb_std = close_series.rolling(20).std(ddof=0)

    # Most recent settled funding rate as of each bar. Only settlements at or
    # before the bar timestamp are visible, so a decision taken on that bar's
    # close cannot see a future rate.
    event_times = np.asarray(
        [int(event["funding_time"]) for event in events], dtype="int64"
    )
    event_rates = np.asarray(
        [float(event["funding_rate"]) for event in events], dtype=float
    )
    funding_at_bar = np.full(len(bar_ms), np.nan, dtype=float)
    if event_times.size:
        positions = np.searchsorted(event_times, bar_ms, side="right") - 1
        known = positions >= 0
        funding_at_bar[known] = event_rates[positions[known]]

    return SymbolData(
        symbol=symbol,
        source=source,
        bar_ms=bar_ms,
        open=frame["open"].to_numpy(dtype=float),
        high=frame["high"].to_numpy(dtype=float),
        low=frame["low"].to_numpy(dtype=float),
        close=frame["close"].to_numpy(dtype=float),
        atr=frame["atr"].to_numpy(dtype=float),
        signals=signals,
        funding_events=events,
        loop_start=loop_start,
        adx_series=frame["adx"].to_numpy(dtype=float),
        rsi_series=frame["rsi"].to_numpy(dtype=float),
        bb_upper=(bb_mid + 2.0 * bb_std).to_numpy(dtype=float),
        bb_lower=(bb_mid - 2.0 * bb_std).to_numpy(dtype=float),
        bb_mid=bb_mid.to_numpy(dtype=float),
        funding_rate_at_bar=funding_at_bar,
    )


_CACHE_DIR = Path(__file__).resolve().parent / ".backtest_cache"


def load_symbol_data_cached(symbol: str, granularity: str, limit: int,
                            months: int, use_cache: bool = True) -> SymbolData:
    """
    Cache prepared inputs per UTC day.

    The requested window is relative to "now", so a cache entry is only valid
    within the same UTC day. Anything older is refetched rather than reused.
    """
    if not use_cache:
        return load_symbol_data(symbol, granularity, limit, months)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    # The key carries every input that changes which signals are produced.
    # Without the entry-rule signature, raising ADX_ENTRY_THRESHOLD would
    # silently reuse signals generated under the old threshold.
    rules = (
        f"adx{ADX_ENTRY_THRESHOLD:g}_dc{DONCHIAN_PERIOD}"
        f"_ema{EMA_FAST_PERIOD}-{EMA_MID_PERIOD}-{EMA_SLOW_PERIOD}"
    )
    path = (
        _CACHE_DIR
        / f"{symbol}_{granularity}_{months}m_{limit}_v4_{rules}_{stamp}.pkl"
    )
    if path.exists():
        try:
            with path.open("rb") as handle:
                data = pickle.load(handle)
            if not isinstance(data, SymbolData):
                raise TypeError("cached object has an unexpected type")
            # An older pickle restores its own field set, so a missing attribute
            # would surface as a silent None deep inside the simulation.
            missing = [
                name for name in SymbolData.__dataclass_fields__
                if not hasattr(data, name)
            ]
            if missing:
                raise TypeError(f"cached object lacks {missing}")
            data.trade_start = 0
            return data
        except Exception:
            path.unlink(missing_ok=True)

    data = load_symbol_data(symbol, granularity, limit, months)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            # The object is pickled whole rather than as a dict, so the nested
            # SignalFeatures instances survive the round trip.
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return data


def build_timeline(datasets: list[SymbolData], end_ms: int | None = None,
                   start_ms: int | None = None) -> list[tuple]:
    """
    Common evaluation window across symbols, ordered by bar time then symbol.

    `start_ms` and `end_ms` carve a sub-window out of one loaded dataset, so a
    search block and a validation block can be cut from the same history
    without refetching it.
    """
    portfolio_start_ms = max(
        int(data.bar_ms[data.loop_start]) for data in datasets
    )
    if start_ms is not None:
        portfolio_start_ms = max(portfolio_start_ms, int(start_ms))
    timeline = []
    for order, data in enumerate(datasets):
        start = int(np.searchsorted(data.bar_ms, portfolio_start_ms, side="left"))
        data.trade_start = max(start, data.loop_start)
        for index in range(data.trade_start, len(data.bar_ms)):
            bar_ms = int(data.bar_ms[index])
            if end_ms is not None and bar_ms > end_ms:
                break
            timeline.append((bar_ms, order, index))
    timeline.sort()
    return timeline


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(datasets: list[SymbolData], timeline: list[tuple],
             params: SimParams) -> dict:
    cash = float(params.start_equity)
    open_trades: dict[int, dict] = {}
    pending: dict[int, dict] = {}
    pending_add: set = set()
    pending_partial: set = set()
    pyramid_adds = 0
    partial_takes = 0
    funding_index = [0] * len(datasets)
    last_close = [float("nan")] * len(datasets)

    closed_trades: list[dict] = []
    curve_ms: list[int] = []
    curve_equity: list[float] = []
    daily_pnl: dict[int, float] = {}
    skipped_no_slot = 0
    skipped_no_size = 0
    skipped_day_stop = 0
    skipped_filtered = 0
    peak_gross_exposure = 0.0
    peak_symbol_exposure = 0.0
    ruined = False
    skipped_ml_gate = 0

    # ML-Gate: precompute per-symbol regime predictions keyed by bar_ms.
    # This is a read-only lookup so the hot loop stays untouched when
    # ml_gate_enabled is False.
    ml_predictions: dict[int, dict[int, tuple]] = {}  # order → {bar_ms → (regime, confidence)}
    if params.ml_gate_enabled:
        try:
            from ml_trainer import WalkForwardTrainer
            from backtest import _load_ohlcv
            from technical_analysis import closed_candles as _cc, prepare_indicators as _pi

            trainer = WalkForwardTrainer(model_dir=params.ml_model_dir)
            for order, data in enumerate(datasets):
                try:
                    raw, _ = _load_ohlcv(
                        data.symbol, "1H", 1000,
                        max(1, int(len(data.bar_ms) / (30.44 * 24) + 1)),
                    )
                    frame = _cc(raw, "1H")
                    frame = _pi(frame)
                    preds = trainer.historical_predictions(
                        data.symbol, frame, [],
                    )
                    lookup: dict[int, tuple] = {}
                    for ts_str, regime, confidence in preds:
                        try:
                            ts_ms = int(
                                pd.Timestamp(ts_str).tz_localize(None)
                                .asm8.astype("datetime64[ms]").astype("int64")
                            )
                        except Exception:
                            continue
                        lookup[ts_ms] = (regime, confidence)
                    ml_predictions[order] = lookup
                except Exception as exc:
                    import logging as _log
                    _log.getLogger(__name__).warning(
                        f"ML-Gate: Keine Vorhersagen fuer {data.symbol}: {exc}"
                    )
        except ImportError:
            import logging as _log
            _log.getLogger(__name__).warning(
                "ML-Gate: ml_trainer nicht verfuegbar, Gate deaktiviert"
            )
            # Fall through — ml_predictions stays empty so no entry is blocked.

    def used_margin() -> float:
        return sum(
            trade["size_usdt"] for trade in open_trades.values()
        ) / params.leverage

    def portfolio_equity() -> float:
        equity = cash
        for order, trade in open_trades.items():
            close_price = last_close[order]
            if close_price == close_price:  # not NaN
                equity += _unrealised_price_fee_pnl(trade, close_price, params)
        return equity

    def available_cap(order: int | None = None) -> float:
        """Notional ceiling from the configured cap, margin and exposure limits."""
        if params.notional_cap_pct is not None:
            cap = cash * params.notional_cap_pct
        else:
            cap = float(params.notional_cap_usdt or 0.0)
        if params.apply_margin_ceiling:
            available = cash - used_margin()
            cap = min(cap, max(0.0, available * params.leverage * 0.90))
        if params.max_gross_exposure is not None:
            gross = sum(item["size_usdt"] for item in open_trades.values())
            cap = min(cap, cash * params.max_gross_exposure - gross)
        if params.max_symbol_exposure is not None and order is not None:
            held = open_trades[order]["size_usdt"] if order in open_trades else 0.0
            cap = min(cap, cash * params.max_symbol_exposure - held)
        return max(0.0, cap)

    current_ms = None
    for bar_ms, order, index in timeline:
        if ruined:
            break
        if current_ms is None:
            current_ms = bar_ms
        elif bar_ms != current_ms:
            equity_now = portfolio_equity()
            curve_ms.append(current_ms)
            curve_equity.append(equity_now)
            current_ms = bar_ms
            # Gross notional over equity. Pyramiding raises this because a
            # trailed-up stop shrinks the stop distance and therefore inflates
            # the risk-based size of every further add.
            if equity_now > 0 and open_trades:
                gross = sum(item["size_usdt"] for item in open_trades.values())
                peak_gross_exposure = max(peak_gross_exposure, gross / equity_now)
                largest = max(item["size_usdt"] for item in open_trades.values())
                peak_symbol_exposure = max(
                    peak_symbol_exposure, largest / equity_now
                )

        data = datasets[order]
        bar_open = data.open[index]
        bar_high = data.high[index]
        bar_low = data.low[index]
        bar_close = data.close[index]
        bar_atr = data.atr[index]
        last_close[order] = bar_close

        trade = open_trades.get(order)

        # 1. Funding settles against the position held before this open.
        events = data.funding_events
        cursor = funding_index[order]
        while cursor < len(events) and events[cursor]["funding_time"] <= bar_ms:
            event = events[cursor]
            cursor += 1
            if trade is None:
                continue
            mark_price = event.get("mark_price")
            if mark_price is None or float(mark_price) <= 0:
                mark_price = bar_open
            amount = funding_payment(
                trade["side"], trade["quantity"], mark_price, event["funding_rate"]
            )
            trade["funding"] = float(trade.get("funding") or 0.0) + amount
            cash += amount
            day_key = event["funding_time"] // 86_400_000
            daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + amount
        funding_index[order] = cursor

        # 2. A signal from the previous close is executed at this open.
        if trade is None and order in pending:
            candidate = pending.pop(order)
            day = bar_ms // 86_400_000
            if params.daily_loss_limit_usdt is not None:
                day_limit = params.daily_loss_limit_usdt
            elif params.daily_loss_limit_pct is not None:
                day_limit = max(0.0, cash) * params.daily_loss_limit_pct
            else:
                day_limit = None
            if day_limit is not None and daily_pnl.get(day, 0.0) <= -day_limit:
                skipped_day_stop += 1
            elif len(open_trades) >= params.max_open_positions:
                skipped_no_slot += 1
            else:
                entry_price = _adverse_fill(
                    bar_open, candidate["side"], True, params.slippage_rate
                )
                stop_loss = _stop_from_atr(
                    candidate["side"], entry_price, candidate["atr"],
                    candidate["stop_atr"],
                )
                if params.fixed_notional_usdt is not None:
                    notional = params.fixed_notional_usdt
                else:
                    notional = _size_notional(
                        cash, entry_price, stop_loss, available_cap(order), params,
                        risk_scale=candidate.get("ml_risk_scale", 1.0),
                    )
                if notional <= 0:
                    skipped_no_size += 1
                else:
                    trade = {
                        "symbol": data.symbol,
                        "side": candidate["side"],
                        "entry": entry_price,
                        "quantity": notional / entry_price,
                        "size_usdt": notional,
                        "entry_price": entry_price,
                        "sl": stop_loss,
                        "stop_loss": stop_loss,
                        "initial_stop_loss": stop_loss,
                        "best_price": entry_price,
                        "entry_atr": candidate["atr"],
                        "entry_fee": notional * params.fee_rate,
                        "funding": 0.0,
                        "entry_ms": bar_ms,
                        "entry_idx": index,
                        "adds_done": 0,
                        "exit_mode": candidate["exit_mode"],
                        "target": candidate["target"],
                        "time_stop_bars": candidate["time_stop_bars"],
                    }
                    open_trades[order] = trade

        # 2b. A pyramid trigger from the previous close is also filled at this
        # open. `entry` becomes the volume-weighted average so PnL stays exact,
        # while `entry_price`/`initial_stop_loss` keep the original values the
        # trailing logic is defined against.
        if order in pending_add:
            pending_add.discard(order)
            if (trade is not None and index > trade["entry_idx"]
                    and trade["adds_done"] < params.pyramid_max_adds):
                add_entry = _adverse_fill(
                    bar_open, trade["side"], True, params.slippage_rate
                )
                current_stop = float(trade["stop_loss"])
                stop_is_valid = (
                    current_stop < add_entry if trade["side"] == "long"
                    else current_stop > add_entry
                )
                if stop_is_valid:
                    add_notional = _size_notional(
                        cash, add_entry, current_stop, available_cap(order), params,
                        risk_scale=params.pyramid_risk_fraction,
                    )
                    if add_notional > 0:
                        add_quantity = add_notional / add_entry
                        total_quantity = trade["quantity"] + add_quantity
                        trade["entry"] = (
                            trade["entry"] * trade["quantity"]
                            + add_entry * add_quantity
                        ) / total_quantity
                        trade["quantity"] = total_quantity
                        trade["size_usdt"] += add_notional
                        trade["entry_fee"] += add_notional * params.fee_rate
                        trade["adds_done"] += 1
                        pyramid_adds += 1

        # 2c. A partial take triggered on the previous close is also filled at
        # this open, before the stop is examined.
        if order in pending_partial:
            pending_partial.discard(order)
            if trade is not None and not trade.get("partial_done"):
                fraction = min(1.0, max(0.0, params.partial_take_fraction))
                close_quantity = trade["quantity"] * fraction
                if close_quantity > 0:
                    exit_price = _adverse_fill(
                        bar_open, trade["side"], False, params.slippage_rate
                    )
                    direction = 1.0 if trade["side"] == "long" else -1.0
                    gross = (
                        (exit_price - trade["entry"]) * close_quantity * direction
                    )
                    exit_fee = exit_price * close_quantity * params.fee_rate
                    entry_fee_share = trade["entry_fee"] * fraction
                    partial_pnl = gross - entry_fee_share - exit_fee

                    trade["entry_fee"] -= entry_fee_share
                    trade["quantity"] -= close_quantity
                    trade["size_usdt"] *= (1.0 - fraction)
                    trade["realized_partial"] = (
                        float(trade.get("realized_partial") or 0.0) + partial_pnl
                    )
                    trade["partial_fees"] = (
                        float(trade.get("partial_fees") or 0.0)
                        + entry_fee_share + exit_fee
                    )
                    trade["partial_done"] = True
                    cash += partial_pnl
                    partial_day = bar_ms // 86_400_000
                    daily_pnl[partial_day] = (
                        daily_pnl.get(partial_day, 0.0) + partial_pnl
                    )
                    partial_takes += 1
                    if trade["quantity"] <= 0:
                        closed_trades.append(
                            _close_position(trade, bar_open, "partial", index, params)
                        )
                        del open_trades[order]
                        pending_add.discard(order)
                        trade = None

        # 3. Stop first, then trail on the same bar.
        if trade is not None:
            stop = float(trade["sl"])
            raw_exit = None
            if trade["side"] == "long":
                if bar_open <= stop:
                    raw_exit = bar_open
                elif bar_low <= stop:
                    raw_exit = stop
            else:
                if bar_open >= stop:
                    raw_exit = bar_open
                elif bar_high >= stop:
                    raw_exit = stop

            exit_reason = "sl"
            # Target and time stop for the non-trend sleeves. The stop is
            # examined first, so a bar that could have hit both is scored as the
            # loss. That is the conservative assumption on bar data.
            if raw_exit is None and trade.get("exit_mode") == "target":
                target = trade.get("target")
                if target is not None:
                    target = float(target)
                    if trade["side"] == "long":
                        if bar_open >= target:
                            raw_exit, exit_reason = bar_open, "target"
                        elif bar_high >= target:
                            raw_exit, exit_reason = target, "target"
                    else:
                        if bar_open <= target:
                            raw_exit, exit_reason = bar_open, "target"
                        elif bar_low <= target:
                            raw_exit, exit_reason = target, "target"
                bars_held = index - trade["entry_idx"]
                time_stop = int(trade.get("time_stop_bars") or 0)
                if raw_exit is None and time_stop > 0 and bars_held >= time_stop:
                    raw_exit, exit_reason = bar_close, "time"

            if raw_exit is not None:
                completed = _close_position(
                    trade, raw_exit, exit_reason, index, params
                )
                completed["exit_ms"] = bar_ms
                closed_trades.append(completed)
                cash += completed["price_fee_pnl"]
                close_day = bar_ms // 86_400_000
                daily_pnl[close_day] = (
                    daily_pnl.get(close_day, 0.0) + completed["price_fee_pnl"]
                )
                del open_trades[order]
                pending_add.discard(order)
                pending_partial.discard(order)
                trade = None
                if cash <= 0:
                    ruined = True
            elif (trade.get("exit_mode") == "trail"
                  and bar_atr == bar_atr and bar_atr > 0):
                trail_high = bar_high
                trail_low = bar_low
                if index == trade["entry_idx"]:
                    trail_high = max(trade["entry"], bar_close)
                    trail_low = min(trade["entry"], bar_close)
                new_stop, best_price, _ = _next_trailing_stop(
                    trade,
                    candle_close=bar_close,
                    candle_high=trail_high,
                    candle_low=trail_low,
                    atr=float(bar_atr),
                    params=params,
                )
                trade["sl"] = new_stop
                trade["stop_loss"] = new_stop
                trade["best_price"] = best_price

        # 3c. Partial-take trigger, judged on the closed bar like every other
        # decision, and measured against the original entry risk.
        if (params.partial_take_at_r > 0 and params.partial_take_fraction > 0
                and order in open_trades):
            live = open_trades[order]
            if not live.get("partial_done"):
                base_entry = float(live["entry_price"])
                base_risk = abs(base_entry - float(live["initial_stop_loss"]))
                if base_risk > 0:
                    excursion = (
                        (bar_close - base_entry) / base_risk
                        if live["side"] == "long"
                        else (base_entry - bar_close) / base_risk
                    )
                    if excursion >= params.partial_take_at_r:
                        pending_partial.add(order)

        # 3b. Pyramid trigger is judged on the closed bar and filled next open,
        # so it never uses a price the live bot could not have acted on.
        if params.pyramid_max_adds > 0 and order in open_trades:
            live = open_trades[order]
            if live["adds_done"] < params.pyramid_max_adds:
                base_entry = float(live["entry_price"])
                base_risk = abs(base_entry - float(live["initial_stop_loss"]))
                if base_risk > 0:
                    excursion = (
                        (bar_close - base_entry) / base_risk
                        if live["side"] == "long"
                        else (base_entry - bar_close) / base_risk
                    )
                    if excursion >= (live["adds_done"] + 1) * params.pyramid_step_r:
                        pending_add.add(order)

        # 4. A fresh setup at this close may only be entered at the next open.
        if order not in open_trades:
            candidate = _entry_candidate(data, index, bar_close, params)
            if candidate is not None:
                if candidate.get("rejected"):
                    skipped_filtered += 1
                else:
                    # ML-Gate filter: block entries in predicted choppy regimes.
                    if params.ml_gate_enabled and order in ml_predictions:
                        pred = ml_predictions[order].get(int(data.bar_ms[index]))
                        if pred is not None:
                            ml_regime, ml_conf = pred
                            if (ml_regime == "choppy"
                                    and ml_conf < params.ml_confidence_threshold):
                                skipped_ml_gate += 1
                                continue
                            # Scale risk by confidence.
                            candidate["ml_risk_scale"] = (
                                params.ml_min_risk_scale
                                + (1.0 - params.ml_min_risk_scale) * ml_conf
                            )
                    pending[order] = candidate

    if current_ms is not None:
        curve_ms.append(current_ms)
        curve_equity.append(portfolio_equity())

    return {
        "params": params,
        "closed_trades": closed_trades,
        "open_trades": len(open_trades),
        "curve_ms": np.asarray(curve_ms, dtype="int64"),
        "curve_equity": np.asarray(curve_equity, dtype=float),
        "skipped_no_slot": skipped_no_slot,
        "skipped_no_size": skipped_no_size,
        "skipped_day_stop": skipped_day_stop,
        "skipped_filtered": skipped_filtered,
        "skipped_ml_gate": skipped_ml_gate,
        "pyramid_adds": pyramid_adds,
        "partial_takes": partial_takes,
        "peak_gross_exposure": peak_gross_exposure,
        "peak_symbol_exposure": peak_symbol_exposure,
        "ruined": ruined,
    }


# ── Percentage performance metrics ────────────────────────────────────────────

def performance(result: dict) -> dict:
    params = result["params"]
    start_equity = float(params.start_equity)
    equity = result["curve_equity"]
    trades = result["closed_trades"]

    if equity.size == 0:
        return {"error": "no simulated bars"}

    series = pd.Series(
        equity, index=pd.to_datetime(result["curve_ms"], unit="ms", utc=True)
    )
    month_end = series.resample("ME").last()
    previous = month_end.shift(1)
    if not previous.empty:
        previous.iloc[0] = start_equity
    monthly_pct = ((month_end / previous) - 1.0) * 100.0
    monthly_pct = monthly_pct.replace([np.inf, -np.inf], np.nan).dropna()

    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
    max_drawdown_pct = float(np.nanmax(drawdown) * 100.0) if drawdown.size else 0.0

    final_equity = float(equity[-1])
    months = int(monthly_pct.size)
    span_days = max(
        1e-9,
        (int(result["curve_ms"][-1]) - int(result["curve_ms"][0])) / 86_400_000.0,
    )

    if final_equity > 0 and start_equity > 0:
        growth = final_equity / start_equity
        geometric_monthly_pct = (growth ** (1.0 / months) - 1.0) * 100.0 if months else 0.0
        cagr_pct = (growth ** (365.25 / span_days) - 1.0) * 100.0
    else:
        geometric_monthly_pct = float("nan")
        cagr_pct = float("nan")

    wins = [item for item in trades if item["pnl"] > 0]
    losses = [item for item in trades if item["pnl"] <= 0]
    gross_profit = sum(item["pnl"] for item in wins)
    gross_loss = abs(sum(item["pnl"] for item in losses))

    # Accounting identity. With no position left open, the equity curve must end
    # exactly at start + the sum of realised results. Any residual means a leg
    # was double counted or dropped, which is how a partial-take bug hides.
    accounting_residual = float("nan")
    if result["open_trades"] == 0 and not result["ruined"]:
        accounting_residual = (
            final_equity - start_equity - sum(item["pnl"] for item in trades)
        )

    worst_streak = streak = 0
    for value in monthly_pct:
        streak = streak + 1 if value < 0 else 0
        worst_streak = max(worst_streak, streak)

    # Consistency over short horizons. A high mean built from two months is a
    # different product than the same mean spread evenly, so the rolling
    # windows and the recovery time matter as much as the average.
    factors = (monthly_pct / 100.0 + 1.0).to_numpy()

    def rolling_returns(window: int) -> np.ndarray:
        if factors.size < window:
            return np.asarray([], dtype=float)
        return np.asarray([
            factors[index:index + window].prod() - 1.0
            for index in range(factors.size - window + 1)
        ], dtype=float)

    rolling = {}
    for window in (3, 6, 12):
        values = rolling_returns(window)
        rolling[window] = {
            "worst_pct": float(values.min() * 100.0) if values.size else float("nan"),
            "best_pct": float(values.max() * 100.0) if values.size else float("nan"),
            "positive_share_pct": (
                float((values > 0).mean() * 100.0) if values.size else float("nan")
            ),
            "windows": int(values.size),
        }

    # Months spent below a previous monthly-equity peak, i.e. how long a bad
    # patch can last before the account sees a new high.
    monthly_equity = month_end.to_numpy(dtype=float)
    longest_underwater = current_underwater = 0
    peak_equity = start_equity
    for value in monthly_equity:
        if value >= peak_equity:
            peak_equity = value
            current_underwater = 0
        else:
            current_underwater += 1
            longest_underwater = max(longest_underwater, current_underwater)

    downside = monthly_pct[monthly_pct < 0]
    downside_deviation = float(np.sqrt((downside ** 2).mean())) if downside.size else 0.0
    monthly_sortino = (
        float(monthly_pct.mean() / downside_deviation)
        if downside_deviation > 0 else float("nan")
    )

    return {
        "start_equity": start_equity,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / start_equity - 1.0) * 100.0,
        "geometric_monthly_pct": geometric_monthly_pct,
        "mean_monthly_pct": float(monthly_pct.mean()) if months else 0.0,
        "median_monthly_pct": float(monthly_pct.median()) if months else 0.0,
        "cagr_pct": cagr_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "months": months,
        "positive_months": int((monthly_pct > 0).sum()),
        "worst_month_pct": float(monthly_pct.min()) if months else 0.0,
        "best_month_pct": float(monthly_pct.max()) if months else 0.0,
        "worst_losing_streak_months": worst_streak,
        "positive_months_pct": (
            float((monthly_pct > 0).mean() * 100.0) if months else 0.0
        ),
        "rolling": rolling,
        "longest_underwater_months": longest_underwater,
        "monthly_sortino": monthly_sortino,
        "trades": len(trades),
        "win_rate_pct": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "net_pnl": sum(item["pnl"] for item in trades),
        "funding": sum(item["funding"] for item in trades),
        "fees": sum(item["fees"] for item in trades),
        "accounting_residual": accounting_residual,
        "skipped_no_slot": result["skipped_no_slot"],
        "skipped_no_size": result["skipped_no_size"],
        "skipped_day_stop": result["skipped_day_stop"],
        "skipped_filtered": result["skipped_filtered"],
        "skipped_ml_gate": result.get("skipped_ml_gate", 0),
        "pyramid_adds": result["pyramid_adds"],
        "partial_takes": result["partial_takes"],
        "peak_gross_exposure": result["peak_gross_exposure"],
        "peak_symbol_exposure": result["peak_symbol_exposure"],
        "still_open": result["open_trades"],
        "ruined": result["ruined"],
        "monthly_pct": monthly_pct,
    }


def print_report(stats: dict, params: SimParams, show_months: bool):
    if "error" in stats:
        print(f"  {stats['error']}")
        return
    bar = "-" * 74
    print(f"\n{bar}")
    print(f"  {params.label()}")
    print(bar)
    if stats["ruined"]:
        print("  ACCOUNT RUINED: equity reached zero; simulation stopped early.")
    residual = stats.get("accounting_residual")
    if residual is not None and residual == residual and abs(residual) > 0.01:
        print(
            f"  !! ACCOUNTING MISMATCH: equity curve and realised results differ "
            f"by {residual:+.4f} USDT. Treat these numbers as invalid."
        )
    print(
        f"  Equity        : {stats['start_equity']:.2f} -> "
        f"{stats['final_equity']:.2f} USDT "
        f"({stats['total_return_pct']:+.1f}%)"
    )
    print(
        f"  Monthly return: {stats['geometric_monthly_pct']:+.2f}% compounded | "
        f"mean {stats['mean_monthly_pct']:+.2f}% | "
        f"median {stats['median_monthly_pct']:+.2f}%"
    )
    print(f"  CAGR          : {stats['cagr_pct']:+.1f}%")
    print(f"  Max drawdown  : {stats['max_drawdown_pct']:.1f}%")
    print(
        f"  Months        : {stats['positive_months']}/{stats['months']} positive "
        f"({stats['positive_months_pct']:.0f}%) | "
        f"worst {stats['worst_month_pct']:+.1f}% | best {stats['best_month_pct']:+.1f}% | "
        f"longest losing streak {stats['worst_losing_streak_months']}"
    )
    for window in (3, 6, 12):
        info = stats["rolling"].get(window) or {}
        if not info.get("windows"):
            continue
        print(
            f"  Rolling {window:>2}m     : {info['positive_share_pct']:.0f}% positive | "
            f"worst {info['worst_pct']:+.1f}% | best {info['best_pct']:+.1f}%"
        )
    print(
        f"  Underwater    : up to {stats['longest_underwater_months']} month(s) below "
        f"a previous peak | monthly Sortino {stats['monthly_sortino']:.2f}"
    )
    print(
        f"  Trades        : {stats['trades']} | WR {stats['win_rate_pct']:.1f}% | "
        f"PF {stats['profit_factor']:.3f} | net {stats['net_pnl']:+.2f} USDT"
        + (f" | {stats['pyramid_adds']} pyramid adds"
           if stats["pyramid_adds"] else "")
        + (f" | {stats['partial_takes']} partial takes"
           if stats["partial_takes"] else "")
    )
    print(
        f"  Costs         : fees {stats['fees']:.2f} | "
        f"funding {stats['funding']:+.2f} USDT"
    )
    print(
        f"  Peak exposure : {stats['peak_gross_exposure']:.2f}x equity gross | "
        f"{stats['peak_symbol_exposure']:.2f}x in a single symbol "
        f"(leverage cap {LEVERAGE}x)"
    )
    print(
        f"  Throttled     : {stats['skipped_no_slot']} no free slot | "
        f"{stats['skipped_no_size']} below size floor | "
        f"{stats['skipped_day_stop']} blocked by day-stop | "
        f"{stats['skipped_filtered']} filtered out | "
        f"{stats.get('skipped_ml_gate', 0)} ML-gate blocked | "
        f"{stats['still_open']} still open"
    )
    if show_months:
        print(f"  {'-' * 40}")
        for timestamp, value in stats["monthly_pct"].items():
            print(f"    {timestamp:%Y-%m}  {value:+7.2f}%")
    print(bar)


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.replace(" ", "").split(",") if part]


def _print_ml_comparison(base: dict, ml: dict) -> None:
    """Side-by-side performance comparison with and without ML gate."""
    bar = "=" * 74
    print(f"\n{bar}")
    print("  ML-GATE COMPARISON (Backtest-Ergebnis)")
    print(f"{bar}")
    print("  Hinweis: Vergangene Backtest-Ergebnisse garantieren keine")
    print("  zukuenftige Performance.")
    print()

    def _row(label: str, key: str, fmt: str = "+.2f", suffix: str = "") -> None:
        b = base.get(key, 0)
        m = ml.get(key, 0)
        b_s = f"{b:{fmt}}{suffix}" if b == b else "n/a"
        m_s = f"{m:{fmt}}{suffix}" if m == m else "n/a"
        print(f"  {label:<26s} {b_s:>14s} {m_s:>14s}")

    print(f"  {'Metrik':<26s} {'Ohne ML-Gate':>14s} {'Mit ML-Gate':>14s}")
    print(f"  {'-' * 26} {'-' * 14} {'-' * 14}")

    # Sharpe ratio is mean / std of monthly returns — computed inline because
    # the main performance() dict does not carry it.
    def _sharpe(stats: dict) -> float:
        mp = stats.get("monthly_pct")
        if mp is None or len(mp) < 2:
            return float("nan")
        std = float(mp.std())
        return float(mp.mean() / std) if std > 0 else float("nan")

    b_sharpe, m_sharpe = _sharpe(base), _sharpe(ml)
    print(
        f"  {'Sharpe (monthly)':<26s} {b_sharpe:>+14.2f} {m_sharpe:>+14.2f}"
    )
    _row("Sortino (monthly)", "monthly_sortino", "+.2f")
    _row("Max Drawdown", "max_drawdown_pct", ".1f", "%")
    _row("Profit Factor", "profit_factor", ".3f")
    _row("Total Return", "total_return_pct", "+.1f", "%")
    _row("Monthly Return (geo)", "geometric_monthly_pct", "+.2f", "%")
    _row("CAGR", "cagr_pct", "+.1f", "%")
    _row("Win Rate", "win_rate_pct", ".1f", "%")
    _row("Trades", "trades", "d")
    _row("Positive Months", "positive_months_pct", ".0f", "%")
    _row("Worst Month", "worst_month_pct", "+.1f", "%")
    _row("Underwater (months)", "longest_underwater_months", "d")

    # Monthly returns side by side if both available.
    b_months = base.get("monthly_pct")
    m_months = ml.get("monthly_pct")
    if b_months is not None and m_months is not None and len(b_months) > 0:
        print(f"\n  {'Month':<10s} {'Ohne ML':>10s} {'Mit ML':>10s} {'Delta':>10s}")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
        all_months = sorted(set(b_months.index) | set(m_months.index))
        for ts in all_months:
            bv = float(b_months.get(ts, float("nan")))
            mv = float(m_months.get(ts, float("nan")))
            delta = mv - bv if bv == bv and mv == mv else float("nan")
            b_s = f"{bv:+.2f}%" if bv == bv else "   n/a"
            m_s = f"{mv:+.2f}%" if mv == mv else "   n/a"
            d_s = f"{delta:+.2f}%" if delta == delta else "   n/a"
            print(f"  {ts:%Y-%m}     {b_s:>10s} {m_s:>10s} {d_s:>10s}")

    # Extra: ML gate stats.
    ml_blocked = ml.get("skipped_ml_gate", 0)
    print(f"\n  ML-Gate blocked entries: {ml_blocked}")
    print(bar)


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio backtest with a shared compounding equity curve"
    )
    parser.add_argument("symbols", nargs="*", help="Symbols (default: research universe)")
    parser.add_argument("--granularity", "-g", default=STRATEGY_TIMEFRAME)
    parser.add_argument("--months", "-m", type=int, default=24)
    parser.add_argument("--limit", "-l", type=int, default=1000)
    parser.add_argument("--equity", type=float, default=1000.0,
                        help="Starting equity in USDT (default 1000)")
    parser.add_argument("--risk", type=float, default=RISK_PER_TRADE_PCT * 100,
                        help="Risk per trade in percent of equity")
    parser.add_argument("--max-positions", type=int, default=MAX_OPEN_POSITIONS)
    parser.add_argument("--cap-usdt", type=float, default=None,
                        help="Fixed notional cap in USDT (default: current config)")
    parser.add_argument("--cap-pct", type=float, default=None,
                        help="Notional cap as percent of equity (replaces --cap-usdt)")
    parser.add_argument("--min-notional", type=float, default=MIN_POSITION_USDT)
    parser.add_argument("--cost-multiplier", type=float, default=1.0,
                        help="Scale fees and slippage for sensitivity tests")
    parser.add_argument("--daily-limit-pct", type=float, default=None,
                        help="Daily loss stop as percent of equity (replaces the "
                             "absolute config limit)")
    parser.add_argument("--no-daily-limit", action="store_true",
                        help="Disable the daily loss stop entirely")
    parser.add_argument("--side", choices=["both", "long", "short"], default="both",
                        help="Restrict entries to one direction")
    parser.add_argument("--stop-atr", type=float, default=TREND_STOP_ATR_MULTIPLIER)
    parser.add_argument("--trail-atr", type=float, default=TREND_TRAIL_ATR_MULTIPLIER)
    parser.add_argument("--activation-r", type=float, default=TRAILING_ACTIVATION_R)
    parser.add_argument("--strategy", choices=["trend", "meanrev", "funding"],
                        default="trend", help="Which return source to test")
    parser.add_argument("--mr-max-adx", type=float, default=20.0)
    parser.add_argument("--mr-rsi-long", type=float, default=30.0)
    parser.add_argument("--mr-rsi-short", type=float, default=70.0)
    parser.add_argument("--mr-stop-atr", type=float, default=2.0)
    parser.add_argument("--mr-time-stop-bars", type=int, default=48)
    parser.add_argument("--sweep-mr-adx", type=str, default=None)
    parser.add_argument("--sweep-mr-rsi", type=str, default=None,
                        help="Long RSI thresholds; the short side mirrors them")
    parser.add_argument("--fund-short-rate", type=float, default=0.05,
                        help="Short when the settled funding rate exceeds this percent")
    parser.add_argument("--fund-long-rate", type=float, default=-0.03,
                        help="Long when the settled funding rate falls below this percent")
    parser.add_argument("--fund-hold-bars", type=int, default=24)
    parser.add_argument("--fund-stop-atr", type=float, default=2.5)
    parser.add_argument("--fund-target-atr", type=float, default=0.0)
    parser.add_argument("--fund-direction", choices=["contrarian", "momentum"],
                        default="contrarian")
    parser.add_argument("--sweep-fund-rate", type=str, default=None,
                        help="Short-side funding thresholds in percent")
    parser.add_argument("--sweep-fund-hold", type=str, default=None)
    parser.add_argument("--min-adx", type=float, default=0.0,
                        help="Require at least this ADX at entry")
    parser.add_argument("--sweep-adx", type=str, default=None,
                        help="Comma separated ADX minimums to sweep")
    parser.add_argument("--min-atr-pct", type=float, default=0.0,
                        help="Require ATR/price at entry to be at least this percent")
    parser.add_argument("--max-atr-pct", type=float, default=0.0,
                        help="Require ATR/price at entry to be at most this percent")
    parser.add_argument("--min-volume-ratio", type=float, default=0.0,
                        help="Require breakout volume to be this multiple of its SMA")
    parser.add_argument("--sweep-volume", type=str, default=None,
                        help="Comma separated volume ratios to sweep")
    parser.add_argument("--min-rsi", type=float, default=0.0)
    parser.add_argument("--max-rsi", type=float, default=100.0)
    parser.add_argument("--sweep-max-rsi", type=str, default=None,
                        help="Comma separated RSI ceilings to sweep")
    parser.add_argument("--min-macd-norm", type=float, default=None,
                        help="Require MACD histogram of at least this many ATR")
    parser.add_argument("--min-di-spread", type=float, default=0.0,
                        help="Require this much distance between +DI and -DI")
    parser.add_argument("--sweep-di", type=str, default=None,
                        help="Comma separated DI spreads to sweep")
    parser.add_argument("--sweep-trail", type=str, default=None,
                        help="Comma separated trailing ATR multipliers to sweep")
    parser.add_argument("--sweep-activation", type=str, default=None,
                        help="Comma separated trailing activation R values to sweep")
    parser.add_argument("--sweep-stop", type=str, default=None,
                        help="Comma separated initial stop ATR multipliers to sweep")
    parser.add_argument("--partial-take-r", type=float, default=0.0,
                        help="Realise part of the position at this R multiple")
    parser.add_argument("--partial-take-fraction", type=float, default=0.5,
                        help="Fraction realised at --partial-take-r")
    parser.add_argument("--sweep-partial-r", type=str, default=None,
                        help="Comma separated R multiples to sweep")
    parser.add_argument("--max-symbol-exposure", type=float, default=None,
                        help="Cap total notional per symbol at this multiple of equity")
    parser.add_argument("--max-gross-exposure", type=float, default=None,
                        help="Cap total notional across symbols at this multiple "
                             "of equity")
    parser.add_argument("--pyramid-adds", type=int, default=0,
                        help="Maximum add-ons per winning position (0 disables)")
    parser.add_argument("--pyramid-step-r", type=float, default=1.0,
                        help="Favourable excursion in R between add-ons")
    parser.add_argument("--pyramid-risk", type=float, default=1.0,
                        help="Add-on risk relative to the base risk per trade")
    parser.add_argument("--sweep-pyramid", type=str, default=None,
                        help="Comma separated add-on counts to sweep")
    parser.add_argument("--skip-oldest-months", type=float, default=0.0,
                        help="Cut this many months off the start of the window")
    parser.add_argument("--skip-recent-months", type=float, default=0.0,
                        help="Cut this many months off the end of the window, so "
                             "train and holdout periods can be separated")
    parser.add_argument("--allow-late", action="store_true",
                        help="Keep symbols whose history starts inside the window "
                             "(otherwise they are dropped to keep windows comparable)")
    parser.add_argument("--months-table", action="store_true",
                        help="Print every monthly return")
    parser.add_argument("--sweep-risk", type=str, default=None,
                        help="Comma separated risk percentages to sweep")
    parser.add_argument("--sweep-positions", type=str, default=None,
                        help="Comma separated max-open-position counts to sweep")
    parser.add_argument("--replicate", action="store_true",
                        help="Reproduce the backtest.py contract as an engine check")
    parser.add_argument("--no-cache", action="store_true",
                        help="Always refetch instead of reusing today's cache")
    parser.add_argument("--ml-gate", action="store_true",
                        help="Activate ML-Gate simulation: run with and without "
                             "ML regime filter and compare side by side")
    parser.add_argument("--ml-model-dir", type=str, default=".models/",
                        help="Directory containing trained ML models")
    parser.add_argument("--ml-confidence", type=float, default=0.5,
                        help="ML confidence threshold for blocking choppy entries")
    parser.add_argument("--ml-min-risk-scale", type=float, default=0.5,
                        help="Minimum risk scale applied at low ML confidence")
    args = parser.parse_args()

    symbols = args.symbols or get_top_symbols(8)
    if not symbols:
        print("No symbols available")
        sys.exit(1)

    print(f"\n{'=' * 74}")
    print("  CryptoBot portfolio backtest (shared compounding equity)")
    print(f"  Symbols   : {', '.join(symbols)}")
    print(f"  Timeframe : {args.granularity} | window: {args.months} months")
    print(f"{'=' * 74}")

    datasets = []
    for symbol in symbols:
        print(f"  Loading {symbol}...", end=" ", flush=True)
        try:
            data = load_symbol_data_cached(
                symbol, args.granularity, args.limit, args.months,
                use_cache=not args.no_cache,
            )
        except (DataUnavailable, FundingDataError) as exc:
            print(f"skipped ({exc})")
            continue
        entries = sum(1 for item in data.signals if item is not None)
        print(f"{len(data.bar_ms)} bars, {entries} entry signals")
        datasets.append(data)

    if not datasets:
        print("\nNo symbol could be loaded completely; aborting.")
        sys.exit(1)

    # A symbol listed inside the window would otherwise truncate the common
    # window for every other symbol, which silently changes what is compared.
    # The reference is the earliest symbol start, not the requested window start:
    # a long warmup shifts every symbol equally and must not drop the universe.
    if len(datasets) > 1 and not args.allow_late:
        reference = min(int(data.bar_ms[data.loop_start]) for data in datasets)
        tolerance = 7 * 86_400_000
        kept = []
        for data in datasets:
            first_tradable = int(data.bar_ms[data.loop_start])
            if first_tradable > reference + tolerance:
                begins = pd.to_datetime(first_tradable, unit="ms", utc=True)
                print(
                    f"  Dropping {data.symbol}: history only starts {begins:%Y-%m-%d} "
                    "(use --allow-late to keep it)"
                )
                continue
            kept.append(data)
        datasets = kept
        if not datasets:
            print("\nNo symbol covers the requested window; aborting.")
            sys.exit(1)

    latest_end = max(int(data.bar_ms[-1]) for data in datasets)
    end_ms = None
    if args.skip_recent_months > 0:
        end_ms = latest_end - int(args.skip_recent_months * 30.44 * 86_400_000)
    start_ms = None
    if args.skip_oldest_months > 0:
        earliest = max(int(data.bar_ms[data.loop_start]) for data in datasets)
        start_ms = earliest + int(args.skip_oldest_months * 30.44 * 86_400_000)

    timeline = build_timeline(datasets, end_ms=end_ms, start_ms=start_ms)
    if not timeline:
        print("\nThe requested window is empty; aborting.")
        sys.exit(1)
    span_start = pd.to_datetime(timeline[0][0], unit="ms", utc=True)
    span_end = pd.to_datetime(timeline[-1][0], unit="ms", utc=True)
    print(
        f"\n  Common window: {span_start:%Y-%m-%d} -> {span_end:%Y-%m-%d} "
        f"({len(datasets)} symbols, {len(timeline)} symbol-bars)"
    )

    cost_scale = max(0.0, args.cost_multiplier)
    if args.no_daily_limit:
        day_limit_usdt, day_limit_pct = None, None
    elif args.daily_limit_pct is not None:
        day_limit_usdt, day_limit_pct = None, args.daily_limit_pct / 100.0
    else:
        day_limit_usdt, day_limit_pct = DAILY_LOSS_LIMIT_USDT, None

    base = SimParams(
        start_equity=args.equity,
        risk_per_trade_pct=args.risk / 100.0,
        max_open_positions=args.max_positions,
        notional_cap_usdt=(
            args.cap_usdt if args.cap_usdt is not None
            else min(POSITION_SIZE_USDT, MAX_POSITION_USDT)
        ),
        notional_cap_pct=(args.cap_pct / 100.0 if args.cap_pct is not None else None),
        min_notional_usdt=args.min_notional,
        fee_rate=TAKER_FEE_RATE * cost_scale,
        slippage_rate=SLIPPAGE_RATE * cost_scale,
        daily_loss_limit_usdt=day_limit_usdt,
        daily_loss_limit_pct=day_limit_pct,
        allowed_sides=(
            ("long", "short") if args.side == "both" else (args.side,)
        ),
        stop_atr_multiplier=args.stop_atr,
        trail_atr_multiplier=args.trail_atr,
        trailing_activation_r=args.activation_r,
        pyramid_max_adds=args.pyramid_adds,
        pyramid_step_r=args.pyramid_step_r,
        pyramid_risk_fraction=args.pyramid_risk,
        max_symbol_exposure=args.max_symbol_exposure,
        max_gross_exposure=args.max_gross_exposure,
        partial_take_at_r=args.partial_take_r,
        partial_take_fraction=args.partial_take_fraction,
        min_adx=args.min_adx,
        min_atr_pct=args.min_atr_pct / 100.0,
        max_atr_pct=args.max_atr_pct / 100.0,
        min_volume_ratio=args.min_volume_ratio,
        min_rsi=args.min_rsi,
        max_rsi=args.max_rsi,
        min_macd_norm=(
            args.min_macd_norm if args.min_macd_norm is not None else float("-inf")
        ),
        min_di_spread=args.min_di_spread,
        strategy=args.strategy,
        mr_max_adx=args.mr_max_adx,
        mr_rsi_long=args.mr_rsi_long,
        mr_rsi_short=args.mr_rsi_short,
        mr_stop_atr=args.mr_stop_atr,
        mr_time_stop_bars=args.mr_time_stop_bars,
        fund_short_rate=args.fund_short_rate / 100.0,
        fund_long_rate=args.fund_long_rate / 100.0,
        fund_hold_bars=args.fund_hold_bars,
        fund_stop_atr=args.fund_stop_atr,
        fund_target_atr=args.fund_target_atr,
        fund_direction=args.fund_direction,
        # ML-Gate fields are NOT set here — they are only injected via
        # dataclasses.replace() when --ml-gate is active, so the base run
        # stays identical to the pre-ML behaviour.
    )

    if args.replicate:
        check = SimParams(
            start_equity=args.equity,
            max_open_positions=len(datasets),
            min_notional_usdt=0.0,
            fixed_notional_usdt=POSITION_SIZE_USDT,
            apply_margin_ceiling=False,
            fee_rate=base.fee_rate,
            slippage_rate=base.slippage_rate,
        )
        stats = performance(simulate(datasets, timeline, check))
        print("\n  ENGINE CHECK vs backtest.py contract "
              "(fixed notional, one slot per symbol, no margin ceiling)")
        print_report(stats, check, args.months_table)
        return

    # ── ML-Gate comparison run ────────────────────────────────────────────
    if args.ml_gate:
        print("\n  ML-Gate comparison: running WITHOUT and WITH ML gate...")
        # 1. Baseline run (no ML gate).
        stats_base = performance(simulate(datasets, timeline, base))
        print("\n  ──── WITHOUT ML-Gate ────")
        print_report(stats_base, base, args.months_table)

        # 2. ML-gated run.
        ml_params = replace(
            base,
            ml_gate_enabled=True,
            ml_model_dir=args.ml_model_dir,
            ml_confidence_threshold=args.ml_confidence,
            ml_min_risk_scale=args.ml_min_risk_scale,
        )
        stats_ml = performance(simulate(datasets, timeline, ml_params))
        print("\n  ──── WITH ML-Gate ────")
        print_report(stats_ml, ml_params, args.months_table)

        # 3. Side-by-side comparison.
        _print_ml_comparison(stats_base, stats_ml)
        return

    # Only the axes that were explicitly requested enter the grid, which keeps
    # the multiple-testing surface visible instead of hidden in nested loops.
    axes: dict[str, list] = {}
    if args.sweep_risk:
        axes["risk_per_trade_pct"] = [
            value / 100.0 for value in _parse_float_list(args.sweep_risk)
        ]
    if args.sweep_positions:
        axes["max_open_positions"] = [
            int(value) for value in _parse_float_list(args.sweep_positions)
        ]
    if args.sweep_pyramid:
        axes["pyramid_max_adds"] = [
            int(value) for value in _parse_float_list(args.sweep_pyramid)
        ]
    if args.sweep_partial_r:
        axes["partial_take_at_r"] = _parse_float_list(args.sweep_partial_r)
    if args.sweep_adx:
        axes["min_adx"] = _parse_float_list(args.sweep_adx)
    if args.sweep_volume:
        axes["min_volume_ratio"] = _parse_float_list(args.sweep_volume)
    if args.sweep_max_rsi:
        axes["max_rsi"] = _parse_float_list(args.sweep_max_rsi)
    if args.sweep_di:
        axes["min_di_spread"] = _parse_float_list(args.sweep_di)
    if args.sweep_trail:
        axes["trail_atr_multiplier"] = _parse_float_list(args.sweep_trail)
    if args.sweep_activation:
        axes["trailing_activation_r"] = _parse_float_list(args.sweep_activation)
    if args.sweep_stop:
        axes["stop_atr_multiplier"] = _parse_float_list(args.sweep_stop)
    if args.sweep_mr_adx:
        axes["mr_max_adx"] = _parse_float_list(args.sweep_mr_adx)
    if args.sweep_mr_rsi:
        axes["mr_rsi_long"] = _parse_float_list(args.sweep_mr_rsi)
    if args.sweep_fund_rate:
        axes["fund_short_rate"] = [
            value / 100.0 for value in _parse_float_list(args.sweep_fund_rate)
        ]
    if args.sweep_fund_hold:
        axes["fund_hold_bars"] = [
            int(value) for value in _parse_float_list(args.sweep_fund_hold)
        ]

    names = list(axes)
    combinations = (
        [dict(zip(names, values))
         for values in product(*(axes[name] for name in names))]
        if names else [{}]
    )

    rows = []
    for overrides in combinations:
        params = replace(base, **overrides)
        # Keep the reversion thresholds symmetric when only the long side is swept.
        if "mr_rsi_long" in overrides:
            params = replace(params, mr_rsi_short=100.0 - params.mr_rsi_long)
        stats = performance(simulate(datasets, timeline, params))
        print_report(stats, params, args.months_table)
        rows.append((overrides, params, stats))

    if len(rows) > 1:
        print(f"\n{'=' * 74}")
        print(f"  SWEEP SUMMARY | {len(rows)} configurations")
        print(f"{'=' * 74}")
        axis_header = " ".join(f"{name.split('_')[0][:7]:>7}" for name in names)
        print(
            f"  {axis_header} {'monthly':>8} {'maxDD':>7} {'pos.mo':>7} "
            f"{'w3m':>7} {'w6m':>7} {'uw':>3} {'sort':>5} {'PF':>6} {'trades':>7}"
        )
        for overrides, params, stats in rows:
            if "error" in stats:
                continue
            three = stats["rolling"].get(3) or {}
            six = stats["rolling"].get(6) or {}
            axis_values = " ".join(
                f"{overrides[name]:>7g}" for name in names
            )
            print(
                f"  {axis_values} "
                f"{stats['geometric_monthly_pct']:+7.2f}% "
                f"{stats['max_drawdown_pct']:6.1f}% "
                f"{stats['positive_months_pct']:6.0f}% "
                f"{three.get('worst_pct', float('nan')):+6.1f}% "
                f"{six.get('worst_pct', float('nan')):+6.1f}% "
                f"{stats['longest_underwater_months']:3d} "
                f"{stats['monthly_sortino']:5.2f} "
                f"{stats['profit_factor']:6.3f} "
                f"{stats['trades']:7d}"
                + ("  RUINED" if stats["ruined"] else "")
            )
        print("  w3m/w6m = worst rolling 3- and 6-month return, "
              "uw = longest months below a previous peak")
        print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
