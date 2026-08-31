#!/usr/bin/env python3
"""
Causal, cost-aware backtest for the exact live trend-breakout core.

Signals are calculated on a completed candle and entered at the next candle open.
Funding settles against the position held before that bar open, before stops or a
new entry at the same timestamp. Stops are checked before a closed candle can
update the following candle's trail. Fees and adverse slippage apply both ways.
"""

import argparse
import sys
import time as _time

import pandas as pd
import requests

from bitget_client import get_top_symbols
from config import (
    DONCHIAN_PERIOD,
    EMA_HISTORY_CANDLES,
    POSITION_SIZE_USDT,
    SLIPPAGE_RATE,
    STRATEGY_TIMEFRAME,
    TAKER_FEE_RATE,
)
from risk_manager import best_sl_tp, next_atr_trailing_stop
from technical_analysis import (
    closed_candles,
    evaluate_signal,
    fetch_ohlcv,
    prepare_indicators,
)

WARMUP = max(EMA_HISTORY_CANDLES, DONCHIAN_PERIOD)
_BINANCE_FUTURES_BASE = "https://fapi.binance.com"
_BINANCE_INTERVAL_MAP = {"15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d"}


class FundingDataError(RuntimeError):
    """Historical funding is required but could not be loaded completely."""


def _finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def fetch_binance_ohlcv(symbol: str, interval: str = "1H",
                         months: int = 12) -> pd.DataFrame:
    """Fetch Binance USDT-M perpetual candles without authentication."""
    api_interval = _BINANCE_INTERVAL_MAP.get(interval, interval.lower())
    end_ms = int(_time.time() * 1000)
    start_ms = end_ms - int(months * 30.44 * 24 * 3600 * 1000)
    candles = []
    cursor = start_ms

    while cursor < end_ms:
        try:
            response = requests.get(
                f"{_BINANCE_FUTURES_BASE}/fapi/v1/klines",
                params={
                    "symbol": symbol,
                    "interval": api_interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=15,
            )
            if response.status_code != 200:
                break
            batch = response.json()
            if not batch:
                break
            candles.extend(batch)
            cursor = int(batch[-1][0]) + 1
            if len(batch) < 1000:
                break
        except Exception:
            break

    if not candles:
        return pd.DataFrame()
    frame = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
    frame.dropna(inplace=True)
    frame.sort_values("timestamp", inplace=True)
    frame.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def fetch_binance_funding_history(symbol: str, start_time_ms: int,
                                   end_time_ms: int) -> list[dict]:
    """Load every Binance funding settlement in the interval or fail closed."""
    start_time_ms = int(start_time_ms)
    end_time_ms = int(end_time_ms)
    if start_time_ms > end_time_ms:
        raise FundingDataError("funding start is after end")
    alignment_tolerance_ms = 1000
    request_start_ms = max(0, start_time_ms - alignment_tolerance_ms)
    request_end_ms = end_time_ms + alignment_tolerance_ms
    cursor = request_start_ms
    events = {}
    while cursor <= request_end_ms:
        try:
            response = requests.get(
                f"{_BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
                params={
                    "symbol": symbol,
                    "startTime": cursor,
                    "endTime": request_end_ms,
                    "limit": 1000,
                },
                timeout=15,
            )
        except Exception as exc:
            raise FundingDataError(f"Binance funding request failed: {exc}") from exc
        if response.status_code != 200:
            raise FundingDataError(
                f"Binance funding HTTP status {response.status_code}"
            )
        try:
            batch = response.json()
        except Exception as exc:
            raise FundingDataError("Binance funding response was not valid JSON") from exc
        if not isinstance(batch, list):
            raise FundingDataError("Binance funding response was not a list")
        if not batch:
            break

        last_time = None
        for row in batch:
            if not isinstance(row, dict):
                raise FundingDataError("Binance funding response contained a malformed row")
            try:
                funding_time = int(row["fundingTime"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FundingDataError("Binance funding row has invalid fundingTime") from exc
            rate = _finite_float(row.get("fundingRate"))
            if rate is None:
                raise FundingDataError("Binance funding row has invalid fundingRate")
            row_symbol = row.get("symbol") or symbol
            if row_symbol != symbol:
                raise FundingDataError(
                    f"Binance funding returned unexpected symbol {row_symbol}"
                )
            mark_price = _finite_float(row.get("markPrice"))
            if mark_price is not None and mark_price <= 0:
                mark_price = None
            last_time = funding_time if last_time is None else max(last_time, funding_time)
            if request_start_ms <= funding_time <= request_end_ms:
                events[funding_time] = {
                    "symbol": row_symbol,
                    "funding_time": funding_time,
                    "funding_rate": rate,
                    "mark_price": mark_price,
                }

        if last_time is None or last_time < cursor:
            raise FundingDataError("Binance funding pagination did not advance")
        cursor = last_time + 1
        if len(batch) < 1000:
            break

    ordered = [events[key] for key in sorted(events)]
    max_gap_ms = 8 * 60 * 60 * 1000
    if end_time_ms - start_time_ms >= max_gap_ms:
        if not ordered:
            raise FundingDataError(
                "Binance returned no funding settlements for an interval of at least 8h"
            )
        if ordered[0]["funding_time"] > start_time_ms + max_gap_ms:
            raise FundingDataError("Binance funding history does not cover the interval start")
        if ordered[-1]["funding_time"] < end_time_ms - max_gap_ms:
            raise FundingDataError("Binance funding history does not cover the interval end")
    return ordered


def _load_ohlcv(symbol: str, granularity: str, limit: int,
                months: int) -> tuple[pd.DataFrame, str]:
    if months > 0:
        # Long EMAs need substantially more history than their nominal period.
        # Fetch a three-month pre-roll, but report/trade only the requested span.
        frame = fetch_binance_ohlcv(symbol, granularity, months + 3)
        if not frame.empty:
            return frame, "Binance Futures"
        print(f"    (no Binance Futures data for {symbol}; falling back to Bitget)")
    return fetch_ohlcv(symbol, granularity, limit), "Bitget Futures"


def _adverse_fill(raw_price: float, side: str, entry: bool) -> float:
    if entry:
        factor = 1 + SLIPPAGE_RATE if side == "long" else 1 - SLIPPAGE_RATE
    else:
        factor = 1 - SLIPPAGE_RATE if side == "long" else 1 + SLIPPAGE_RATE
    return raw_price * factor


def funding_payment(side: str, quantity: float, mark_price: float,
                    funding_rate: float) -> float:
    """Signed settlement cashflow; leverage is deliberately not part of it."""
    direction = 1.0 if side == "long" else -1.0
    return -direction * float(quantity) * float(mark_price) * float(funding_rate)


def _align_funding_events_to_bars(events: list[dict], bar_times_ms: list[int],
                                   tolerance_ms: int = 1000) -> tuple[list[dict], list[int]]:
    """Snap exchange timestamp jitter only; reject genuine intrabar settlements."""
    from bisect import bisect_left

    boundaries = sorted(set(int(timestamp) for timestamp in bar_times_ms))
    aligned = []
    unaligned = []
    for event in events:
        raw_time = int(event["funding_time"])
        position = bisect_left(boundaries, raw_time)
        candidates = boundaries[max(0, position - 1):position + 1]
        if not candidates:
            unaligned.append(raw_time)
            continue
        nearest = min(candidates, key=lambda timestamp: abs(timestamp - raw_time))
        if abs(nearest - raw_time) > max(0, int(tolerance_ms)):
            unaligned.append(raw_time)
            continue
        normalized = dict(event)
        normalized["raw_funding_time"] = raw_time
        normalized["funding_time"] = nearest
        aligned.append(normalized)
    aligned.sort(key=lambda event: event["funding_time"])
    return aligned, unaligned


def _settle_funding_before_bar(trade: dict | None, events: list[dict],
                               event_index: int, bar_time_ms: int,
                               bar_open: float) -> tuple[int, float, int, int]:
    """Consume settlements through this open before any stop or next-open entry."""
    cashflow = 0.0
    fallback_count = 0
    applied_count = 0
    while event_index < len(events) and events[event_index]["funding_time"] <= bar_time_ms:
        event = events[event_index]
        event_index += 1
        if trade is None:
            continue
        mark_price = event.get("mark_price")
        mark_source = "api_markPrice"
        if mark_price is None or float(mark_price) <= 0:
            mark_price = bar_open
            mark_source = "bar_open_fallback"
            fallback_count += 1
        amount = funding_payment(
            trade["side"], trade["quantity"], mark_price, event["funding_rate"]
        )
        trade["funding"] = float(trade.get("funding") or 0.0) + amount
        trade.setdefault("funding_events", []).append({
            "funding_time": event["funding_time"],
            "funding_rate": event["funding_rate"],
            "mark_price": float(mark_price),
            "mark_source": mark_source,
            "amount": amount,
        })
        cashflow += amount
        applied_count += 1
    return event_index, cashflow, fallback_count, applied_count


def _finish_trade(trade: dict, raw_exit: float, result: str,
                  timestamp, exit_index: int) -> dict:
    exit_price = _adverse_fill(raw_exit, trade["side"], entry=False)
    direction = 1.0 if trade["side"] == "long" else -1.0
    gross_pnl = (exit_price - trade["entry"]) * trade["quantity"] * direction
    exit_fee = exit_price * trade["quantity"] * TAKER_FEE_RATE
    total_fees = trade["entry_fee"] + exit_fee
    price_fee_pnl = gross_pnl - total_fees
    funding_value = trade.get("funding")
    funding_known = funding_value is not None
    funding = float(funding_value) if funding_known else 0.0
    net_pnl = price_fee_pnl + funding
    trade.update({
        "exit_price": exit_price,
        "result": "trail" if result == "sl" and gross_pnl > 0 else result,
        "gross_pnl": round(gross_pnl, 6),
        "fees": round(total_fees, 6),
        "price_fee_pnl": round(price_fee_pnl, 6),
        "funding": round(funding, 6) if funding_known else None,
        "pnl": round(net_pnl, 6),
        "exit_ts": str(timestamp)[:16],
        "duration": exit_index - trade["entry_idx"],
    })
    return trade


def _mark_to_market(equity: float, trade: dict | None,
                    close_price: float) -> float:
    if not trade:
        return equity
    hypothetical_exit = _adverse_fill(close_price, trade["side"], entry=False)
    direction = 1.0 if trade["side"] == "long" else -1.0
    gross = (hypothetical_exit - trade["entry"]) * trade["quantity"] * direction
    exit_fee = hypothetical_exit * trade["quantity"] * TAKER_FEE_RATE
    return equity + gross - trade["entry_fee"] - exit_fee


def backtest(symbol: str, granularity: str = STRATEGY_TIMEFRAME,
             limit: int = 1000, months: int = 0) -> dict:
    raw_frame, source = _load_ohlcv(symbol, granularity, limit, months)
    raw_frame = closed_candles(raw_frame, granularity)
    if raw_frame.empty or len(raw_frame) < WARMUP + 10:
        return {"symbol": symbol, "trades": 0, "error": f"only {len(raw_frame)} candles"}

    frame = prepare_indicators(raw_frame)
    if months > 0:
        evaluation_start = (
            frame["timestamp"].iloc[-1] - pd.Timedelta(days=months * 30.44)
        ).floor("h")
        evaluation_index = int(frame["timestamp"].searchsorted(evaluation_start))
    else:
        evaluation_index = WARMUP
    loop_start = max(WARMUP, evaluation_index - 1)

    funding_available = source == "Binance Futures"
    funding_status = "available from Binance /fapi/v1/fundingRate"
    funding_events = []
    if funding_available:
        start_ms = int(pd.Timestamp(frame["timestamp"].iloc[loop_start]).timestamp() * 1000)
        end_ms = int(pd.Timestamp(frame["timestamp"].iloc[-1]).timestamp() * 1000)
        try:
            funding_events = fetch_binance_funding_history(symbol, start_ms, end_ms)
        except FundingDataError as exc:
            return {
                "symbol": symbol,
                "granularity": granularity,
                "source": source,
                "candles": len(frame) - evaluation_index,
                "trades": 0,
                "funding_available": False,
                "funding_status": f"unavailable: {exc}",
                "error": f"funding unavailable (fail-closed): {exc}",
            }
        bar_times_ms = [
            int(pd.Timestamp(timestamp).timestamp() * 1000)
            for timestamp in frame["timestamp"].iloc[loop_start:]
        ]
        funding_events, unaligned = _align_funding_events_to_bars(
            funding_events, bar_times_ms, tolerance_ms=1000
        )
        if unaligned:
            return {
                "symbol": symbol,
                "granularity": granularity,
                "source": source,
                "candles": len(frame) - evaluation_index,
                "trades": 0,
                "funding_available": False,
                "funding_status": "unavailable: settlements fall inside candles",
                "error": (
                    "funding unavailable (fail-closed): settlement ordering cannot "
                    f"be resolved on {granularity} bars"
                ),
            }
    else:
        funding_status = "funding unavailable for Bitget fallback; not treated as zero"

    trades = []
    trade = None
    pending = None
    cash_equity = 0.0
    peak_mtm = 0.0
    max_mtm_drawdown = 0.0
    funding_index = 0
    applied_funding_events = 0
    funding_mark_fallbacks = 0

    for index in range(loop_start, len(frame)):
        row = frame.iloc[index]
        bar_open = float(row["open"])
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])
        bar_time_ms = int(pd.Timestamp(row["timestamp"]).timestamp() * 1000)

        # The old position receives a settlement at this timestamp. A stop at the
        # same open still pays it; a pending entry at this open does not.
        if funding_available:
            funding_index, funding_cashflow, fallback_count, applied_count = (
                _settle_funding_before_bar(
                    trade, funding_events, funding_index, bar_time_ms, bar_open
                )
            )
            cash_equity += funding_cashflow
            funding_mark_fallbacks += fallback_count
            applied_funding_events += applied_count

        # A previous close signal is executed no earlier than this bar's open.
        if trade is None and pending is not None:
            entry_price = _adverse_fill(bar_open, pending["side"], entry=True)
            stop_loss, _ = best_sl_tp(pending["side"], entry_price, pending["atr"])
            quantity = POSITION_SIZE_USDT / entry_price
            trade = {
                "symbol": symbol,
                "side": pending["side"],
                "entry": entry_price,
                "entry_price": entry_price,
                "quantity": quantity,
                "size_usdt": POSITION_SIZE_USDT,
                "sl": stop_loss,
                "stop_loss": stop_loss,
                "initial_stop_loss": stop_loss,
                "best_price": entry_price,
                "entry_atr": pending["atr"],
                "entry_fee": POSITION_SIZE_USDT * TAKER_FEE_RATE,
                "funding": 0.0 if funding_available else None,
                "funding_events": [],
                "score": pending["score"],
                "entry_ts": str(row["timestamp"])[:16],
                "entry_idx": index,
            }
            pending = None

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

            if raw_exit is not None:
                completed = _finish_trade(
                    trade, raw_exit, "sl", row["timestamp"], index
                )
                trades.append(completed)
                # Funding was booked when each settlement occurred; add only the
                # close's price/fee component here to avoid double counting it.
                cash_equity += completed["price_fee_pnl"]
                trade = None
            else:
                trail_high = bar_high
                trail_low = bar_low
                if index == trade["entry_idx"]:
                    # Live enters shortly after the bar opens and cannot know the
                    # pre-entry extrema. Use close-only first-bar extrema in both paths.
                    trail_high = max(trade["entry"], bar_close)
                    trail_low = min(trade["entry"], bar_close)
                new_stop, best_price, _ = next_atr_trailing_stop(
                    trade,
                    candle_close=bar_close,
                    candle_high=trail_high,
                    candle_low=trail_low,
                    atr=float(row["atr"]),
                )
                trade["sl"] = new_stop
                trade["stop_loss"] = new_stop
                trade["best_price"] = best_price

        mtm = _mark_to_market(cash_equity, trade, bar_close)
        peak_mtm = max(peak_mtm, mtm)
        max_mtm_drawdown = max(max_mtm_drawdown, peak_mtm - mtm)

        # A position closed intrabar may form a fresh setup only at this bar close.
        if trade is None:
            signal = evaluate_signal(frame, index)
            if signal.get("entry_signal") in ("long", "short"):
                pending = {
                    "side": signal["entry_signal"],
                    "atr": float(signal["atr"]),
                    "score": float(signal["score"]),
                }

    open_price_fee_mtm = 0.0
    open_funding = None
    open_mtm = 0.0
    if trade is not None:
        # Live has no artificial end-of-dataset exit. Keep the open position out
        # of trade statistics and expose price/fee and settled funding separately.
        open_price_fee_mtm = _mark_to_market(
            0.0, trade, float(frame.iloc[-1]["close"])
        )
        open_funding = trade.get("funding")
        open_mtm = open_price_fee_mtm + (
            float(open_funding) if open_funding is not None else 0.0
        )

    closed_price_fee_pnl = sum(item["price_fee_pnl"] for item in trades)
    closed_funding = (
        sum(float(item["funding"]) for item in trades)
        if funding_available else None
    )
    closed_pnl = sum(item["pnl"] for item in trades)
    total_funding = (
        float(closed_funding or 0.0) + float(open_funding or 0.0)
        if funding_available else None
    )
    common = {
        "symbol": symbol,
        "granularity": granularity,
        "source": source,
        "candles": len(frame) - evaluation_index,
        "from": str(frame["timestamp"].iloc[evaluation_index])[:16],
        "to": str(frame["timestamp"].iloc[-1])[:16],
        "funding_available": funding_available,
        "funding_status": funding_status,
        "funding": round(total_funding, 2) if total_funding is not None else None,
        "closed_funding": round(closed_funding, 2) if closed_funding is not None else None,
        "open_funding": round(open_funding, 2) if open_funding is not None else None,
        "funding_events": applied_funding_events if funding_available else None,
        "funding_mark_fallbacks": funding_mark_fallbacks if funding_available else None,
        "price_fee_pnl": round(closed_price_fee_pnl, 2),
        "pnl": round(closed_pnl, 2),
        "open_price_fee_mtm": round(open_price_fee_mtm, 2),
        "open_mtm": round(open_mtm, 2),
        "max_drawdown": round(max_mtm_drawdown, 2),
    }

    if not trades:
        return {
            **common,
            "trades": 0,
            "error": "no closed trades triggered",
            "all_trades": [],
        }

    wins = [item for item in trades if item["pnl"] > 0]
    losses = [item for item in trades if item["pnl"] <= 0]
    gross_profit = sum(item["pnl"] for item in wins)
    gross_loss = abs(sum(item["pnl"] for item in losses))
    fees = sum(item["fees"] for item in trades)

    return {
        **common,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "fees": round(fees, 2),
        "roi_pct": round(closed_pnl / POSITION_SIZE_USDT * 100, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "best_trade": round(max(item["pnl"] for item in trades), 2),
        "worst_trade": round(min(item["pnl"] for item in trades), 2),
        "avg_duration": round(sum(item["duration"] for item in trades) / len(trades), 1),
        "all_trades": trades,
    }


def _bar(character="-", width=68):
    return character * width


def print_result(result: dict, show_trades: bool = False):
    if result.get("trades", 0) == 0:
        print(f"\n  [{result['symbol']}] {result.get('error', 'no trades')}")
        if "funding_status" in result:
            print(f"  Funding       : {result['funding_status']}")
        if "open_mtm" in result:
            suffix = "" if result.get("funding_available") else " (funding unavailable)"
            print(f"  Open MTM      : {result['open_mtm']:+.2f} USDT{suffix}")
        return

    print(f"\n{_bar()}")
    print(
        f"  {result['symbol']} | {result['granularity']} | {result['source']} | "
        f"{result['from']} -> {result['to']}"
    )
    print(
        f"  {result['candles']} candles | next-open | "
        f"fee {TAKER_FEE_RATE * 100:.3f}% + slippage {SLIPPAGE_RATE * 100:.3f}% per side"
    )
    print(_bar("."))
    print(
        f"  Trades        : {result['trades']} (W {result['wins']} / L {result['losses']}, "
        f"WR {result['win_rate']}%)"
    )
    if result["funding_available"]:
        print(f"  Net PnL       : {result['pnl']:+.2f} USDT (modeled fees/slippage + funding)")
        print(f"  Price/fee PnL : {result['price_fee_pnl']:+.2f} USDT")
        print(
            f"  Funding       : {result['closed_funding']:+.2f} USDT closed | "
            f"{result.get('open_funding') or 0:+.2f} USDT open | "
            f"{result['funding_events']} events"
        )
        if result.get("funding_mark_fallbacks"):
            print(
                f"  Mark fallback : {result['funding_mark_fallbacks']} settlement(s) used bar-open"
            )
    else:
        print(f"  Price/fee PnL : {result['price_fee_pnl']:+.2f} USDT (funding unavailable)")
        print(f"  Funding       : {result['funding_status']}")
    open_suffix = "" if result["funding_available"] else " (ex funding)"
    print(f"  Open MTM      : {result.get('open_mtm', 0):+.2f} USDT{open_suffix}")
    print(f"  Explicit fees : {result['fees']:.2f} USDT (slippage already in fills)")
    print(f"  Profit factor : {result['profit_factor']:.2f}")
    print(f"  Max MTM DD    : {result['max_drawdown']:.2f} USDT")
    print(f"  Avg win/loss  : {result['avg_win']:+.2f} / {result['avg_loss']:+.2f} USDT")
    print(f"  Best/worst    : {result['best_trade']:+.2f} / {result['worst_trade']:+.2f} USDT")
    print(f"  Avg duration  : {result['avg_duration']:.1f} candles")
    print(_bar())

    if show_trades:
        for item in result["all_trades"]:
            funding_text = (
                f" funding={item['funding']:+.3f}"
                if item.get("funding") is not None else " funding=unavailable"
            )
            print(
                f"  {item['entry_ts']} {item['side'].upper():5s} "
                f"{item['entry']:.8g} -> {item['exit_price']:.8g} "
                f"{item['result']:5s} {item['pnl']:+.3f}{funding_text}"
            )


def _aggregate_summary(results: list[dict]) -> dict:
    funding_complete = all(result.get("funding_available") for result in results)
    price_fee_pnl = sum(float(result.get("price_fee_pnl") or 0.0) for result in results)
    known_funding = sum(
        float(result.get("closed_funding") or 0.0)
        for result in results if result.get("funding_available")
    )
    return {
        "funding_complete": funding_complete,
        "price_fee_pnl": price_fee_pnl,
        "known_funding": known_funding,
        "net_pnl": (
            sum(float(result.get("pnl") or 0.0) for result in results)
            if funding_complete else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Cost-aware CryptoBot backtest")
    parser.add_argument("symbols", nargs="*", help="Symbols (default: research universe)")
    parser.add_argument("--granularity", "-g", default=STRATEGY_TIMEFRAME)
    parser.add_argument("--limit", "-l", type=int, default=1000)
    parser.add_argument("--months", "-m", type=int, default=0)
    parser.add_argument("--trades", action="store_true")
    # Accepted for compatibility with old commands; the new entry is a boolean setup.
    parser.add_argument("--threshold", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    symbols = args.symbols or get_top_symbols(8)
    if not symbols:
        print("No symbols available")
        sys.exit(1)

    print(f"\n{'=' * 68}")
    print("  CryptoBot signal/exit-core study (not a portfolio or fill simulation)")
    print(f"  Symbols: {', '.join(symbols)}")
    print(
        f"  Timeframe: {args.granularity} | notional: {POSITION_SIZE_USDT:.2f} USDT/trade | "
        f"source: {'Binance Futures' if args.months else 'Bitget Futures'}"
    )
    print(f"{'=' * 68}")

    results = []
    for symbol in symbols:
        print(f"  Backtesting {symbol}...", end=" ", flush=True)
        result = backtest(symbol, args.granularity, args.limit, args.months)
        results.append(result)
        if result.get("trades", 0):
            metric_label = "net" if result.get("funding_available") else "price/fee"
            print(
                f"{result['trades']} trades, {metric_label} {result['pnl']:+.2f}, "
                f"PF {result['profit_factor']:.2f}"
            )
        else:
            print(result.get("error", "no trades"))
        print_result(result, args.trades)

    valid = [result for result in results if result.get("trades", 0) > 0]
    if valid:
        total_trades = sum(result["trades"] for result in valid)
        total_wins = sum(result["wins"] for result in valid)
        total_fees = sum(result["fees"] for result in valid)
        summary = _aggregate_summary(valid)
        print(f"\n{'=' * 68}")
        print(f"  SUMMARY | {len(valid)} symbols | {total_trades} trades")
        print(f"  Win rate: {total_wins / total_trades * 100:.1f}%")
        if summary["funding_complete"]:
            print(
                f"  Net PnL: {summary['net_pnl']:+.2f} USDT | "
                f"price/fee: {summary['price_fee_pnl']:+.2f} | "
                f"funding: {summary['known_funding']:+.2f} | "
                f"explicit fees: {total_fees:.2f} USDT"
            )
        else:
            print(
                f"  Price/fee PnL: {summary['price_fee_pnl']:+.2f} USDT | "
                f"explicit fees: {total_fees:.2f} USDT"
            )
            print(
                f"  Known funding subset: {summary['known_funding']:+.2f} USDT; "
                "combined net PnL unavailable because at least one result lacks funding"
            )
        print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
