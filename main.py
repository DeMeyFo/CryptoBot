import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

from bitget_client import (
    close_order,
    get_account_info,
    get_current_price,
    get_funding_bills,
    get_historical_funding_rates,
    get_historical_mark_price,
    get_top_symbols,
    place_limit_order,
    place_order,
)
from config import (
    ALLOWED_TRADE_SIDES,
    DAILY_LOSS_LIMIT_PCT,
    DAILY_LOSS_LIMIT_USDT,
    DRY_RUN,
    FUNDING_MARK_WINDOW_MS,
    FUNDING_RATE_MATCH_TOLERANCE_SECONDS,
    FUNDING_SYNC_CLOSE_GRACE_SECONDS,
    FUNDING_SYNC_INTERVAL_SECONDS,
    FUNDING_SYNC_MAX_PAGES,
    FUNDING_SYNC_OVERLAP_SECONDS,
    FUNDING_SYNC_PAGE_SIZE,
    LEVERAGE,
    MAX_GROSS_EXPOSURE_PCT,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT,
    MAX_POSITION_USDT,
    MAX_SIGNAL_AGE_SECONDS,
    MAX_SYMBOL_EXPOSURE_PCT,
    MIN_POSITION_PCT,
    MIN_POSITION_USDT,
    POSITION_SIZE_USDT,
    PYRAMID_MAX_ADDS,
    PYRAMID_RISK_FRACTION,
    PYRAMID_STEP_R,
    RISK_PER_TRADE_PCT,
    SIZING_MODE,
    SECONDARY_ENABLED,
    SECONDARY_RISK_PER_TRADE_PCT,
    SECONDARY_TIMEFRAME,
    ENTRY_ORDER_TYPE,
    LIMIT_PULLBACK_ATR,
    STRATEGY_TIMEFRAME,
    VOL_ADAPTIVE_ENABLED,
    VOL_ADAPTIVE_HIGH_MULT,
    VOL_ADAPTIVE_LOOKBACK,
    VOL_ADAPTIVE_LOW_MULT,
    STRATEGY_VERSION,
    DYNAMIC_UNIVERSE_ENABLED,
    DYNAMIC_UNIVERSE_MAX,
    TOP_COINS_COUNT,
    TRADE_DIRECTION,
)
from database import (
    apply_pyramid_add,
    close_trade,
    find_trade_for_funding,
    funding_event_exists,
    get_all_trades,
    get_daily_funding,
    get_daily_pnl,
    get_funding_sync_candidates,
    get_funding_summary,
    get_incomplete_recent_funding_trade_ids,
    get_legacy_open_live_trades,
    get_open_trades,
    get_trade,
    init_db,
    is_entry_signal_consumed,
    mark_entry_signal_consumed,
    mark_funding_synced,
    quantity_at,
    record_funding_event,
    save_trade,
    update_trade_management,
)
from risk_manager import (
    best_sl_tp,
    calculate_position_notional,
    check_exit,
    exposure_headroom,
    next_atr_trailing_stop,
    position_notional_cap,
    position_notional_floor,
    unrealized_pnl,
)
import funding_carry
import ml_gate
import ml_prediction_log
from strategy import analyze_symbol
from technical_analysis import calculate_signal_history
from telegram_notify import (
    notify_bot_started,
    notify_daily_loss_limit,
    notify_daily_summary,
    notify_pyramid_add,
    notify_trade_closed,
    notify_trade_opened,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

_POSITION_CHECK_INTERVAL = 60
_ENTRY_RETRY_INTERVAL = 60
_INTERVAL_SECONDS = {"15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
_FUNDING_MODE = "dry_run" if DRY_RUN else "live"
_last_funding_refresh_at: float | None = None
_funding_sync_thread: threading.Thread | None = None
_funding_sync_lock = threading.Lock()


def _timeframe_seconds() -> int:
    return _INTERVAL_SECONDS.get(STRATEGY_TIMEFRAME.lower(), 3600)


def _secondary_timeframe_seconds() -> int:
    return _INTERVAL_SECONDS.get(SECONDARY_TIMEFRAME.lower(), 14400)


def _secondary_entry_window_open(now: float) -> bool:
    return 0 <= now % _secondary_timeframe_seconds() <= MAX_SIGNAL_AGE_SECONDS


def _parse_utc(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat()


def _timestamp_ms(timestamp: str | None) -> int | None:
    parsed = _parse_utc(timestamp)
    return int(parsed.timestamp() * 1000) if parsed else None


def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=__file__.rsplit("/", 1)[0] or ".",
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _sizing_description() -> str:
    if SIZING_MODE == "percent":
        return (
            f"risk {RISK_PER_TRADE_PCT * 100:.2f}%/trade, "
            f"cap {MAX_POSITION_PCT * 100:.0f}% of equity, "
            f"floor {MIN_POSITION_PCT * 100:.2f}% of equity"
        )
    return (
        f"risk {RISK_PER_TRADE_PCT * 100:.2f}%/trade, "
        f"cap {min(POSITION_SIZE_USDT, MAX_POSITION_USDT):.2f} USDT, "
        f"floor {MIN_POSITION_USDT:.2f} USDT"
    )


def _daily_limit_description() -> str:
    if SIZING_MODE == "percent":
        return f"{DAILY_LOSS_LIMIT_PCT * 100:.1f}% of equity"
    return f"{DAILY_LOSS_LIMIT_USDT:.2f} USDT"


def _exposure_description() -> str:
    return (
        f"{MAX_SYMBOL_EXPOSURE_PCT:g}x equity per symbol, "
        f"{MAX_GROSS_EXPOSURE_PCT:g}x gross"
    )


def _banner():
    mode = "DRY-RUN" if DRY_RUN else "LIVE"
    logger.info("=" * 72)
    logger.info(f"  Crypto Futures Bot - {mode} | code {_git_version()}")
    logger.info(
        f"  Strategy: {STRATEGY_VERSION} | candles: {STRATEGY_TIMEFRAME} | "
        f"direction: {TRADE_DIRECTION}"
    )
    logger.info(f"  Sizing ({SIZING_MODE}): {_sizing_description()}")
    logger.info(f"  Exposure ceiling: {_exposure_description()}")
    if PYRAMID_MAX_ADDS > 0:
        logger.warning(
            f"  Pyramiding ON: up to {PYRAMID_MAX_ADDS} adds every "
            f"{PYRAMID_STEP_R:g}R at {PYRAMID_RISK_FRACTION:g}x base risk. "
            "This won 2 of 4 half-years and lost the other 2; it is a bet on "
            "trend persistence, not a validated edge"
        )
    else:
        logger.info("  Pyramiding: off")
    logger.info(
        f"  Leverage: {LEVERAGE}x (margin only) | max open: {MAX_OPEN_POSITIONS} | "
        f"daily loss limit: {_daily_limit_description()}"
    )
    logger.info(
        f"  Entry window: {MAX_SIGNAL_AGE_SECONDS}s after candle close | "
        f"position checks: {_POSITION_CHECK_INTERVAL}s | "
        f"funding sync: >= {FUNDING_SYNC_INTERVAL_SECONDS}s"
    )
    if not DRY_RUN:
        logger.warning(
            "LIVE MODE: this strategy has only paper/backtest evidence and no profit guarantee"
        )
    logger.info("=" * 72)
    notify_bot_started(
        mode, LEVERAGE, _sizing_description(), TRADE_DIRECTION,
        _exposure_description(),
    )


def _open_position_count() -> int:
    return len(get_open_trades(dry_run=DRY_RUN))


def _symbol_has_open_trade(symbol: str) -> bool:
    """Fresh DB check: is there already an open trade for this symbol?

    Guards against two multi-timeframe sleeves opening the same symbol in the
    same loop tick, which would double the intended risk.
    """
    return any(
        t["symbol"] == symbol for t in get_open_trades(dry_run=DRY_RUN)
    )


def _funding_risk_state_complete() -> bool:
    if _funding_sync_thread and _funding_sync_thread.is_alive():
        logger.info("Funding reconciliation still running; entry scan deferred")
        return False
    incomplete = get_incomplete_recent_funding_trade_ids(DRY_RUN)
    if incomplete:
        logger.warning(
            f"Funding reconciliation incomplete for active/today trades {incomplete}; "
            "no new entries until the funding cursor is current"
        )
        return False
    return True


def _daily_loss_limit(equity: float) -> float:
    """
    Daily loss stop in USDT.

    An absolute limit does not scale: 25 USDT is 2.5% of a 1k account but 0.5%
    of a 5k account, so the same setting is either a constant interruption or
    no protection at all. In percent mode it tracks equity instead.
    """
    if SIZING_MODE == "percent" and equity > 0:
        return max(0.0, equity * DAILY_LOSS_LIMIT_PCT)
    return DAILY_LOSS_LIMIT_USDT


def _daily_loss_limit_hit(equity: float) -> bool:
    limit = _daily_loss_limit(equity)
    if limit <= 0:
        return False
    daily_pnl = get_daily_pnl(dry_run=DRY_RUN)
    if daily_pnl <= -limit:
        logger.warning(
            f"Daily loss limit hit: {daily_pnl:+.2f} USDT after modeled fees and funding "
            f"(limit -{limit:.2f}); no new entries"
        )
        return True
    return False


def _exposure_usdt(open_trades: list) -> tuple[float, dict]:
    """
    Entry notional held in total and per symbol.

    Matches how portfolio_backtest.py measured exposure, so the live ceilings
    bind at the same point that was validated.
    """
    per_symbol: dict[str, float] = {}
    gross = 0.0
    for trade in open_trades:
        notional = abs(float(trade.get("size_usdt") or 0.0))
        if notional <= 0:
            continue
        gross += notional
        symbol = trade["symbol"]
        per_symbol[symbol] = per_symbol.get(symbol, 0.0) + notional
    return gross, per_symbol



# ── Volatility-adaptive risk scaling ──────────────────────────────────────

_vol_history: dict[str, list[float]] = {}


def _vol_adaptive_risk_scale(symbol: str, current_atr_pct: float) -> float:
    """
    Scale risk inversely with realised volatility.

    When ATR/price is in the upper quartile of its recent history, reduce
    risk to VOL_ADAPTIVE_HIGH_MULT (0.7). When in the lower quartile,
    increase to VOL_ADAPTIVE_LOW_MULT (1.3). Linear between.
    """
    if not VOL_ADAPTIVE_ENABLED or current_atr_pct <= 0:
        return 1.0

    history = _vol_history.setdefault(symbol, [])
    history.append(current_atr_pct)
    # Keep only the lookback window
    if len(history) > VOL_ADAPTIVE_LOOKBACK:
        _vol_history[symbol] = history[-VOL_ADAPTIVE_LOOKBACK:]
        history = _vol_history[symbol]

    if len(history) < 50:
        return 1.0

    import numpy as np
    q25 = float(np.percentile(history, 25))
    q75 = float(np.percentile(history, 75))
    if q75 <= q25:
        return 1.0

    position = max(0.0, min(1.0, (current_atr_pct - q25) / (q75 - q25)))
    scale = VOL_ADAPTIVE_LOW_MULT + (VOL_ADAPTIVE_HIGH_MULT - VOL_ADAPTIVE_LOW_MULT) * position
    return max(0.3, min(1.5, scale))


def _trade_quantity(trade: dict) -> float:
    quantity = float(trade.get("quantity") or 0)
    if quantity > 0:
        return quantity
    if not trade.get("dry_run"):
        return 0.0
    entry = float(trade.get("entry_price") or 0)
    return float(trade.get("size_usdt") or 0) / entry if entry > 0 else 0.0


def _signal_is_fresh(candle_timestamp: str | None, now: float | None = None) -> bool:
    candle_open = _parse_utc(candle_timestamp)
    if candle_open is None:
        return False
    close_time = candle_open.timestamp() + _timeframe_seconds()
    age = (time.time() if now is None else now) - close_time
    return 0 <= age <= MAX_SIGNAL_AGE_SECONDS


def _entry_window_open(now: float) -> bool:
    return 0 <= now % _timeframe_seconds() <= MAX_SIGNAL_AGE_SECONDS


def _trailing_refresh_due(trade: dict, now: float) -> bool:
    last_candle = _parse_utc(trade.get("last_trail_at"))
    if last_candle is None:
        return True
    # A candle with the next timestamp is available after another full interval.
    return now >= last_candle.timestamp() + 2 * _timeframe_seconds() + 2


def _candidate_needs_funding_sync(trade: dict, now: datetime) -> bool:
    if trade.get("status") == "open":
        return True
    synced = _parse_utc(trade.get("funding_synced_at"))
    closed = _parse_utc(trade.get("closed_at"))
    if synced is None or closed is None:
        return True
    final_cursor = closed + timedelta(seconds=FUNDING_SYNC_CLOSE_GRACE_SECONDS)
    return synced < final_cursor


def _funding_sync_start_ms(trades: list[dict]) -> int | None:
    starts = []
    for trade in trades:
        opened = _parse_utc(trade.get("opened_at"))
        if opened is None:
            continue
        synced = _parse_utc(trade.get("funding_synced_at"))
        start = (
            synced - timedelta(seconds=FUNDING_SYNC_OVERLAP_SECONDS)
            if synced else opened
        )
        starts.append(max(opened, start))
    return int(min(starts).timestamp() * 1000) if starts else None


def _funding_required_start_ms(trade: dict) -> int | None:
    opened = _parse_utc(trade.get("opened_at"))
    if opened is None:
        return None
    synced = _parse_utc(trade.get("funding_synced_at"))
    required = max(opened, synced) if synced else opened
    return int(required.timestamp() * 1000)


def _trades_with_public_coverage(trades: list[dict], rate_result: dict,
                                 end_ms: int) -> list[dict]:
    coverage = rate_result.get("coverage") or {}
    observed_start = coverage.get("observed_start_ms")
    verified_through = coverage.get("verified_through_ms")
    interval_ms = coverage.get("schedule_interval_ms")
    latest_expected = coverage.get("latest_expected_ms")
    observed_times = sorted({
        int(timestamp) for timestamp in coverage.get("observed_times_ms") or []
    })
    if (
        observed_start is None
        or verified_through is None
        or int(verified_through) < end_ms
        or interval_ms is None
        or latest_expected is None
        or not observed_times
    ):
        return []

    gap_tolerance_ms = 5 * 60 * 1000
    covered = []
    for trade in trades:
        required_start = _funding_required_start_ms(trade)
        if required_start is None or int(observed_start) > required_start:
            continue
        anchors = [timestamp for timestamp in observed_times if timestamp <= required_start]
        if not anchors:
            continue
        anchor = anchors[-1]
        relevant = [
            timestamp for timestamp in observed_times
            if anchor <= timestamp <= int(latest_expected) + gap_tolerance_ms
        ]
        if not relevant:
            continue
        has_gap = any(
            current - previous > int(interval_ms) + gap_tolerance_ms
            for previous, current in zip(relevant, relevant[1:])
        )
        if not has_gap:
            covered.append(trade)
    return covered


def _matching_rate(rates: list[dict], timestamp_ms: int) -> dict | None:
    if not rates:
        return None
    closest = min(rates, key=lambda item: abs(item["funding_time"] - timestamp_ms))
    tolerance_ms = FUNDING_RATE_MATCH_TOLERANCE_SECONDS * 1000
    return closest if abs(closest["funding_time"] - timestamp_ms) <= tolerance_ms else None


def _quantity_at_settlement(trade: dict, settlement_at: str) -> float:
    """
    Quantity held by this trade when the settlement occurred.

    A pyramid add changes the position size mid-flight, so the current quantity
    is the wrong multiplier for a settlement that happened before the add. The
    recorded quantity history is authoritative; the current quantity is used
    only for trades that predate that history, which is exact for any position
    that never received an add.
    """
    historical = quantity_at(trade["id"], settlement_at)
    if historical is not None:
        return float(historical)
    return _trade_quantity(trade)


def _modeled_funding_amount(trade: dict, mark_price: float,
                            funding_rate: float, quantity: float) -> float:
    direction = 1.0 if trade["side"] == "long" else -1.0
    return -direction * float(quantity) * mark_price * funding_rate


def _record_dry_funding(symbol: str, rates: list[dict], stats: dict,
                        eligible_trade_ids: set[int]) -> bool:
    mark_cache = {}
    for rate in rates:
        external_id = f"{symbol}:{rate['funding_time']}"
        settlement_at = _iso_from_ms(rate["funding_time"])
        match = find_trade_for_funding(symbol, settlement_at, dry_run=True)
        if match["match_count"] > 1:
            affected = eligible_trade_ids.intersection(match["trade_ids"])
            if affected:
                logger.error(
                    f"Funding sync ambiguous for {symbol} at {settlement_at}: "
                    f"trades {match['trade_ids']}; cursor not advanced"
                )
                return False
            continue
        trade = match["trade"]
        if trade is None or trade["id"] not in eligible_trade_ids:
            continue
        if funding_event_exists("bitget_public_model", external_id):
            stats["duplicates"] += 1
            continue
        quantity = _quantity_at_settlement(trade, settlement_at)
        if quantity <= 0:
            logger.error(
                f"Funding sync cannot model {symbol} trade {trade['id']} at "
                f"{settlement_at}: no known quantity for that moment; "
                "cursor not advanced"
            )
            return False

        mark_result = mark_cache.get(rate["funding_time"])
        if mark_result is None:
            mark_result = get_historical_mark_price(
                symbol, rate["funding_time"], window_ms=FUNDING_MARK_WINDOW_MS
            )
            mark_cache[rate["funding_time"]] = mark_result
        mark_price = mark_result.get("mark_price") if mark_result.get("ok") else None
        if mark_price is None or float(mark_price) <= 0:
            logger.error(
                f"Funding sync missing mark price for {symbol} at {settlement_at}: "
                f"{mark_result.get('error')}; cursor not advanced"
            )
            return False

        amount = _modeled_funding_amount(
            trade, float(mark_price), float(rate["funding_rate"]), quantity
        )
        inserted = record_funding_event(
            source="bitget_public_model",
            external_id=external_id,
            trade_id=trade["id"],
            symbol=symbol,
            settlement_at=settlement_at,
            funding_rate=float(rate["funding_rate"]),
            mark_price=float(mark_price),
            amount_usdt=amount,
            mode="dry_run",
            raw_type="modeled_public_rate",
        )
        stats["inserted"] += int(inserted)
        stats["duplicates"] += int(not inserted)
    return True


def _record_live_funding(symbol: str, bills: list[dict], rates: list[dict],
                         stats: dict, eligible_trade_ids: set[int]) -> set[int]:
    """Persist authoritative bills and return trades with complete settlements."""
    incomplete_trade_ids: set[int] = set()
    matched_settlements: set[tuple[int, int]] = set()
    mark_cache = {}

    for bill in bills:
        rate = _matching_rate(rates, bill["created_time"])
        matching_time_ms = (
            int(rate["funding_time"]) if rate is not None else int(bill["created_time"])
        )
        matching_at = _iso_from_ms(matching_time_ms)
        match = find_trade_for_funding(symbol, matching_at, dry_run=False)
        if match["match_count"] > 1:
            affected = eligible_trade_ids.intersection(match["trade_ids"])
            if affected:
                logger.error(
                    f"Funding sync ambiguous for LIVE {symbol} bill {bill['bill_id']} "
                    f"at {matching_at}: trades {match['trade_ids']}"
                )
                incomplete_trade_ids.update(affected)
            continue
        trade = match["trade"]
        if trade is None:
            logger.error(
                f"LIVE funding bill {bill['bill_id']} for {symbol} at "
                f"{_iso_from_ms(bill['created_time'])} is unresolved; "
                "all candidate cursors remain incomplete"
            )
            incomplete_trade_ids.update(eligible_trade_ids)
            continue
        if trade["id"] not in eligible_trade_ids:
            logger.warning(
                f"Ignoring LIVE funding bill {bill['bill_id']} uniquely attributed "
                f"to non-candidate trade {trade['id']}"
            )
            continue

        trade_id = int(trade["id"])
        fee = float(bill.get("fee") or 0.0)
        raw_coin = bill.get("coin")
        bill_coin = str(raw_coin).upper() if raw_coin else ""
        if bill_coin != "USDT":
            logger.error(
                f"LIVE funding bill {bill['bill_id']} uses missing/unexpected coin "
                f"{bill_coin or 'unknown'}; trade {trade_id} cursor not advanced"
            )
            incomplete_trade_ids.add(trade_id)
            continue
        # Bitget exposes amount and fee separately. Until a non-zero funding-fee
        # fixture proves the net formula, refuse to guess and preserve unknown PnL.
        if abs(fee) > 1e-12:
            logger.error(
                f"LIVE funding bill {bill['bill_id']} has non-zero fee {fee}; "
                f"trade {trade_id} cursor not advanced"
            )
            incomplete_trade_ids.add(trade_id)
            continue

        amount = float(bill["amount"])
        if rate is not None:
            funding_rate = float(rate["funding_rate"])
            direction = 1.0 if trade["side"] == "long" else -1.0
            expected_sign = -direction * funding_rate
            if abs(funding_rate) <= 1e-15 and abs(amount) > 1e-12:
                logger.error(
                    f"LIVE funding bill {bill['bill_id']} has non-zero amount at a "
                    f"zero public rate; trade {trade_id} cursor not advanced"
                )
                incomplete_trade_ids.add(trade_id)
                continue
            if abs(amount) > 1e-12 and abs(expected_sign) > 1e-15 and amount * expected_sign < 0:
                logger.error(
                    f"LIVE funding bill {bill['bill_id']} sign contradicts {trade['side']} "
                    f"trade {trade_id}; cursor not advanced"
                )
                incomplete_trade_ids.add(trade_id)
                continue
            matched_settlements.add((trade_id, int(rate["funding_time"])))
        else:
            # The private cashflow remains authoritative and is persisted, but
            # without a public schedule match its interval cannot be certified.
            incomplete_trade_ids.add(trade_id)

        if funding_event_exists("bitget_bill", bill["bill_id"]):
            stats["duplicates"] += 1
            continue

        mark_price = None
        if rate is not None:
            settlement_ms = int(rate["funding_time"])
            mark_result = mark_cache.get(settlement_ms)
            if mark_result is None:
                mark_result = get_historical_mark_price(
                    symbol, settlement_ms, window_ms=FUNDING_MARK_WINDOW_MS
                )
                mark_cache[settlement_ms] = mark_result
            if mark_result.get("ok"):
                candidate_mark = mark_result.get("mark_price")
                if candidate_mark is not None and float(candidate_mark) > 0:
                    mark_price = float(candidate_mark)
            else:
                logger.warning(
                    f"LIVE funding metadata missing mark price for {symbol} bill "
                    f"{bill['bill_id']}: {mark_result.get('error')}"
                )

        # amount and cTime are the authoritative LIVE cashflow and posting time.
        # Public rate/mark fields remain optional audit metadata only.
        inserted = record_funding_event(
            source="bitget_bill",
            external_id=bill["bill_id"],
            trade_id=trade_id,
            symbol=symbol,
            settlement_at=_iso_from_ms(bill["created_time"]),
            funding_rate=float(rate["funding_rate"]) if rate is not None else None,
            mark_price=mark_price,
            amount_usdt=amount,
            mode="live",
            raw_type=bill["business_type"],
        )
        stats["inserted"] += int(inserted)
        stats["duplicates"] += int(not inserted)

    # Public rates only define when a cashflow should exist. They never supply a
    # LIVE amount: each non-zero expected settlement needs a private bill.
    for rate in rates:
        settlement_ms = int(rate["funding_time"])
        settlement_at = _iso_from_ms(settlement_ms)
        match = find_trade_for_funding(symbol, settlement_at, dry_run=False)
        if match["match_count"] > 1:
            incomplete_trade_ids.update(
                eligible_trade_ids.intersection(match["trade_ids"])
            )
            continue
        trade = match["trade"]
        if trade is None or trade["id"] not in eligible_trade_ids:
            continue
        trade_id = int(trade["id"])
        if abs(float(rate["funding_rate"])) <= 1e-15:
            continue
        if (trade_id, settlement_ms) not in matched_settlements:
            logger.error(
                f"LIVE funding bill missing for {symbol} trade {trade_id} at "
                f"{settlement_at}; cursor not advanced"
            )
            incomplete_trade_ids.add(trade_id)

    return eligible_trade_ids - incomplete_trade_ids


def _sync_funding_cashflows(now: float) -> dict:
    now_dt = datetime.fromtimestamp(now, timezone.utc)
    candidates = [
        trade for trade in get_funding_sync_candidates(
            DRY_RUN,
            close_grace_seconds=FUNDING_SYNC_CLOSE_GRACE_SECONDS,
        )
        if _candidate_needs_funding_sync(trade, now_dt)
    ]
    stats = {
        "symbols": 0,
        "inserted": 0,
        "duplicates": 0,
        "failed_symbols": 0,
        "uncovered_trades": 0,
    }
    grouped = {}
    for trade in candidates:
        grouped.setdefault(trade["symbol"], []).append(trade)

    end_ms = int(now * 1000)
    synced_at = now_dt.isoformat()
    for symbol, trades in grouped.items():
        stats["symbols"] += 1
        start_ms = _funding_sync_start_ms(trades)
        if start_ms is None or start_ms > end_ms:
            logger.error(f"Funding sync has invalid time range for {symbol}; cursor not advanced")
            stats["failed_symbols"] += 1
            continue
        try:
            rate_result = get_historical_funding_rates(
                symbol,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                page_size=FUNDING_SYNC_PAGE_SIZE,
                max_pages=FUNDING_SYNC_MAX_PAGES,
            )
            if not rate_result.get("ok"):
                logger.error(
                    f"Funding rate sync failed for {symbol}: {rate_result.get('error')}; "
                    "cursor not advanced"
                )
                stats["failed_symbols"] += 1
                continue

            covered_trades = _trades_with_public_coverage(trades, rate_result, end_ms)
            covered_ids = {int(trade["id"]) for trade in covered_trades}
            uncovered_ids = {int(trade["id"]) for trade in trades} - covered_ids
            stats["uncovered_trades"] += len(uncovered_ids)
            if uncovered_ids:
                coverage = rate_result.get("coverage") or {}
                logger.warning(
                    f"Funding history does not cover {symbol} trades "
                    f"{sorted(uncovered_ids)} from their cursor; oldest public settlement="
                    f"{coverage.get('observed_start_ms')}; left unknown"
                )
            if not covered_ids:
                stats["failed_symbols"] += 1
                continue

            if DRY_RUN:
                complete_ids = (
                    covered_ids
                    if _record_dry_funding(
                        symbol, rate_result["items"], stats, covered_ids
                    )
                    else set()
                )
            else:
                live_start_ms = _funding_sync_start_ms(covered_trades)
                bill_result = get_funding_bills(
                    symbol=symbol,
                    start_time_ms=live_start_ms,
                    end_time_ms=end_ms,
                    limit=FUNDING_SYNC_PAGE_SIZE,
                    max_pages=FUNDING_SYNC_MAX_PAGES,
                )
                if not bill_result.get("ok"):
                    logger.error(
                        f"Funding bill sync failed for {symbol}: {bill_result.get('error')}; "
                        "cursor not advanced"
                    )
                    stats["failed_symbols"] += 1
                    continue
                complete_ids = _record_live_funding(
                    symbol, bill_result["items"], rate_result["items"],
                    stats, covered_ids,
                )

            if complete_ids:
                mark_funding_synced(sorted(complete_ids), synced_at)
            if complete_ids != {int(trade["id"]) for trade in trades}:
                stats["failed_symbols"] += 1
        except Exception as exc:
            logger.error(
                f"Funding sync failed unexpectedly for {symbol}: {exc}; cursor not advanced",
                exc_info=True,
            )
            stats["failed_symbols"] += 1

    if stats["symbols"]:
        logger.info(
            f"Funding sync {_FUNDING_MODE}: symbols={stats['symbols']} "
            f"inserted={stats['inserted']} deduped={stats['duplicates']} "
            f"uncovered_trades={stats['uncovered_trades']} "
            f"failed={stats['failed_symbols']}"
        )
    return stats


def _funding_sync_worker(now: float) -> None:
    try:
        _sync_funding_cashflows(now)
    except Exception as exc:
        logger.error(f"Funding sync worker failed: {exc}", exc_info=True)


def refresh_funding_cashflows(now: float | None = None,
                              background: bool = False) -> dict:
    """Rate-limited funding reconciliation; background mode cannot delay exits."""
    global _last_funding_refresh_at, _funding_sync_thread
    now = time.time() if now is None else float(now)
    with _funding_sync_lock:
        if background and _funding_sync_thread and _funding_sync_thread.is_alive():
            return {"skipped": "already running"}
        if (
            _last_funding_refresh_at is not None
            and now - _last_funding_refresh_at < FUNDING_SYNC_INTERVAL_SECONDS
        ):
            return {"skipped": "rate limited"}
        _last_funding_refresh_at = now
        if background:
            _funding_sync_thread = threading.Thread(
                target=_funding_sync_worker,
                args=(now,),
                name="funding-sync",
                daemon=True,
            )
            _funding_sync_thread.start()
            return {"started": True}
    return _sync_funding_cashflows(now)


def _close_confirmed(trade: dict, observed_price: float, reason: str) -> bool:
    quantity = _trade_quantity(trade)
    if quantity <= 0:
        logger.critical(
            f"Cannot close {trade['symbol']} trade {trade['id']}: quantity is unknown"
        )
        return False
    result = close_order(trade["symbol"], trade["side"], quantity, observed_price)
    if not result:
        logger.error(
            f"CLOSE NOT CONFIRMED {trade['side'].upper()} {trade['symbol']} - "
            f"trade {trade['id']} remains open in the database"
        )
        return False

    exit_price = float(result.get("fill_price") or observed_price)
    realised = close_trade(
        trade["id"], exit_price, reason,
        exit_order_id=result.get("orderId") or result.get("clientOid"),
    )
    closed_trade = get_trade(trade["id"]) or trade
    funding_value = (
        closed_trade.get("funding_usdt")
        if int(closed_trade.get("pnl_model_version") or 1) >= 3
        else None
    )
    pnl_description = (
        "PnL after modeled fees and reconciled funding"
        if funding_value is not None
        else "provisional PnL after modeled fees; funding pending"
    )
    logger.info(
        f"{'OK' if realised >= 0 else 'LOSS'} {reason.upper()} "
        f"{trade['side'].upper()} {trade['symbol']} exit={exit_price:.8g} "
        f"{pnl_description}={realised:+.4f} USDT"
    )
    notify_trade_closed(
        trade["symbol"], trade["side"], reason, exit_price, realised, DRY_RUN,
        funding_usdt=funding_value,
    )
    return True


def refresh_trailing_stops(now: float | None = None) -> dict[int, float]:
    """
    Replay every newly closed candle before checking current prices.

    Returns the close of the last newly replayed candle per trade id. Pyramiding
    uses it so its trigger is judged on closed-candle data only, exactly as it
    was validated, instead of on an intrabar tick.
    """
    now = time.time() if now is None else now
    replayed_closes: dict[int, float] = {}
    for trade in get_open_trades(dry_run=DRY_RUN):
        if not _trailing_refresh_due(trade, now):
            continue
        history = calculate_signal_history(
            trade["symbol"],
            after_timestamp=trade.get("last_trail_at"),
            granularity=STRATEGY_TIMEFRAME,
        )
        if history.get("error"):
            logger.error(
                f"Trailing catch-up blocked for {trade['symbol']}: {history['error']}; "
                "cursor was not advanced"
            )
            continue
        snapshots = history.get("snapshots", [])
        if not snapshots:
            continue

        working_trade = dict(trade)
        moved_count = 0
        for snapshot in snapshots:
            candle_time = snapshot["timestamp"]
            indicators = snapshot["indicators"]
            candle_close = float(indicators["current_price"])
            candle_high = float(indicators["candle_high"])
            candle_low = float(indicators["candle_low"])

            # Live enters just after the bar opens. For that first bar, use only
            # Entry/Close extrema because pre-entry ticks cannot be separated.
            candle_open = _parse_utc(candle_time)
            opened_at = _parse_utc(working_trade.get("opened_at"))
            if candle_open and opened_at and candle_open <= opened_at < (
                candle_open + timedelta(seconds=_timeframe_seconds())
            ):
                entry = float(working_trade["entry_price"])
                candle_high = max(entry, candle_close)
                candle_low = min(entry, candle_close)

            new_stop, best_price, moved = next_atr_trailing_stop(
                working_trade,
                candle_close=candle_close,
                candle_high=candle_high,
                candle_low=candle_low,
                atr=float(snapshot["atr"]),
            )
            moved_count += int(moved)
            working_trade["stop_loss"] = new_stop
            working_trade["best_price"] = best_price
            working_trade["last_trail_at"] = candle_time

        update_trade_management(
            trade["id"],
            float(working_trade["stop_loss"]),
            float(working_trade["best_price"]),
            working_trade["last_trail_at"],
        )
        replayed_closes[trade["id"]] = float(
            snapshots[-1]["indicators"]["current_price"]
        )
        if moved_count:
            logger.info(
                f"TRAIL {trade['side'].upper()} {trade['symbol']} replayed "
                f"{len(snapshots)} candle(s), SL "
                f"{float(trade.get('stop_loss') or 0):.8g} -> "
                f"{float(working_trade['stop_loss']):.8g}"
            )
    return replayed_closes


def manage_pyramid_adds(replayed_closes: dict[int, float]):
    """
    Add to winners that advanced another PYRAMID_STEP_R since the last add.

    Contract, matching how this was validated in portfolio_backtest.py:
      * the trigger is judged on a closed candle's close, never on a tick,
      * the fill happens right after that candle closed, i.e. at the next open,
      * all units share the one trailing stop, so the add is sized against the
        current stop rather than the original one,
      * entry_price and initial_stop_loss stay untouched, so the trail and the R
        measurement keep referring to the original entry.

    Deliberate deviation: at most one add per refresh. After downtime the
    excursion may already span several thresholds; the backtest would have added
    once per bar. Adding once is the conservative direction.

    The daily loss stop intentionally does not block adds, because it did not
    block them in the measurement either. The exposure ceilings do apply.
    """
    if PYRAMID_MAX_ADDS <= 0 or not replayed_closes:
        return

    for trade in get_open_trades(dry_run=DRY_RUN):
        candle_close = replayed_closes.get(trade["id"])
        if candle_close is None or candle_close <= 0:
            continue
        adds_done = int(trade.get("pyramid_count") or 0)
        if adds_done >= PYRAMID_MAX_ADDS:
            continue

        symbol = trade["symbol"]
        side = trade["side"]
        original_entry = float(trade.get("entry_price") or 0)
        initial_stop = float(trade.get("initial_stop_loss") or 0)
        base_risk = abs(original_entry - initial_stop)
        if original_entry <= 0 or base_risk <= 0:
            continue

        excursion = (
            (candle_close - original_entry) if side == "long"
            else (original_entry - candle_close)
        ) / base_risk
        if excursion < (adds_done + 1) * PYRAMID_STEP_R:
            continue

        current_stop = float(trade.get("stop_loss") or 0)
        observed_price = get_current_price(symbol)
        if observed_price <= 0:
            logger.warning(f"No price for pyramid add on {symbol}")
            continue
        stop_is_valid = (
            current_stop < observed_price if side == "long"
            else current_stop > observed_price
        )
        if not stop_is_valid or current_stop <= 0:
            logger.info(
                f"{symbol}: pyramid add skipped, stop {current_stop:.8g} is not "
                f"protective against {observed_price:.8g}"
            )
            continue

        account = get_account_info()
        equity = float(account.get("equity") or 0)
        gross_exposure, symbol_exposure = _exposure_usdt(
            get_open_trades(dry_run=DRY_RUN)
        )
        notional = calculate_position_notional(
            equity=equity,
            entry_price=observed_price,
            stop_loss=current_stop,
            max_notional=_available_notional_cap(account),
            symbol_exposure_usdt=symbol_exposure.get(symbol, 0.0),
            gross_exposure_usdt=gross_exposure,
            risk_scale=PYRAMID_RISK_FRACTION,
        )
        if notional <= 0:
            logger.info(
                f"{symbol}: pyramid add #{adds_done + 1} has no room. "
                f"headroom="
                f"{exposure_headroom(equity, symbol_exposure.get(symbol, 0.0), gross_exposure):.2f} "
                f"floor={position_notional_floor(equity):.2f} USDT"
            )
            continue

        # Entry order: limit or market based on config
        if ENTRY_ORDER_TYPE == "limit" and analysis.get("atr"):
            # Limit price: slightly below market for longs (buy the pullback)
            atr = float(analysis["atr"])
            if side == "long":
                limit_price = observed_price - atr * LIMIT_PULLBACK_ATR
            else:
                limit_price = observed_price + atr * LIMIT_PULLBACK_ATR
            order = place_limit_order(
                symbol, side, notional, LEVERAGE, limit_price, observed_price
            )
        else:
            order = place_order(symbol, side, notional, LEVERAGE, observed_price)
        if not order:
            continue
        fill_price = float(order.get("fill_price") or observed_price)
        add_quantity = float(order.get("quantity") or 0)
        order_id = order.get("orderId") or order.get("clientOid")
        if add_quantity <= 0:
            logger.critical(
                f"Pyramid add {order_id} for {symbol} has no tracked quantity; "
                "manual exchange reconciliation required"
            )
            continue

        state = apply_pyramid_add(
            trade["id"], add_quantity, fill_price,
            datetime.now(timezone.utc).isoformat(), PYRAMID_MAX_ADDS,
        )
        if state is None:
            logger.critical(
                f"Pyramid add {order_id} for {symbol} filled {add_quantity} but "
                f"could not be recorded on trade {trade['id']}; the exchange "
                "position is now larger than the database. Manual "
                "reconciliation required"
            )
            continue

        prefix = "[DRY-RUN] " if DRY_RUN else ""
        logger.info(
            f"{prefix}PYRAMID #{state['pyramid_count']}/{PYRAMID_MAX_ADDS} "
            f"{side.upper()} {symbol} @ {fill_price:.8g} "
            f"added={state['added_notional']:.2f} USDT "
            f"total={state['size_usdt']:.2f} USDT "
            f"avg_entry={state['avg_entry_price']:.8g} "
            f"({excursion:.2f}R since entry) trade_id={trade['id']}"
        )
        notify_pyramid_add(
            symbol=symbol,
            side=side,
            add_number=state["pyramid_count"],
            max_adds=PYRAMID_MAX_ADDS,
            price=fill_price,
            added_notional=state["added_notional"],
            total_notional=state["size_usdt"],
            avg_entry=state["avg_entry_price"],
            reached_r=excursion,
            dry_run=DRY_RUN,
        )


def manage_open_trades():
    for trade in get_open_trades(dry_run=DRY_RUN):
        price = get_current_price(trade["symbol"])
        if price <= 0:
            logger.warning(f"No current price for open trade {trade['symbol']}")
            continue
        reason = check_exit(trade, price)
        if reason:
            _close_confirmed(trade, price, reason)
            continue
        funding = trade.get("funding_usdt")
        funding_text = f"{float(funding):+.4f}" if funding is not None else "unknown"
        logger.debug(
            f"HOLD {trade['side'].upper()} {trade['symbol']} "
            f"entry={trade['entry_price']:.8g} now={price:.8g} "
            f"price-uPnL gross={unrealized_pnl(trade, price):+.4f} USDT "
            f"settled funding={funding_text} USDT"
        )


def _available_notional_cap(account: dict) -> float:
    equity = float(account.get("equity") or 0)
    configured_cap = position_notional_cap(equity)
    margin_cap = float(account.get("available") or 0) * LEVERAGE * 0.90
    return max(0.0, min(configured_cap, margin_cap))


def scan_secondary_entries():
    """
    Secondary timeframe sleeve (e.g. 4H). Same signal logic, own risk budget.

    Runs independently of the primary scan, with its own entry window timing.
    Positions from both sleeves share the same MAX_OPEN_POSITIONS and exposure
    limits, so the total risk stays bounded.
    """
    if not SECONDARY_ENABLED:
        return
    if _open_position_count() >= MAX_OPEN_POSITIONS:
        return
    if not _funding_risk_state_complete():
        return

    scan_account = get_account_info()
    scan_equity = float(scan_account.get("equity") or 0)
    if _daily_loss_limit_hit(scan_equity):
        return

    symbols = get_top_symbols(TOP_COINS_COUNT)
    if not symbols:
        return

    logger.info(
        f"Secondary scan ({SECONDARY_TIMEFRAME}): {len(symbols)} symbols"
    )
    open_symbols = {trade["symbol"] for trade in get_open_trades(dry_run=DRY_RUN)}

    for symbol in symbols:
        if symbol in open_symbols:
            continue
        if _open_position_count() >= MAX_OPEN_POSITIONS:
            break

        analysis = analyze_symbol(symbol, granularity=SECONDARY_TIMEFRAME)
        if analysis["action"] == "hold":
            continue
        if analysis["action"] not in ALLOWED_TRADE_SIDES:
            continue
        candle_timestamp = analysis.get("timestamp")
        if not candle_timestamp:
            continue
        # Freshness check adapted for secondary timeframe
        candle_open = _parse_utc(candle_timestamp)
        if candle_open is None:
            continue
        close_time = candle_open.timestamp() + _secondary_timeframe_seconds()
        age = time.time() - close_time
        if not (0 <= age <= MAX_SIGNAL_AGE_SECONDS):
            continue
        if is_entry_signal_consumed(symbol, candle_timestamp, DRY_RUN):
            continue

        # ML Gate
        gate_decision = ml_gate.evaluate(symbol, analysis)
        if gate_decision.action == "block":
            logger.info(
                f"{symbol}: ML-Gate blocked secondary entry "
                f"({gate_decision.filter_reason})"
            )
            try:
                ml_prediction_log.record(gate_decision, symbol)
            except Exception:
                pass
            continue
        risk_scale = gate_decision.risk_scale

        # Volatility-adaptive risk scaling
        if VOL_ADAPTIVE_ENABLED and analysis.get("atr") and analysis.get("indicators"):
            price = float(analysis["indicators"].get("current_price", 0))
            if price > 0:
                atr_pct = float(analysis["atr"]) / price
                vol_scale = _vol_adaptive_risk_scale(symbol, atr_pct)
                risk_scale *= vol_scale

        observed_price = get_current_price(symbol)
        if observed_price <= 0:
            continue

        side = analysis["action"]
        stop_loss, take_profit = best_sl_tp(side, observed_price, analysis.get("atr"))
        account = get_account_info()
        equity = float(account.get("equity") or 0)
        gross_exposure, symbol_exposure = _exposure_usdt(
            get_open_trades(dry_run=DRY_RUN)
        )
        # Secondary sleeve uses its own risk budget
        notional = calculate_position_notional(
            equity=equity,
            entry_price=observed_price,
            stop_loss=stop_loss,
            max_notional=_available_notional_cap(account),
            symbol_exposure_usdt=symbol_exposure.get(symbol, 0.0),
            gross_exposure_usdt=gross_exposure,
            risk_scale=risk_scale * (SECONDARY_RISK_PER_TRADE_PCT / RISK_PER_TRADE_PCT),
        )
        if notional <= 0:
            continue

        if _symbol_has_open_trade(symbol):
            logger.info(
                f"{symbol}: skip entry, position already open (concurrent sleeve)"
            )
            continue
        order = place_order(symbol, side, notional, LEVERAGE, observed_price)
        if not order:
            continue
        order_id = order.get("orderId") or order.get("clientOid")
        mark_entry_signal_consumed(symbol, candle_timestamp, DRY_RUN, order_id=order_id)

        entry_price = float(order.get("fill_price") or observed_price)
        quantity = float(order.get("quantity") or 0)
        if quantity <= 0:
            logger.critical(
                f"Secondary order {order_id} for {symbol} has no tracked quantity"
            )
            continue

        stop_loss, take_profit = best_sl_tp(side, entry_price, analysis.get("atr"))
        actual_notional = quantity * entry_price
        trade_id = save_trade(
            symbol=symbol, side=side, entry_price=entry_price,
            size_usdt=actual_notional, quantity=quantity, leverage=LEVERAGE,
            stop_loss=stop_loss, initial_stop_loss=stop_loss,
            take_profit=take_profit, dry_run=DRY_RUN,
            signal_score=analysis["final_score"], order_id=order_id,
            best_price=entry_price, entry_atr=analysis.get("atr"),
            last_trail_at=candle_timestamp,
        )
        prefix = "[DRY-RUN] " if DRY_RUN else ""
        logger.info(
            f"{prefix}OPEN-4H {side.upper()} {symbol} @ {entry_price:.8g} "
            f"qty={quantity} notional={actual_notional:.2f} SL={stop_loss:.8g} "
            f"trade_id={trade_id}"
        )
        notify_trade_opened(
            symbol=symbol, side=side, price=entry_price, size=actual_notional,
            score=analysis["final_score"], sl=stop_loss, tp=take_profit,
            adx=analysis.get("adx", 0), regime=f"core-4H",
            dry_run=DRY_RUN,
        )
        open_symbols.add(symbol)


def scan_new_entries():
    if _open_position_count() >= MAX_OPEN_POSITIONS:
        logger.info(f"Max positions reached ({MAX_OPEN_POSITIONS}); scan skipped")
        return
    if not _funding_risk_state_complete():
        return

    scan_account = get_account_info()
    scan_equity = float(scan_account.get("equity") or 0)
    if _daily_loss_limit_hit(scan_equity):
        notify_daily_loss_limit(
            get_daily_pnl(dry_run=DRY_RUN), _daily_loss_limit(scan_equity)
        )
        return

    if DYNAMIC_UNIVERSE_ENABLED:
        from bitget_client import get_dynamic_universe
        symbols = get_dynamic_universe(DYNAMIC_UNIVERSE_MAX)
    else:
        symbols = get_top_symbols(TOP_COINS_COUNT)
    if not symbols:
        logger.warning("No verified active research-universe symbols available")
        return

    logger.info(
        f"Scanning {len(symbols)} symbols ({TRADE_DIRECTION})"
        f"{' [dynamic]' if DYNAMIC_UNIVERSE_ENABLED else ''}: "
        f"{', '.join(symbols)}"
    )
    open_symbols = {trade["symbol"] for trade in get_open_trades(dry_run=DRY_RUN)}

    for symbol in symbols:
        if symbol in open_symbols:
            continue
        if _open_position_count() >= MAX_OPEN_POSITIONS:
            break

        analysis = analyze_symbol(symbol)
        if analysis["action"] == "hold":
            continue
        if analysis["action"] not in ALLOWED_TRADE_SIDES:
            logger.info(
                f"{symbol}: {analysis['action']} signal ignored; "
                f"TRADE_DIRECTION={TRADE_DIRECTION}"
            )
            continue
        candle_timestamp = analysis.get("timestamp")
        if not _signal_is_fresh(candle_timestamp):
            logger.info(f"Stale signal ignored for {symbol}: {candle_timestamp}")
            continue
        if is_entry_signal_consumed(symbol, candle_timestamp, DRY_RUN):
            logger.info(f"Already consumed signal ignored for {symbol}: {candle_timestamp}")
            continue

        # --- ML-Gate evaluation (Anforderung 1.1, 1.2, 1.3, 1.4, 2.2) ---
        gate_decision = ml_gate.evaluate(symbol, analysis)
        if gate_decision.action == "block":
            logger.info(
                f"{symbol}: ML-Gate blocked entry "
                f"(regime={gate_decision.regime}, "
                f"confidence={gate_decision.confidence:.3f}, "
                f"reason={gate_decision.filter_reason})"
            )
            try:
                ml_prediction_log.record(
                    gate_decision, symbol,
                    training_period=ml_gate.get_training_period(symbol),
                )
            except Exception as exc:
                logger.error(f"ML prediction log failed for {symbol}: {exc}")
            continue
        risk_scale = gate_decision.risk_scale

        observed_price = get_current_price(symbol)
        if observed_price <= 0:
            logger.warning(f"No price for actionable signal {symbol}")
            continue

        side = analysis["action"]
        stop_loss, take_profit = best_sl_tp(side, observed_price, analysis.get("atr"))
        account = get_account_info()
        equity = float(account.get("equity") or 0)
        gross_exposure, symbol_exposure = _exposure_usdt(
            get_open_trades(dry_run=DRY_RUN)
        )
        notional = calculate_position_notional(
            equity=equity,
            entry_price=observed_price,
            stop_loss=stop_loss,
            max_notional=_available_notional_cap(account),
            symbol_exposure_usdt=symbol_exposure.get(symbol, 0.0),
            gross_exposure_usdt=gross_exposure,
            risk_scale=risk_scale,
        )
        if notional <= 0:
            headroom = exposure_headroom(
                equity, symbol_exposure.get(symbol, 0.0), gross_exposure
            )
            logger.warning(
                f"{symbol}: no tradable size. equity={equity:.2f} "
                f"floor={position_notional_floor(equity):.2f} "
                f"exposure_headroom={headroom:.2f} "
                f"gross_open={gross_exposure:.2f} USDT"
            )
            continue

        if _symbol_has_open_trade(symbol):
            logger.info(
                f"{symbol}: skip entry, position already open (concurrent sleeve)"
            )
            continue
        order = place_order(symbol, side, notional, LEVERAGE, observed_price)
        if not order:
            continue
        order_id = order.get("orderId") or order.get("clientOid")
        mark_entry_signal_consumed(
            symbol, candle_timestamp, DRY_RUN, order_id=order_id
        )

        entry_price = float(order.get("fill_price") or observed_price)
        quantity = float(order.get("quantity") or 0)
        if quantity <= 0:
            logger.critical(
                f"Accepted order {order_id} for {symbol} has no tracked quantity; "
                "manual exchange reconciliation required"
            )
            continue

        stop_loss, take_profit = best_sl_tp(side, entry_price, analysis.get("atr"))
        actual_notional = quantity * entry_price
        trade_id = save_trade(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size_usdt=actual_notional,
            quantity=quantity,
            leverage=LEVERAGE,
            stop_loss=stop_loss,
            initial_stop_loss=stop_loss,
            take_profit=take_profit,
            dry_run=DRY_RUN,
            signal_score=analysis["final_score"],
            order_id=order_id,
            best_price=entry_price,
            entry_atr=analysis.get("atr"),
            last_trail_at=candle_timestamp,
        )
        prefix = "[DRY-RUN] " if DRY_RUN else ""
        logger.info(
            f"{prefix}OPEN {side.upper()} {symbol} @ {entry_price:.8g} "
            f"qty={quantity} notional={actual_notional:.2f} SL={stop_loss:.8g} "
            f"TP=ATR trail (no fixed TP) trade_id={trade_id}"
        )
        notify_trade_opened(
            symbol=symbol,
            side=side,
            price=entry_price,
            size=actual_notional,
            score=analysis["final_score"],
            sl=stop_loss,
            tp=take_profit,
            adx=analysis.get("adx", 0),
            regime=analysis.get("regime", "unknown"),
            dry_run=DRY_RUN,
        )

        # --- ML prediction log (Anforderung 7.1, 7.4, 8.4) ---
        try:
            ml_prediction_log.record(
                gate_decision, symbol,
                training_period=ml_gate.get_training_period(symbol),
            )
        except Exception as exc:
            logger.error(f"ML prediction log failed for {symbol}: {exc}")

        open_symbols.add(symbol)


def _send_daily_summary():
    today = datetime.now(timezone.utc).date().isoformat()
    trades_today = [
        trade for trade in get_all_trades(limit=200, dry_run=DRY_RUN)
        if (trade.get("closed_at") or "").startswith(today)
    ]
    total_funding = get_funding_summary(dry_run=DRY_RUN)
    open_funding = get_funding_summary(dry_run=DRY_RUN, open_only=True)
    notify_daily_summary(
        trades_today=trades_today,
        open_positions=_open_position_count(),
        daily_pnl=get_daily_pnl(dry_run=DRY_RUN),
        daily_funding=get_daily_funding(dry_run=DRY_RUN),
        total_funding=total_funding["amount"],
        open_funding=open_funding["amount"],
        total_funding_unknown=total_funding["unknown_trades"],
        open_funding_unknown=open_funding["unknown_trades"],
    )


def main():
    init_db()
    if not DRY_RUN:
        legacy = get_legacy_open_live_trades()
        if legacy:
            symbols = ", ".join(f"{trade['symbol']}#{trade['id']}" for trade in legacy)
            raise RuntimeError(
                "Live startup blocked: open legacy trades have no reliable quantity. "
                f"Reconcile them with Bitget first: {symbols}"
            )
    _banner()

    # --- ML-Gate Initialisierung (Anforderung 9.1, 12.3, 13.3) ---
    if ml_gate.ML_GATE_ENABLED:
        try:
            symbols = get_top_symbols(TOP_COINS_COUNT)
            ml_gate.init(symbols)
        except Exception as exc:
            logger.error(
                f"ML-Gate init failed, continuing without ML: {exc}",
                exc_info=True,
            )

    last_entry_scan = 0.0
    last_daily_summary = 0.0
    last_outcome_backfill = 0.0
    _OUTCOME_BACKFILL_INTERVAL = 3600  # once per hour
    while True:
        try:
            now = time.time()
            # Start reconciliation first, but never let its API latency or failure
            # delay safety-critical trailing and price exits.
            try:
                refresh_funding_cashflows(now, background=True)
            except Exception as exc:
                logger.error(f"Funding refresh could not start: {exc}", exc_info=True)

            # The new closed candle's stop must be active before checking its price.
            replayed_closes: dict[int, float] = {}
            try:
                replayed_closes = refresh_trailing_stops(now)
            except Exception as exc:
                logger.error(f"Trailing refresh failed: {exc}", exc_info=True)
            try:
                manage_open_trades()
            except Exception as exc:
                logger.error(f"Open-trade management failed: {exc}", exc_info=True)
            # Adds come after the stop and price checks, so a position that was
            # already due to close is never enlarged first.
            try:
                manage_pyramid_adds(replayed_closes)
            except Exception as exc:
                logger.error(f"Pyramid management failed: {exc}", exc_info=True)

            if (
                _entry_window_open(now)
                and now - last_entry_scan >= _ENTRY_RETRY_INTERVAL
            ):
                logger.info("--- entry scan ----------------------------------------------")
                scan_new_entries()
                last_entry_scan = time.time()

            # Secondary timeframe sleeve (e.g. 4H)
            if (
                SECONDARY_ENABLED
                and _secondary_entry_window_open(now)
                and now - last_entry_scan >= _ENTRY_RETRY_INTERVAL
            ):
                try:
                    scan_secondary_entries()
                except Exception as exc:
                    logger.error(f"Secondary scan failed: {exc}", exc_info=True)

            # --- ML outcome tracking (Anforderung 7.2) ---
            if (
                ml_gate.ML_GATE_ENABLED
                and now - last_outcome_backfill >= _OUTCOME_BACKFILL_INTERVAL
            ):
                try:
                    ml_prediction_log.backfill_outcomes(
                        timeframe_seconds=_timeframe_seconds(),
                    )
                except Exception as exc:
                    logger.error(
                        f"ML outcome backfill failed: {exc}", exc_info=True
                    )
                last_outcome_backfill = time.time()

            # Funding-Carry-Scan (hourly)
            if funding_carry.CARRY_ENABLED and funding_carry.should_check(now):
                try:
                    funding_carry.log_carry_scan(symbols)
                except Exception as exc:
                    logger.error(f"Carry scan failed: {exc}", exc_info=True)

            today_midnight = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp()
            if last_daily_summary < today_midnight:
                _send_daily_summary()
                last_daily_summary = time.time()
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as exc:
            logger.error(f"Unhandled error: {exc}", exc_info=True)

        time.sleep(_POSITION_CHECK_INTERVAL)


if __name__ == "__main__":
    main()
