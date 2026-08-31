from config import (
    ATR_TP_MULTIPLIER,
    MAX_GROSS_EXPOSURE_PCT,
    MAX_POSITION_PCT,
    MAX_POSITION_USDT,
    MAX_SYMBOL_EXPOSURE_PCT,
    MIN_POSITION_PCT,
    MIN_POSITION_USDT,
    POSITION_SIZE_USDT,
    RISK_PER_TRADE_PCT,
    SIZING_MODE,
    SLIPPAGE_RATE,
    STOP_LOSS_PCT,
    TAKER_FEE_RATE,
    TAKE_PROFIT_PCT,
    TRAILING_ACTIVATION_R,
    TREND_STOP_ATR_MULTIPLIER,
    TREND_TRAIL_ATR_MULTIPLIER,
    USE_FIXED_TAKE_PROFIT,
)


def effective_entry_price(trade: dict) -> float:
    """
    Entry price to use for profit and loss.

    `entry_price` deliberately keeps the original fill so the trailing stop and
    the R measurement retain their reference point. `avg_entry_price` carries
    the volume-weighted entry once pyramid adds exist, and is NULL otherwise.
    """
    average = trade.get("avg_entry_price")
    if average is not None and float(average) > 0:
        return float(average)
    return float(trade.get("entry_price") or 0)


def calculate_sl_tp(side: str, entry_price: float,
                    sl_pct: float | None = None,
                    tp_pct: float | None = None) -> tuple[float, float | None]:
    sl_pct = STOP_LOSS_PCT if sl_pct is None else sl_pct
    tp_pct = TAKE_PROFIT_PCT if tp_pct is None else tp_pct
    if side == "long":
        stop_loss = entry_price * (1 - sl_pct)
        take_profit = entry_price * (1 + tp_pct) if USE_FIXED_TAKE_PROFIT else None
    else:
        stop_loss = entry_price * (1 + sl_pct)
        take_profit = entry_price * (1 - tp_pct) if USE_FIXED_TAKE_PROFIT else None
    return round(stop_loss, 8), round(take_profit, 8) if take_profit else None


def calculate_atr_sl_tp(side: str, entry_price: float, atr: float,
                        sl_mult: float | None = None,
                        tp_mult: float | None = None) -> tuple[float, float | None]:
    sl_mult = TREND_STOP_ATR_MULTIPLIER if sl_mult is None else sl_mult
    tp_mult = ATR_TP_MULTIPLIER if tp_mult is None else tp_mult
    if side == "long":
        stop_loss = entry_price - atr * sl_mult
        take_profit = entry_price + atr * tp_mult if USE_FIXED_TAKE_PROFIT else None
    else:
        stop_loss = entry_price + atr * sl_mult
        take_profit = entry_price - atr * tp_mult if USE_FIXED_TAKE_PROFIT else None
    return round(stop_loss, 8), round(take_profit, 8) if take_profit else None


def best_sl_tp(side: str, entry_price: float,
               atr: float | None = None) -> tuple[float, float | None]:
    if atr is not None and atr > 0:
        return calculate_atr_sl_tp(side, entry_price, atr)
    return calculate_sl_tp(side, entry_price)


def position_notional_cap(equity: float) -> float:
    """Configured ceiling for one entry, before margin and exposure limits."""
    if SIZING_MODE == "percent":
        return max(0.0, equity * MAX_POSITION_PCT)
    return min(POSITION_SIZE_USDT, MAX_POSITION_USDT)


def position_notional_floor(equity: float) -> float:
    """Entries smaller than this are skipped rather than rounded up."""
    if SIZING_MODE == "percent":
        return max(0.0, equity * MIN_POSITION_PCT)
    return MIN_POSITION_USDT


def exposure_headroom(equity: float, symbol_exposure_usdt: float,
                      gross_exposure_usdt: float) -> float:
    """
    Remaining notional allowed by the per-symbol and gross exposure ceilings.

    Risk-based sizing bounds the loss only while the stop fills at its price.
    These ceilings bound the loss when it does not, which is the case that
    actually removes accounts: an adverse gap, a liquidity hole, or a bot
    process that died while the position was open.
    """
    headroom = float("inf")
    if MAX_SYMBOL_EXPOSURE_PCT > 0:
        headroom = min(
            headroom,
            equity * MAX_SYMBOL_EXPOSURE_PCT - max(0.0, symbol_exposure_usdt),
        )
    if MAX_GROSS_EXPOSURE_PCT > 0:
        headroom = min(
            headroom,
            equity * MAX_GROSS_EXPOSURE_PCT - max(0.0, gross_exposure_usdt),
        )
    return headroom


def calculate_position_notional(equity: float, entry_price: float,
                                stop_loss: float, max_notional: float,
                                symbol_exposure_usdt: float = 0.0,
                                gross_exposure_usdt: float = 0.0,
                                risk_scale: float = 1.0) -> float:
    """
    Size so stop loss plus modeled roundtrip costs remains within risk budget,
    then clamp the result to the configured exposure ceilings.

    Exposure is measured as entry notional, matching how it was validated in
    portfolio_backtest.py. A winning position's true market exposure is larger
    than its entry notional, so this understates it slightly while in profit.
    """
    if equity <= 0 or entry_price <= 0 or max_notional <= 0:
        return 0.0
    stop_fraction = abs(entry_price - stop_loss) / entry_price
    roundtrip_cost_fraction = 2 * (TAKER_FEE_RATE + SLIPPAGE_RATE)
    loss_fraction = stop_fraction + roundtrip_cost_fraction
    if loss_fraction <= 0:
        return 0.0

    ceiling = min(
        max_notional,
        exposure_headroom(equity, symbol_exposure_usdt, gross_exposure_usdt),
    )
    if ceiling <= 0:
        return 0.0

    risk_budget = equity * RISK_PER_TRADE_PCT * max(0.0, risk_scale)
    notional = min(ceiling, risk_budget / loss_fraction)
    if notional < position_notional_floor(equity):
        return 0.0
    return round(notional, 2)


def check_exit(trade: dict, current_price: float) -> str | None:
    side = trade["side"]
    stop = trade.get("stop_loss")
    target = trade.get("take_profit")
    if side == "long":
        if stop and current_price <= stop:
            return "closed_sl"
        if target and current_price >= target:
            return "closed_tp"
    else:
        if stop and current_price >= stop:
            return "closed_sl"
        if target and current_price <= target:
            return "closed_tp"
    return None


def next_atr_trailing_stop(trade: dict, candle_close: float, candle_high: float,
                           candle_low: float, atr: float) -> tuple[float, float, bool]:
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
        if reached_r < TRAILING_ACTIVATION_R:
            return current_stop, best_price, False
        candidate = candle_close - atr * TREND_TRAIL_ATR_MULTIPLIER
        new_stop = max(current_stop, candidate)
    else:
        best_price = min(previous_best, candle_low)
        reached_r = (entry - best_price) / initial_risk
        if reached_r < TRAILING_ACTIVATION_R:
            return current_stop, best_price, False
        candidate = candle_close + atr * TREND_TRAIL_ATR_MULTIPLIER
        new_stop = min(current_stop, candidate) if current_stop else candidate
    return round(new_stop, 8), round(best_price, 8), new_stop != current_stop


def unrealized_pnl(trade: dict, current_price: float) -> float:
    """Gross unrealized PnL; fees and funding are not knowable before exit."""
    side = trade["side"]
    entry = effective_entry_price(trade)
    quantity = float(trade.get("quantity") or 0)
    direction = 1.0 if side == "long" else -1.0
    if quantity > 0:
        return round((current_price - entry) * quantity * direction, 4)
    notional = float(trade["size_usdt"])
    return round(notional * ((current_price - entry) / entry) * direction, 4)
