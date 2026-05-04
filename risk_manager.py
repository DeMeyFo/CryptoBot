from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT, LEVERAGE, USE_ATR_SL_TP, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER


def calculate_sl_tp(side: str, entry_price: float,
                    sl_pct: float = None, tp_pct: float = None) -> tuple[float, float]:
    """Fixed-percentage SL/TP (fallback when ATR is unavailable)."""
    sl_pct = sl_pct or STOP_LOSS_PCT
    tp_pct = tp_pct or TAKE_PROFIT_PCT

    if side == "long":
        stop_loss   = entry_price * (1 - sl_pct)
        take_profit = entry_price * (1 + tp_pct)
    else:
        stop_loss   = entry_price * (1 + sl_pct)
        take_profit = entry_price * (1 - tp_pct)

    return round(stop_loss, 8), round(take_profit, 8)


def calculate_atr_sl_tp(side: str, entry_price: float, atr: float,
                         sl_mult: float = None, tp_mult: float = None) -> tuple[float, float]:
    """
    Volatility-adaptive SL/TP using ATR.
    Default: SL = 2× ATR, TP = 3.5× ATR (risk:reward ~1:1.75).
    """
    sl_mult = sl_mult if sl_mult is not None else ATR_SL_MULTIPLIER
    tp_mult = tp_mult if tp_mult is not None else ATR_TP_MULTIPLIER

    if side == "long":
        stop_loss   = entry_price - atr * sl_mult
        take_profit = entry_price + atr * tp_mult
    else:
        stop_loss   = entry_price + atr * sl_mult
        take_profit = entry_price - atr * tp_mult

    return round(stop_loss, 8), round(take_profit, 8)


def best_sl_tp(side: str, entry_price: float, atr: float | None = None) -> tuple[float, float]:
    """
    Choose ATR-based SL/TP when USE_ATR_SL_TP is enabled and ATR is available,
    otherwise fall back to fixed percentages.
    """
    if USE_ATR_SL_TP and atr and atr > 0:
        return calculate_atr_sl_tp(side, entry_price, atr)
    return calculate_sl_tp(side, entry_price)


def progress_to_tp(trade: dict, current_price: float) -> float:
    """
    How far the trade has moved towards its take-profit, as a fraction 0.0–1.0.
    Returns 0.0 if TP is not set or the trade is moving the wrong way.
    """
    side = trade["side"]
    entry = trade["entry_price"]
    tp    = trade.get("take_profit")
    if not tp:
        return 0.0

    if side == "long":
        total = tp - entry
        done  = current_price - entry
    else:
        total = entry - tp
        done  = entry - current_price

    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, done / total))


def check_exit(trade: dict, current_price: float) -> str | None:
    """Return 'closed_tp', 'closed_sl', or None."""
    side = trade["side"]
    sl   = trade.get("stop_loss")
    tp   = trade.get("take_profit")

    if side == "long":
        if tp and current_price >= tp:
            return "closed_tp"
        if sl and current_price <= sl:
            return "closed_sl"
    else:
        if tp and current_price <= tp:
            return "closed_tp"
        if sl and current_price >= sl:
            return "closed_sl"
    return None


def unrealized_pnl(trade: dict, current_price: float) -> float:
    side  = trade["side"]
    entry = trade["entry_price"]
    size  = trade["size_usdt"]
    lev   = trade["leverage"]

    pct = (current_price - entry) / entry if side == "long" else (entry - current_price) / entry
    return round(size * pct * lev, 4)
