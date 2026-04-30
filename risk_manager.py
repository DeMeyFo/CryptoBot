from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT, LEVERAGE


def calculate_sl_tp(side: str, entry_price: float,
                    sl_pct: float = None, tp_pct: float = None) -> tuple[float, float]:
    sl_pct = sl_pct or STOP_LOSS_PCT
    tp_pct = tp_pct or TAKE_PROFIT_PCT

    if side == "long":
        stop_loss = entry_price * (1 - sl_pct)
        take_profit = entry_price * (1 + tp_pct)
    else:
        stop_loss = entry_price * (1 + sl_pct)
        take_profit = entry_price * (1 - tp_pct)

    return round(stop_loss, 8), round(take_profit, 8)


def check_exit(trade: dict, current_price: float) -> str | None:
    """Return 'closed_tp', 'closed_sl', or None."""
    side = trade["side"]
    sl = trade.get("stop_loss")
    tp = trade.get("take_profit")

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
    side = trade["side"]
    entry = trade["entry_price"]
    size = trade["size_usdt"]
    lev = trade["leverage"]

    pct = (current_price - entry) / entry if side == "long" else (entry - current_price) / entry
    return round(size * pct * lev, 4)
