import logging
import sys
import time

from config import (
    DRY_RUN, LEVERAGE, POSITION_SIZE_USDT,
    MIN_POSITION_USDT, MAX_POSITION_USDT, HIGH_CONVICTION_SCORE,
    TOP_COINS_COUNT, LOOP_INTERVAL_SECONDS, MAX_OPEN_POSITIONS,
    LONG_THRESHOLD, TRAILING_STOP_PCT, CANDLE_INTERVAL,
    DAILY_LOSS_LIMIT_USDT,
)
from bitget_client import get_top_symbols, get_current_price, place_order, close_order
from strategy import analyze_symbol
from risk_manager import best_sl_tp, calculate_atr_sl_tp, progress_to_tp, check_exit, unrealized_pnl
from news_sentiment import validate_trade, get_claude_exit_signals
from database import (
    init_db, save_trade, close_trade, get_open_trades,
    update_trade_sl, increment_pyramid_count, get_latest_signal_score,
    get_daily_pnl,
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

# Pyramiding config
_PYRAMID_PROGRESS_THRESHOLD = 0.50
_PYRAMID_SCORE_THRESHOLD    = 70.0
_PYRAMID_SIZE_FACTOR        = 0.50
_PYRAMID_ATR_SL_MULT        = 1.0
_PYRAMID_ATR_TP_MULT        = 2.5


_POSITION_CHECK_INTERVAL = 60    # trailing stop + SL/TP check every 60s
_ENTRY_SCAN_INTERVAL     = LOOP_INTERVAL_SECONDS  # new entries every 5min


def _banner():
    mode = "DRY-RUN" if DRY_RUN else "LIVE"
    logger.info("=" * 60)
    logger.info(f"  Crypto Futures Bot  –  {mode}")
    logger.info(f"  Leverage: {LEVERAGE}x  │  Positions: {MIN_POSITION_USDT}–{MAX_POSITION_USDT} USDT")
    logger.info(f"  Max open: {MAX_OPEN_POSITIONS}  │  Trailing: {TRAILING_STOP_PCT*100:.1f}%  │  Candles: {CANDLE_INTERVAL}")
    logger.info(f"  Position check: {_POSITION_CHECK_INTERVAL}s  │  Entry scan: {_ENTRY_SCAN_INTERVAL}s")
    logger.info("=" * 60)


# ── Dynamic position sizing ───────────────────────────────────────────────────

def _position_size(score: float) -> float:
    """
    Scale position size by signal conviction:
      score ≥ HIGH_CONVICTION_SCORE  → MAX_POSITION_USDT
      score between threshold and HC  → POSITION_SIZE_USDT (default)
      score just above threshold       → MIN_POSITION_USDT
    """
    abs_score = abs(score)
    if abs_score >= HIGH_CONVICTION_SCORE:
        return MAX_POSITION_USDT
    mid = (LONG_THRESHOLD + HIGH_CONVICTION_SCORE) / 2
    if abs_score >= mid:
        return POSITION_SIZE_USDT
    return MIN_POSITION_USDT


# ── Trailing stop ─────────────────────────────────────────────────────────────

def _update_trailing_stop(trade: dict, current_price: float):
    """
    Ratchet the stop-loss upward (long) or downward (short) as price moves in favor.
    The SL trails TRAILING_STOP_PCT below/above the current price.
    Only updates when the new SL is better than the existing one.
    """
    side     = trade["side"]
    cur_sl   = trade.get("stop_loss") or 0.0

    if side == "long":
        new_sl = round(current_price * (1 - TRAILING_STOP_PCT), 8)
        if new_sl > cur_sl:
            update_trade_sl(trade["id"], new_sl)
            logger.debug(
                f"  TRAIL LONG  {trade['symbol']}  SL {cur_sl:.6g} → {new_sl:.6g}"
            )
    else:
        new_sl = round(current_price * (1 + TRAILING_STOP_PCT), 8)
        if cur_sl == 0.0 or new_sl < cur_sl:
            update_trade_sl(trade["id"], new_sl)
            logger.debug(
                f"  TRAIL SHORT {trade['symbol']}  SL {cur_sl:.6g} → {new_sl:.6g}"
            )


# ── Pyramiding ────────────────────────────────────────────────────────────────

def _try_pyramid(trade: dict, current_price: float):
    if trade.get("pyramid_count", 0) >= 1:
        return
    if trade.get("is_pyramid"):
        return

    prog = progress_to_tp(trade, current_price)
    if prog < _PYRAMID_PROGRESS_THRESHOLD:
        return

    score = get_latest_signal_score(trade["symbol"])
    if abs(score) < _PYRAMID_SCORE_THRESHOLD:
        return

    side   = trade["side"]
    symbol = trade["symbol"]
    entry  = trade["entry_price"]

    update_trade_sl(trade["id"], entry)
    logger.info(
        f"  △ PYRAMID trigger  {side.upper()} {symbol}"
        f"  progress={prog:.0%}  score={score:+.1f}"
        f"  → SL moved to break-even @ {entry:.6g}"
    )

    atr = trade.get("indicators", {}).get("atr") if isinstance(trade.get("indicators"), dict) else None
    if atr and atr > 0:
        sl, tp = calculate_atr_sl_tp(side, current_price, atr,
                                     sl_mult=_PYRAMID_ATR_SL_MULT,
                                     tp_mult=_PYRAMID_ATR_TP_MULT)
    else:
        sl, tp = best_sl_tp(side, current_price)

    add_on_size = round(POSITION_SIZE_USDT * _PYRAMID_SIZE_FACTOR, 2)
    order = place_order(symbol, side, add_on_size, LEVERAGE, current_price)
    if not order:
        logger.warning(f"  △ PYRAMID order failed for {symbol}")
        return

    pyramid_id = save_trade(
        symbol=symbol, side=side,
        entry_price=current_price, size_usdt=add_on_size, leverage=LEVERAGE,
        stop_loss=sl, take_profit=tp, dry_run=DRY_RUN,
        signal_score=score, order_id=order.get("orderId"),
        is_pyramid=True, parent_trade_id=trade["id"],
    )
    increment_pyramid_count(trade["id"])

    prefix = "[DRY-RUN] " if DRY_RUN else ""
    logger.info(
        f"  {prefix}△ PYRAMID ADD-ON  {side.upper()} {symbol}"
        f"  @ {current_price:.6g}  size={add_on_size} USDT"
        f"  SL={sl:.6g}  TP={tp:.6g}  id={pyramid_id}"
    )


# ── Trade management ──────────────────────────────────────────────────────────

def manage_open_trades():
    # Claude exit signals – piggybacked on the 15-min news refresh (no extra cost)
    claude_exits = get_claude_exit_signals()

    for trade in get_open_trades(dry_run=DRY_RUN):
        price = get_current_price(trade["symbol"])
        if price == 0:
            continue

        coin = trade["symbol"].replace("USDT", "")

        # ── Claude emergency exit ─────────────────────────────────────────────
        if claude_exits.get(coin) or claude_exits.get(trade["symbol"]):
            pnl_now = unrealized_pnl(trade, price)
            close_order(trade["symbol"], trade["side"], trade["entry_price"], trade["size_usdt"])
            realised = close_trade(trade["id"], price, "closed_claude_exit")
            logger.warning(
                f"🤖 CLAUDE EXIT  {trade['side'].upper()} {trade['symbol']}"
                f"  exit={price:.6g}  PnL={realised:+.2f} USDT  (emergency news signal)"
            )
            continue

        pnl    = unrealized_pnl(trade, price)
        reason = check_exit(trade, price)

        if reason:
            close_order(trade["symbol"], trade["side"], trade["entry_price"], trade["size_usdt"])
            realised = close_trade(trade["id"], price, reason)
            icon = "✅" if reason == "closed_tp" else "🛑"
            logger.info(
                f"{icon} {reason.upper():12s}  {trade['side'].upper()} {trade['symbol']}"
                f"  exit={price:.6g}  PnL={realised:+.2f} USDT"
            )
        else:
            logger.debug(
                f"  HOLD  {trade['side'].upper()} {trade['symbol']}"
                f"  entry={trade['entry_price']:.6g}  now={price:.6g}"
                f"  uPnL={pnl:+.2f} USDT"
            )
            _update_trailing_stop(trade, price)
            _try_pyramid(trade, price)


# ── New entry scanning ────────────────────────────────────────────────────────

def _open_position_count() -> int:
    return sum(1 for t in get_open_trades(dry_run=DRY_RUN) if not t.get("is_pyramid"))


def _daily_loss_limit_hit() -> bool:
    """Circuit breaker: block new entries if today's realised loss exceeds the limit."""
    daily_pnl = get_daily_pnl()
    if daily_pnl < -DAILY_LOSS_LIMIT_USDT:
        logger.warning(
            f"🛑 Daily loss limit hit: {daily_pnl:+.2f} USDT "
            f"(limit −{DAILY_LOSS_LIMIT_USDT:.0f} USDT) – no new entries today"
        )
        return True
    return False


def scan_new_entries():
    if _open_position_count() >= MAX_OPEN_POSITIONS:
        logger.info(f"Max positions reached ({MAX_OPEN_POSITIONS}) – skipping scan")
        return

    if _daily_loss_limit_hit():
        return

    symbols = get_top_symbols(TOP_COINS_COUNT)
    if not symbols:
        logger.warning("Could not fetch top symbols")
        return

    logger.info(f"Scanning {len(symbols)} symbols: {', '.join(symbols)}")
    open_syms = {t["symbol"] for t in get_open_trades(dry_run=DRY_RUN)}

    for symbol in symbols:
        if symbol in open_syms:
            continue
        if _open_position_count() >= MAX_OPEN_POSITIONS:
            break

        analysis = analyze_symbol(symbol)
        if analysis["action"] == "hold":
            continue

        price = get_current_price(symbol)
        if price == 0:
            logger.warning(f"No price for {symbol}")
            continue

        action   = analysis["action"]
        pos_size = _position_size(analysis["final_score"])

        # ── Claude trade validation ───────────────────────────────────────────
        approved, reason = validate_trade(symbol, action, analysis)
        if not approved:
            logger.info(f"  🚫 Claude rejected {action.upper()} {symbol}: {reason}")
            continue

        order = place_order(symbol, action, pos_size, LEVERAGE, price)
        if not order:
            continue

        sl, tp = best_sl_tp(action, price, atr=analysis.get("atr"))
        trade_id = save_trade(
            symbol=symbol, side=action,
            entry_price=price, size_usdt=pos_size, leverage=LEVERAGE,
            stop_loss=sl, take_profit=tp, dry_run=DRY_RUN,
            signal_score=analysis["final_score"], order_id=order.get("orderId"),
        )
        prefix = "[DRY-RUN] " if DRY_RUN else ""
        logger.info(
            f"{prefix}OPEN {action.upper():5s} {symbol}  @ {price:.6g}"
            f"  size={pos_size} USDT  SL={sl:.6g}  TP={tp:.6g}"
            f"  score={analysis['final_score']:+.1f}  ADX={analysis.get('adx', 0):.1f}"
            f"  trade_id={trade_id}"
        )
        open_syms.add(symbol)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    _banner()
    init_db()

    last_entry_scan = 0.0

    while True:
        try:
            now = time.time()

            # ── Trailing stop + SL/TP check – every 60s ──────────────────────
            manage_open_trades()

            # ── New entry scan – every 5 min ──────────────────────────────────
            if now - last_entry_scan >= _ENTRY_SCAN_INTERVAL:
                logger.info("─── entry scan ────────────────────────────────────")
                scan_new_entries()
                last_entry_scan = time.time()

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as exc:
            logger.error(f"Unhandled error: {exc}", exc_info=True)

        time.sleep(_POSITION_CHECK_INTERVAL)


if __name__ == "__main__":
    main()
