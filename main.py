import logging
import sys
import time

from config import (
    DRY_RUN, LEVERAGE, POSITION_SIZE_USDT,
    TOP_COINS_COUNT, LOOP_INTERVAL_SECONDS, MAX_OPEN_POSITIONS,
)
from bitget_client import get_top_symbols, get_current_price, place_order, close_order
from strategy import analyze_symbol
from risk_manager import calculate_sl_tp, check_exit, unrealized_pnl
from database import init_db, save_trade, close_trade, get_open_trades

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def _banner():
    mode = "DRY-RUN" if DRY_RUN else "LIVE"
    logger.info("=" * 60)
    logger.info(f"  Crypto Futures Bot  –  {mode}")
    logger.info(f"  Leverage: {LEVERAGE}x  |  Position: {POSITION_SIZE_USDT} USDT")
    logger.info(f"  Max open positions: {MAX_OPEN_POSITIONS}")
    logger.info(f"  Scan interval: {LOOP_INTERVAL_SECONDS}s")
    logger.info("=" * 60)


def manage_open_trades():
    for trade in get_open_trades(dry_run=DRY_RUN):
        price = get_current_price(trade["symbol"])
        if price == 0:
            continue

        pnl = unrealized_pnl(trade, price)
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


def scan_new_entries():
    open_trades = get_open_trades(dry_run=DRY_RUN)
    if len(open_trades) >= MAX_OPEN_POSITIONS:
        logger.info(f"Max positions reached ({MAX_OPEN_POSITIONS}) – skipping scan")
        return

    symbols = get_top_symbols(TOP_COINS_COUNT)
    if not symbols:
        logger.warning("Could not fetch top symbols")
        return

    logger.info(f"Scanning {len(symbols)} symbols: {', '.join(symbols)}")
    open_syms = {t["symbol"] for t in open_trades}

    for symbol in symbols:
        if symbol in open_syms:
            continue
        if len(get_open_trades(dry_run=DRY_RUN)) >= MAX_OPEN_POSITIONS:
            break

        analysis = analyze_symbol(symbol)
        if analysis["action"] == "hold":
            continue

        price = get_current_price(symbol)
        if price == 0:
            logger.warning(f"No price for {symbol}")
            continue

        action = analysis["action"]
        order = place_order(symbol, action, POSITION_SIZE_USDT, LEVERAGE, price)
        if not order:
            continue

        sl, tp = calculate_sl_tp(action, price)
        trade_id = save_trade(
            symbol=symbol,
            side=action,
            entry_price=price,
            size_usdt=POSITION_SIZE_USDT,
            leverage=LEVERAGE,
            stop_loss=sl,
            take_profit=tp,
            dry_run=DRY_RUN,
            signal_score=analysis["final_score"],
            order_id=order.get("orderId"),
        )
        prefix = "[DRY-RUN] " if DRY_RUN else ""
        logger.info(
            f"{prefix}OPEN {action.upper():5s} {symbol}  @ {price:.6g}"
            f"  SL={sl:.6g}  TP={tp:.6g}  score={analysis['final_score']:+.1f}"
            f"  trade_id={trade_id}"
        )
        open_syms.add(symbol)


def main():
    _banner()
    init_db()

    while True:
        try:
            logger.info("─── cycle start ───────────────────────────────────")
            manage_open_trades()
            scan_new_entries()
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as exc:
            logger.error(f"Unhandled error: {exc}", exc_info=True)

        logger.info(f"Sleeping {LOOP_INTERVAL_SECONDS}s …")
        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
