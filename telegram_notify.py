"""Telegram push notifications - non-blocking and optional."""

import logging
import threading
from datetime import datetime, timezone

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def _send(text: str) -> None:
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
        if response.status_code != 200:
            logger.debug(f"Telegram {response.status_code}: {response.text[:120]}")
    except Exception as exc:
        logger.debug(f"Telegram send error: {exc}")


def notify(text: str) -> None:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    threading.Thread(target=_send, args=(text,), daemon=True).start()


def _funding_text(value: float | None, unknown_trades: int = 0) -> str:
    if value is None:
        return "unbekannt"
    amount = float(value)
    if unknown_trades and amount == 0:
        return f"unbekannt ({unknown_trades} Legacy-Trades)"
    if unknown_trades:
        return f"{amount:+.2f} USDT (teilweise; {unknown_trades} unbekannt)"
    return f"{amount:+.2f} USDT"


def notify_trade_opened(symbol: str, side: str, price: float,
                        size: float, score: float, sl: float,
                        tp: float | None, adx: float, regime: str,
                        dry_run: bool) -> None:
    icon = "UP" if side == "long" else "DOWN"
    label = "[DRY-RUN] " if dry_run else ""
    target = f"<code>{tp:.6g}</code>" if tp else "ATR trailing (kein fixes TP)"
    notify(
        f"{label}<b>TRADE OPENED</b>\n"
        f"{icon} {side.upper()} <b>{symbol}</b>\n"
        f"Size: {size:.2f} USDT | Score: {score:+.1f}\n"
        f"Entry: <code>{price:.6g}</code>\n"
        f"TP: {target} | SL: <code>{sl:.6g}</code>\n"
        f"ADX: {adx:.1f} | Regime: {regime}\n"
        "Funding: wird bei jedem Settlement separat verbucht"
    )


def notify_trade_closed(symbol: str, side: str, reason: str,
                        exit_price: float, pnl: float, dry_run: bool,
                        funding_usdt: float | None = None) -> None:
    labels = {
        "closed_tp": "TAKE PROFIT",
        "closed_sl": "STOP / TRAIL",
        "closed_claude_exit": "CLAUDE EMERGENCY EXIT",
    }
    prefix = "[DRY-RUN] " if dry_run else ""
    pnl_text = (
        f"Net PnL after modeled fees and reconciled funding: "
        f"<b>{pnl:+.2f} USDT</b>"
        if funding_usdt is not None
        else f"Provisional PnL after modeled fees (funding pending): "
             f"<b>{pnl:+.2f} USDT</b>"
    )
    notify(
        f"{prefix}<b>{labels.get(reason, reason.upper())}</b>\n"
        f"{side.upper()} <b>{symbol}</b>\n"
        f"{pnl_text}\n"
        f"Funding dieses Trades: <b>{_funding_text(funding_usdt)}</b>\n"
        f"Exit: <code>{exit_price:.6g}</code>"
    )


def notify_pyramid_add(symbol: str, side: str, add_number: int, max_adds: int,
                       price: float, added_notional: float,
                       total_notional: float, avg_entry: float,
                       reached_r: float, dry_run: bool) -> None:
    label = "[DRY-RUN] " if dry_run else ""
    notify(
        f"{label}<b>PYRAMID ADD {add_number}/{max_adds}</b>\n"
        f"{side.upper()} <b>{symbol}</b> bei {reached_r:.2f}R\n"
        f"Nachkauf: {added_notional:.2f} USDT @ <code>{price:.6g}</code>\n"
        f"Position jetzt: {total_notional:.2f} USDT\n"
        f"Durchschnittseinstieg: <code>{avg_entry:.6g}</code>\n"
        "Stop bleibt gemeinsam für alle Einheiten"
    )


def notify_claude_rejected(symbol: str, action: str, reason: str) -> None:
    notify(
        f"<b>CLAUDE REJECTED TRADE</b>\n"
        f"{action.upper()} {symbol}\n{reason}"
    )


def notify_daily_loss_limit(daily_pnl: float, limit: float) -> None:
    notify(
        f"<b>DAILY LOSS LIMIT HIT</b>\n"
        f"Net PnL today after modeled fees and funding: "
        f"<b>{daily_pnl:+.2f} USDT</b>\n"
        f"Limit: -{limit:.2f} USDT\nNo new entries until UTC midnight"
    )


def notify_daily_summary(trades_today: list, open_positions: int,
                         daily_pnl: float, daily_funding: float | None = None,
                         total_funding: float | None = None,
                         open_funding: float | None = None,
                         total_funding_unknown: int = 0,
                         open_funding_unknown: int = 0) -> None:
    wins = sum(1 for trade in trades_today if (trade.get("pnl_usdt") or 0) > 0)
    losses = len(trades_today) - wins
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    notify(
        f"<b>DAILY SUMMARY {date}</b>\n"
        f"Closed: {len(trades_today)} (wins {wins}, losses {losses})\n"
        f"Net PnL after modeled fees and funding: <b>{daily_pnl:+.2f} USDT</b>\n"
        f"Funding heute: <b>{_funding_text(daily_funding)}</b>\n"
        f"Funding erfasst gesamt: "
        f"<b>{_funding_text(total_funding, total_funding_unknown)}</b>\n"
        f"Funding erfasst offene Trades: "
        f"<b>{_funding_text(open_funding, open_funding_unknown)}</b>\n"
        f"Open positions: {open_positions}"
    )


def notify_bot_started(mode: str, leverage: int, sizing: str,
                       direction: str, exposure: str) -> None:
    notify(
        f"<b>Bot started - {mode}</b>\n"
        f"Leverage: {leverage}x | Direction: {direction}\n"
        f"Sizing: {sizing}\n"
        f"Exposure ceiling: {exposure}\n"
        "PnL accounting: modeled fees plus separately reconciled funding"
    )
