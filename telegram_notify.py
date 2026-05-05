"""
Telegram push notifications – non-blocking (background thread).
All functions are no-ops when TELEGRAM_BOT_TOKEN is not set.
"""

import logging
import threading
from datetime import datetime, timezone

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def _send(text: str) -> None:
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
        if resp.status_code != 200:
            logger.debug(f"Telegram {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        logger.debug(f"Telegram send error: {e}")


def notify(text: str) -> None:
    """Fire-and-forget Telegram message. Does nothing if token is not configured."""
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    threading.Thread(target=_send, args=(text,), daemon=True).start()


# ── Pre-formatted event notifications ─────────────────────────────────────────

def notify_trade_opened(symbol: str, side: str, price: float,
                        size: float, score: float, sl: float, tp: float,
                        adx: float, regime: str, dry_run: bool) -> None:
    icon  = "📈" if side == "long" else "📉"
    label = "[DRY-RUN] " if dry_run else ""
    notify(
        f"{label}🟢 <b>TRADE OPENED</b>\n"
        f"{icon} {side.upper()} <b>{symbol}</b>\n"
        f"💰 Size: {size:.0f} USDT  |  Score: {score:+.1f}\n"
        f"📌 Entry: <code>{price:.6g}</code>\n"
        f"🎯 TP: <code>{tp:.6g}</code>  |  🛑 SL: <code>{sl:.6g}</code>\n"
        f"📊 ADX: {adx:.1f}  |  Regime: {regime}"
    )


def notify_trade_closed(symbol: str, side: str, reason: str,
                        exit_price: float, pnl: float, dry_run: bool) -> None:
    if reason == "closed_tp":
        icon, label = "✅", "TAKE PROFIT"
    elif reason == "closed_sl":
        icon, label = "🛑", "STOP LOSS"
    elif reason == "closed_claude_exit":
        icon, label = "🤖", "CLAUDE EMERGENCY EXIT"
    else:
        icon, label = "⚪", reason.upper()

    direction = "📈" if side == "long" else "📉"
    pnl_str   = f"{pnl:+.2f} USDT"
    prefix    = "[DRY-RUN] " if dry_run else ""
    notify(
        f"{prefix}{icon} <b>{label}</b>\n"
        f"{direction} {side.upper()} <b>{symbol}</b>\n"
        f"💵 PnL: <b>{pnl_str}</b>\n"
        f"📌 Exit: <code>{exit_price:.6g}</code>"
    )


def notify_claude_rejected(symbol: str, action: str, reason: str) -> None:
    notify(
        f"🚫 <b>CLAUDE REJECTED TRADE</b>\n"
        f"❌ {action.upper()} {symbol}\n"
        f"💬 {reason}"
    )


def notify_daily_loss_limit(daily_pnl: float, limit: float) -> None:
    notify(
        f"🛑 <b>DAILY LOSS LIMIT HIT</b>\n"
        f"📉 Today's PnL: <b>{daily_pnl:+.2f} USDT</b>\n"
        f"⛔ Limit: -{limit:.0f} USDT\n"
        f"No new entries until tomorrow (UTC midnight)"
    )


def notify_daily_summary(trades_today: list, open_positions: int,
                         daily_pnl: float) -> None:
    wins   = sum(1 for t in trades_today if (t.get("pnl_usdt") or 0) > 0)
    losses = len(trades_today) - wins
    date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    notify(
        f"📊 <b>DAILY SUMMARY  {date}</b>\n"
        f"Closed trades: {len(trades_today)}  (✅ {wins}  🛑 {losses})\n"
        f"💵 PnL today: <b>{daily_pnl:+.2f} USDT</b>\n"
        f"📂 Open positions: {open_positions}"
    )


def notify_bot_started(mode: str, leverage: int,
                       min_pos: float, max_pos: float) -> None:
    notify(
        f"🚀 <b>Bot started  –  {mode}</b>\n"
        f"Leverage: {leverage}x  |  Size: {min_pos:.0f}–{max_pos:.0f} USDT"
    )
