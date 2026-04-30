import json
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DRY_RUN, LEVERAGE, POSITION_SIZE_USDT, MAX_OPEN_POSITIONS, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from database import (
    init_db, get_open_trades, get_all_trades,
    get_latest_signals, get_total_pnl, get_pnl_history,
)
from bitget_client import get_account_info, get_current_price
from risk_manager import unrealized_pnl

st.set_page_config(
    page_title="Crypto Futures Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.title("⚙️ Konfiguration")
    mode_color = "🔵" if DRY_RUN else "🔴"
    st.sidebar.markdown(f"**Modus:** {mode_color} {'DRY-RUN' if DRY_RUN else 'LIVE'}")
    st.sidebar.metric("Hebel", f"{LEVERAGE}x")
    st.sidebar.metric("Positionsgröße", f"{POSITION_SIZE_USDT} USDT")
    st.sidebar.metric("Max. Positionen", MAX_OPEN_POSITIONS)
    st.sidebar.metric("Stop-Loss", f"{STOP_LOSS_PCT*100:.1f}%")
    st.sidebar.metric("Take-Profit", f"{TAKE_PROFIT_PCT*100:.1f}%")

    st.sidebar.divider()
    refresh = st.sidebar.checkbox("Auto-Refresh (30 s)", value=True)
    if st.sidebar.button("🔄 Jetzt aktualisieren"):
        st.rerun()
    return refresh


# ── Helpers ────────────────────────────────────────────────────────────────────

def _side_badge(side: str) -> str:
    return "🟢 LONG" if side == "long" else "🔴 SHORT"


def _status_badge(status: str) -> str:
    return {"closed_tp": "✅ TP", "closed_sl": "🛑 SL", "closed_manual": "⚪ Manual"}.get(status, status)


# ── Sections ───────────────────────────────────────────────────────────────────

def section_metrics():
    open_trades = get_open_trades()
    all_trades = get_all_trades(500)
    closed = [t for t in all_trades if t["status"] != "open"]
    total_pnl = get_total_pnl()
    wins = [t for t in closed if (t.get("pnl_usdt") or 0) > 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0.0

    acc = get_account_info()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💼 Offene Positionen", len(open_trades))
    c2.metric("📊 Gesamt P&L", f"{total_pnl:+.2f} USDT")
    c3.metric("🏁 Abgeschlossen", len(closed))
    c4.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    c5.metric("💰 Kapital (equity)", f"{acc['equity']:.2f} USDT")


def section_open_positions():
    st.subheader("📂 Offene Positionen")
    trades = get_open_trades()
    if not trades:
        st.info("Keine offenen Positionen.")
        return

    rows = []
    for t in trades:
        price = get_current_price(t["symbol"])
        pnl = unrealized_pnl(t, price) if price else 0.0
        rows.append({
            "Symbol":       t["symbol"],
            "Seite":        _side_badge(t["side"]),
            "Entry":        f"{t['entry_price']:.6g}",
            "Aktuell":      f"{price:.6g}" if price else "–",
            "uPnL USDT":    f"{pnl:+.2f}",
            "Stop-Loss":    f"{t['stop_loss']:.6g}" if t["stop_loss"] else "–",
            "Take-Profit":  f"{t['take_profit']:.6g}" if t["take_profit"] else "–",
            "Größe USDT":   t["size_usdt"],
            "Hebel":        f"{t['leverage']}x",
            "Score":        f"{t['signal_score']:+.1f}" if t["signal_score"] else "–",
            "Geöffnet":     t["opened_at"][:16],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def section_signal_scanner():
    st.subheader("🔍 Signal Scanner")
    signals = get_latest_signals(20)
    if not signals:
        st.info("Noch keine Signale – Bot starten.")
        return

    rows = []
    for s in signals:
        ind = json.loads(s["indicators"]) if s["indicators"] else {}
        icon = {"long": "🟢", "short": "🔴", "hold": "⚪"}.get(s["action"], "⚪")
        rows.append({
            "Symbol":       s["symbol"],
            "Signal":       f"{icon} {s['action'].upper()}",
            "Score":        f"{s['final_score']:+.1f}",
            "TA":           f"{s['ta_score']:+.1f}",
            "News":         f"{s['news_score']:+.1f}",
            "RSI":          f"{ind.get('rsi', '–')}",
            "MACD Hist":    f"{ind.get('macd_hist', '–')}",
            "Vol Ratio":    f"{ind.get('volume_ratio', '–')}",
            "Zeitstempel":  s["timestamp"][:16],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def section_pnl_chart():
    st.subheader("📈 Kumulativer P&L")
    history = get_pnl_history()
    if not history:
        st.info("Noch keine abgeschlossenen Trades.")
        return

    df = pd.DataFrame(history)
    df["cumulative"] = df["pnl_usdt"].cumsum()
    df["closed_at"] = pd.to_datetime(df["closed_at"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["closed_at"],
        y=df["cumulative"],
        mode="lines+markers",
        fill="tozeroy",
        line=dict(color="#00cc88" if df["cumulative"].iloc[-1] >= 0 else "#ff4444", width=2),
        name="Kum. P&L",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="USDT",
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def section_trade_history():
    st.subheader("📋 Trade Historie")
    all_trades = get_all_trades(200)
    closed = [t for t in all_trades if t["status"] != "open"]
    if not closed:
        st.info("Noch keine abgeschlossenen Trades.")
        return

    rows = []
    for t in closed:
        pnl = t.get("pnl_usdt") or 0
        rows.append({
            "Symbol":       t["symbol"],
            "Seite":        _side_badge(t["side"]),
            "Status":       _status_badge(t["status"]),
            "Entry":        f"{t['entry_price']:.6g}",
            "Exit":         f"{t['exit_price']:.6g}" if t["exit_price"] else "–",
            "P&L USDT":     f"{pnl:+.2f}",
            "P&L %":        f"{t.get('pnl_pct') or 0:+.2f}%",
            "Score":        f"{t['signal_score']:+.1f}" if t["signal_score"] else "–",
            "Geöffnet":     t["opened_at"][:16],
            "Geschlossen":  t["closed_at"][:16] if t["closed_at"] else "–",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_db()

    mode_label = "🔵 DRY-RUN" if DRY_RUN else "🔴 LIVE"
    st.title(f"📈 Crypto Futures Bot  {mode_label}")

    auto_refresh = sidebar()

    section_metrics()
    st.divider()
    section_open_positions()
    st.divider()
    section_signal_scanner()
    st.divider()
    section_pnl_chart()
    st.divider()
    section_trade_history()

    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
