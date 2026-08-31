"""
CryptoBot Dashboard — Streamlit-basierte Uebersicht aller Bot-Aktivitaeten.

Starten mit: streamlit run dashboard.py
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bitget_client import get_account_info, get_current_price
from config import (
    ADX_ENTRY_THRESHOLD,
    DAILY_LOSS_LIMIT_PCT,
    DAILY_LOSS_LIMIT_USDT,
    DRY_RUN,
    ENTRY_ORDER_TYPE,
    LEVERAGE,
    LIMIT_PULLBACK_ATR,
    MAX_GROSS_EXPOSURE_PCT,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT,
    MAX_POSITION_USDT,
    MAX_SYMBOL_EXPOSURE_PCT,
    POSITION_SIZE_USDT,
    RISK_PER_TRADE_PCT,
    SECONDARY_ENABLED,
    SECONDARY_RISK_PER_TRADE_PCT,
    SECONDARY_TIMEFRAME,
    SIZING_MODE,
    STRATEGY_TIMEFRAME,
    STRATEGY_VERSION,
    TRADE_DIRECTION,
    TREND_STOP_ATR_MULTIPLIER,
    TREND_TRAIL_ATR_MULTIPLIER,
    VOL_ADAPTIVE_ENABLED,
    VOL_ADAPTIVE_HIGH_MULT,
    VOL_ADAPTIVE_LOOKBACK,
    VOL_ADAPTIVE_LOW_MULT,
)
from database import (
    get_all_trades,
    get_latest_signals,
    get_open_trades,
    get_pnl_history,
    get_funding_summary,
    get_total_pnl,
    init_db,
)
from risk_manager import unrealized_pnl

st.set_page_config(
    page_title="CryptoBot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Farbpalette ───────────────────────────────────────────────────────────
GREEN = "#00cc88"
RED = "#ff4444"
BLUE = "#5b8def"
YELLOW = "#ffaa00"
GRAY = "#888888"


# ── Sidebar ───────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.title("⚙ Konfiguration")
    mode = "🟢 DRY-RUN" if DRY_RUN else "🔴 LIVE"
    st.sidebar.markdown(f"### {mode}")
    st.sidebar.markdown(f"**Strategie:** {STRATEGY_VERSION}")

    with st.sidebar.expander("Trading-Parameter", expanded=False):
        st.markdown(f"- **Richtung:** {TRADE_DIRECTION}")
        st.markdown(f"- **Timeframe:** {STRATEGY_TIMEFRAME}")
        st.markdown(f"- **Risiko/Trade:** {RISK_PER_TRADE_PCT*100:.1f}%")
        st.markdown(f"- **ADX-Gate:** >= {ADX_ENTRY_THRESHOLD:.0f}")
        st.markdown(f"- **Max Positionen:** {MAX_OPEN_POSITIONS}")
        st.markdown(f"- **Hebel:** {LEVERAGE}x")
        st.markdown(f"- **Entry:** {ENTRY_ORDER_TYPE.upper()}")
        if ENTRY_ORDER_TYPE == "limit":
            st.markdown(f"- **Pullback:** {LIMIT_PULLBACK_ATR} ATR")

    with st.sidebar.expander("Risiko-Limits", expanded=False):
        st.markdown(f"- **Exposure/Symbol:** {MAX_SYMBOL_EXPOSURE_PCT}x")
        st.markdown(f"- **Exposure brutto:** {MAX_GROSS_EXPOSURE_PCT}x")
        if SIZING_MODE == "percent":
            st.markdown(f"- **Tages-Stop:** {DAILY_LOSS_LIMIT_PCT*100:.1f}% Equity")
        else:
            st.markdown(f"- **Tages-Stop:** {DAILY_LOSS_LIMIT_USDT:.0f} USDT")
        st.markdown(f"- **Stop:** {TREND_STOP_ATR_MULTIPLIER:.1f} ATR")
        st.markdown(f"- **Trail:** {TREND_TRAIL_ATR_MULTIPLIER:.1f} ATR ab 1R")

    with st.sidebar.expander("Erweiterungen", expanded=False):
        if SECONDARY_ENABLED:
            st.markdown(f"- **4H-Sleeve:** ✅ ({SECONDARY_RISK_PER_TRADE_PCT*100:.2f}%)")
        else:
            st.markdown("- **4H-Sleeve:** ❌")
        if VOL_ADAPTIVE_ENABLED:
            st.markdown(
                f"- **Vol-Adaptiv:** ✅ ({VOL_ADAPTIVE_LOW_MULT}x/{VOL_ADAPTIVE_HIGH_MULT}x)"
            )
        else:
            st.markdown("- **Vol-Adaptiv:** ❌")

    st.sidebar.divider()
    refresh = st.sidebar.checkbox("Auto-Refresh (30s)", value=True)
    if st.sidebar.button("🔄 Jetzt aktualisieren"):
        st.rerun()
    return refresh


# ── Kennzahlen-Header ────────────────────────────────────────────────────

def section_kpi():
    open_trades = get_open_trades(dry_run=DRY_RUN)
    all_trades = get_all_trades(500, dry_run=DRY_RUN)
    closed = [t for t in all_trades if t["status"] != "open"]
    total_pnl = get_total_pnl(dry_run=DRY_RUN)
    total_funding = get_funding_summary(dry_run=DRY_RUN)
    account = get_account_info()
    equity = float(account.get("equity", 0))

    wins = [t for t in closed if (t.get("pnl_usdt") or 0) > 0]
    losses = [t for t in closed if (t.get("pnl_usdt") or 0) <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0
    avg_win = np.mean([float(t["pnl_usdt"]) for t in wins]) if wins else 0
    avg_loss = np.mean([float(t["pnl_usdt"]) for t in losses]) if losses else 0
    pf = abs(sum(float(t["pnl_usdt"]) for t in wins)) / abs(sum(float(t["pnl_usdt"]) for t in losses)) if losses and sum(float(t["pnl_usdt"]) for t in losses) != 0 else 0

    # Exposure
    gross_exposure = sum(float(t.get("size_usdt", 0)) for t in open_trades)
    exposure_pct = gross_exposure / equity * 100 if equity > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Equity", f"{equity:,.2f} USDT")
    c2.metric("📊 Netto-P&L", f"{total_pnl:+,.2f} USDT",
              delta=f"Funding: {total_funding['amount']:+.2f}")
    c3.metric("📈 Offene Positionen", f"{len(open_trades)}/{MAX_OPEN_POSITIONS}",
              delta=f"Exposure: {exposure_pct:.0f}%")
    c4.metric("🎯 Win-Rate", f"{win_rate:.1f}%",
              delta=f"{len(closed)} Trades")
    c5.metric("⚖ Profit-Faktor", f"{pf:.2f}" if pf > 0 else "—",
              delta=f"Ø Win: {avg_win:+.2f} / Loss: {avg_loss:+.2f}")
    c6.metric("💹 Funding (gesamt)",
              f"{total_funding['amount']:+.2f} USDT",
              delta="vollständig" if total_funding["complete"] else "teilweise")


# ── Offene Positionen ─────────────────────────────────────────────────────

def section_positions():
    st.subheader("📋 Offene Positionen")
    trades = get_open_trades(dry_run=DRY_RUN)
    if not trades:
        st.info("Keine offenen Positionen.")
        return

    rows = []
    for t in trades:
        price = get_current_price(t["symbol"])
        upnl = unrealized_pnl(t, price) if price else 0
        funding = float(t.get("funding_usdt") or 0)
        rows.append({
            "Symbol": t["symbol"],
            "Seite": "🟢 LONG" if t["side"] == "long" else "🔴 SHORT",
            "Entry": float(t["entry_price"]),
            "Aktuell": price,
            "uPnL": upnl,
            "Funding": funding,
            "Gesamt": upnl + funding,
            "Stop": float(t.get("stop_loss") or 0),
            "Notional": float(t["size_usdt"]),
            "Seit": t["opened_at"][:16],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.map(
            lambda v: f"color: {GREEN}" if isinstance(v, (int, float)) and v > 0
            else f"color: {RED}" if isinstance(v, (int, float)) and v < 0
            else "",
            subset=["uPnL", "Funding", "Gesamt"],
        ),
        use_container_width=True, hide_index=True,
    )


# ── P&L-Kurve ────────────────────────────────────────────────────────────

def section_pnl_chart():
    st.subheader("📈 Kumulative P&L-Kurve")
    history = get_pnl_history(dry_run=DRY_RUN)
    if not history:
        st.info("Noch keine P&L-Events. Bot starten und warten.")
        return

    df = pd.DataFrame(history)
    df["cumulative"] = df["pnl_usdt"].cumsum()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cumulative"],
        mode="lines+markers",
        marker=dict(
            color=df["event_type"].map({"funding": BLUE, "close": GREEN}).fillna(GRAY),
            size=6,
        ),
        fill="tozeroy",
        fillcolor="rgba(0,204,136,0.1)" if df["cumulative"].iloc[-1] >= 0 else "rgba(255,68,68,0.1)",
        line=dict(color=GREEN if df["cumulative"].iloc[-1] >= 0 else RED, width=2),
        name="Kum. P&L",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="USDT", height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔵 Funding-Settlement | 🟢 Trade-Close")


# ── Monatsrenditen-Heatmap ────────────────────────────────────────────────

def section_monthly_heatmap():
    st.subheader("📅 Monatsrenditen")
    history = get_pnl_history(dry_run=DRY_RUN)
    if not history:
        return

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly = df.groupby("month")["pnl_usdt"].sum()

    if monthly.empty:
        return

    months_df = pd.DataFrame({
        "Monat": [str(m) for m in monthly.index],
        "P&L (USDT)": monthly.values,
    })

    colors = [GREEN if v >= 0 else RED for v in months_df["P&L (USDT)"]]
    fig = go.Figure(go.Bar(
        x=months_df["Monat"],
        y=months_df["P&L (USDT)"],
        marker_color=colors,
        text=[f"{v:+.2f}" for v in months_df["P&L (USDT)"]],
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="USDT", height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    pos = int((monthly > 0).sum())
    total = len(monthly)
    st.caption(f"✅ {pos}/{total} positive Monate ({pos/total*100:.0f}%)")


# ── Live-Sentiment ────────────────────────────────────────────────────────

def _sentiment_bar(value: float, vmin: float = -1, vmax: float = 1) -> str:
    """Erzeugt einen visuellen Balken fuer einen Sentiment-Wert."""
    norm = max(0.0, min(1.0, (value - vmin) / (vmax - vmin))) if vmax > vmin else 0.5
    blocks = int(norm * 10)
    if value > 0.05:
        return "🟢" * min(blocks, 10) + "⚪" * (10 - min(blocks, 10))
    elif value < -0.05:
        return "🔴" * min(10 - blocks, 10) + "⚪" * (10 - min(10 - blocks, 10))
    return "⚪" * 10


def _sentiment_label(value: float, neutral: float = 0.0) -> str:
    """Kurzes Label: bullish/bearish/neutral."""
    if value > neutral + 0.1:
        return "bullish"
    elif value < neutral - 0.1:
        return "bearish"
    return "neutral"


def section_sentiment():
    st.subheader("🌡 Live-Marktsentiment")
    try:
        from market_sentiment import get_features
        all_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                       "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"]

        # Spaltendefinitionen: (key, kurzer Header, Tooltip)
        cols_def = [
            ("ls_account_ratio",  "L/S Konten",
             "Long/Short Konten-Ratio. >1 = mehr Konten long (Crowd bullish)"),
            ("ls_position_ratio", "L/S Positionen",
             "Volumengewichtetes L/S-Verhaeltnis. >1 = mehr Volumen long"),
            ("taker_buy_ratio",   "Taker Kauf%",
             "Anteil aggressiver Kaeufer (0-100%). >50% = Kaufdruck"),
            ("taker_imbalance",   "Taker Netto",
             "Netto-Kaufdruck (-1 bis +1). >0 = Kaeufer dominieren"),
            ("book_imbalance_5",  "Buch Top5",
             "Bid/Ask-Imbalance Top 5 Levels. >0 = mehr Kaufliquiditaet"),
            ("book_imbalance_20", "Buch Top20",
             "Bid/Ask-Imbalance Top 20 Levels. >0 = breitere Kaufunterstuetzung"),
            ("oi_change_pct",     "OI Aend.",
             "Open-Interest-Veraenderung. >0 = neue Positionen werden eroeffnet"),
        ]

        rows = []
        for sym in all_symbols:
            try:
                data = get_features(sym)
                row = {"Symbol": sym}
                for key, header, _ in cols_def:
                    row[header] = data.get(key, 0.0)
                rows.append(row)
            except Exception:
                pass

        if not rows:
            st.warning("Keine Sentiment-Daten verfuegbar.")
            return

        df = pd.DataFrame(rows)

        # Tooltip-Header mit HTML
        header_html = "".join(
            f'<th title="{tip}" style="cursor:help;padding:4px 8px;'
            f'font-size:0.85em;white-space:nowrap">{h}</th>'
            for _, h, tip in cols_def
        )

        # Farbfunktion
        def _color_val(key, val):
            if key in ("taker_imbalance", "book_imbalance_5", "book_imbalance_20"):
                if val > 0.05: return GREEN
                if val < -0.05: return RED
            elif key == "oi_change_pct":
                if val > 0.001: return GREEN
                if val < -0.001: return RED
            elif key in ("ls_account_ratio", "ls_position_ratio"):
                if val > 1.05: return GREEN
                if val < 0.95: return RED
            elif key == "taker_buy_ratio":
                if val > 0.52: return GREEN
                if val < 0.48: return RED
            return GRAY

        def _fmt_val(key, val):
            if key == "taker_buy_ratio": return f"{val:.1%}"
            if key == "oi_change_pct": return f"{val:+.2%}"
            if key in ("taker_imbalance", "book_imbalance_5", "book_imbalance_20"):
                return f"{val:+.3f}"
            return f"{val:.3f}"

        # HTML-Tabelle fuer maximale Kontrolle
        table_rows = ""
        for row in rows:
            sym = row["Symbol"]
            cells = f'<td style="font-weight:bold;padding:4px 8px">{sym}</td>'
            for key, header, tip in cols_def:
                val = row[header]
                color = _color_val(key, val)
                display = _fmt_val(key, val)
                cells += (
                    f'<td title="{tip}" style="cursor:help;color:{color};'
                    f'padding:4px 8px;text-align:right;white-space:nowrap;'
                    f'font-weight:bold">{display}</td>'
                )
            table_rows += f"<tr>{cells}</tr>\n"

        html = f"""
        <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:0.9em">
        <thead><tr>
            <th style="padding:4px 8px;text-align:left">Symbol</th>
            {header_html}
        </tr></thead>
        <tbody>{table_rows}</tbody>
        </table>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        st.caption(
            "💡 Maus ueber Spaltenname fuer Erklaerung | "
            "🟢 bullish | 🔴 bearish | grau = neutral"
        )
    except Exception as e:
        st.warning(f"Sentiment nicht verfuegbar: {e}")


# ── Signal-Scanner ────────────────────────────────────────────────────────

def section_signals():
    st.subheader("📡 Letzte Signale")
    signals = get_latest_signals(20, dry_run=DRY_RUN, strategy_version=STRATEGY_VERSION)
    if not signals:
        st.info("Noch keine Signale.")
        return

    rows = []
    for s in signals:
        ind = json.loads(s["indicators"]) if s["indicators"] else {}
        action = s["action"].upper()
        if action == "HOLD":
            emoji = "⏸"
        elif action == "LONG":
            emoji = "🟢"
        else:
            emoji = "🔴"
        rows.append({
            "": emoji,
            "Symbol": s["symbol"],
            "Signal": action,
            "Score": f"{s['final_score']:+.0f}",
            "ADX": f"{ind.get('adx', 0):.1f}",
            "RSI": f"{ind.get('rsi', 0):.1f}",
            "Vol": f"{ind.get('volume_ratio', 0):.1f}x",
            "Filter": ind.get("filter_reason", "✅ setup complete"),
            "Zeit": (s.get("candle_timestamp") or s["timestamp"])[:16],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Trade-Historie ────────────────────────────────────────────────────────

def section_trades():
    st.subheader("📜 Trade-Historie")
    closed = [t for t in get_all_trades(100, dry_run=DRY_RUN) if t["status"] != "open"]
    if not closed:
        st.info("Noch keine abgeschlossenen Trades.")
        return

    rows = []
    for t in closed:
        pnl = float(t.get("pnl_usdt") or 0)
        rows.append({
            "": "✅" if pnl > 0 else "❌",
            "Symbol": t["symbol"],
            "Seite": "LONG" if t["side"] == "long" else "SHORT",
            "Entry": float(t["entry_price"]),
            "Exit": float(t["exit_price"]) if t["exit_price"] else 0,
            "P&L": pnl,
            "Funding": float(t.get("funding_usdt") or 0),
            "Status": t["status"].replace("closed_", "").upper(),
            "Geschlossen": (t["closed_at"] or "")[:16],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.map(
            lambda v: f"color: {GREEN}" if isinstance(v, (int, float)) and v > 0
            else f"color: {RED}" if isinstance(v, (int, float)) and v < 0
            else "",
            subset=["P&L", "Funding"],
        ),
        use_container_width=True, hide_index=True,
    )


# ── Backtest-Referenz ─────────────────────────────────────────────────────

def section_backtest_reference():
    st.subheader("📊 Backtest-Referenz (24 Monate)")
    st.caption("Vergangene Backtest-Ergebnisse garantieren keine zukünftige Performance.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monatsrendite", "+8,15%", help="Kompoundiert, Backtest")
    col2.metric("Max Drawdown", "13,8%")
    col3.metric("Sortino", "2,58")
    col4.metric("Median-Monat", "+4,06%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Equity 1k→", "6.057 USDT", delta="+505,7%")
    col6.metric("Profit-Faktor", "1,90")
    col7.metric("Pos. Monate", "52%")
    col8.metric("Trades (24M)", "282", help="207 auf 1H + 75 auf 4H")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    init_db()
    st.title("📈 CryptoBot Dashboard")
    st.markdown(
        f"**{STRATEGY_VERSION}** | "
        f"{'🟢 DRY-RUN' if DRY_RUN else '🔴 LIVE'} | "
        f"{TRADE_DIRECTION.upper()} | "
        f"{STRATEGY_TIMEFRAME}"
        + (f" + {SECONDARY_TIMEFRAME}" if SECONDARY_ENABLED else "")
        + f" | Risiko {RISK_PER_TRADE_PCT*100:.1f}%"
        + (f" | Limit-Entry" if ENTRY_ORDER_TYPE == "limit" else "")
        + (f" | Vol-Adaptiv" if VOL_ADAPTIVE_ENABLED else "")
    )

    auto_refresh = sidebar()

    # KPI Header
    section_kpi()
    st.divider()

    section_positions()
    st.divider()
    section_sentiment()
    st.divider()

    # Charts
    tab1, tab2 = st.tabs(["📈 P&L-Kurve", "📅 Monatsrenditen"])
    with tab1:
        section_pnl_chart()
    with tab2:
        section_monthly_heatmap()

    st.divider()

    # Signals + Trades
    tab3, tab4, tab5 = st.tabs(["📡 Signale", "📜 Trades", "📊 Backtest"])
    with tab3:
        section_signals()
    with tab4:
        section_trades()
    with tab5:
        section_backtest_reference()

    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
