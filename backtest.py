#!/usr/bin/env python3
"""
CryptoBot Backtest – Walk-forward simulation on historical OHLCV data.

Only the TA component is simulated (News / Fear&Greed / Funding have no
historical data). All indicators mirror the live bot exactly.

Usage:
  python backtest.py                            # whitelist, 15m, Bitget (10 days)
  python backtest.py --months 3                 # 3 months via Binance
  python backtest.py --months 6 --granularity 1H
  python backtest.py BTCUSDT SOLUSDT --months 3
  python backtest.py --trades                   # print every trade
"""

import argparse
import sys
import time as _time

import pandas as pd
import requests
import ta

from technical_analysis import fetch_ohlcv, _supertrend
from config import (
    LONG_THRESHOLD, SHORT_THRESHOLD,
    USE_ATR_SL_TP, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    LEVERAGE, POSITION_SIZE_USDT, TOP_COINS_COUNT,
    ADX_NO_TREND, ADX_WEAK_TREND, ADX_STRONG_TREND,
    TRAILING_STOP_PCT,
)
from bitget_client import get_top_symbols

WARMUP = 60  # candles needed before indicators are reliable

_BINANCE_BASE = "https://api.binance.com"
_BINANCE_INTERVAL_MAP = {"15m": "15m", "1H": "1h", "4H": "4h", "1D": "1d"}


def fetch_binance_ohlcv(symbol: str, interval: str = "15m", months: int = 3) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Binance public API – no API key required.
    Paginates automatically to cover the requested number of months.
    Falls back to an empty DataFrame if the symbol is not listed on Binance.
    """
    bi = _BINANCE_INTERVAL_MAP.get(interval, interval.lower())
    end_ms   = int(_time.time() * 1000)
    start_ms = end_ms - int(months * 30.44 * 24 * 3600 * 1000)

    candles = []
    cur = start_ms
    while cur < end_ms:
        try:
            resp = requests.get(
                f"{_BINANCE_BASE}/api/v3/klines",
                params={"symbol": symbol, "interval": bi,
                        "startTime": cur, "endTime": end_ms, "limit": 1000},
                timeout=10,
            )
            if resp.status_code != 200:
                break
            batch = resp.json()
            if not batch:
                break
            candles.extend(batch)
            if len(batch) < 1000:
                break
            cur = int(batch[-1][0]) + 1
        except Exception:
            break

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _load_ohlcv(symbol: str, granularity: str, limit: int, months: int) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, source_label). Uses Binance when months>0, else Bitget."""
    if months > 0:
        df = fetch_binance_ohlcv(symbol, granularity, months)
        if not df.empty:
            return df, "Binance"
        print(f"    (Binance has no data for {symbol}, falling back to Bitget)")
    df = fetch_ohlcv(symbol, granularity, limit)
    return df, "Bitget"


# ── Indicator pre-computation ─────────────────────────────────────────────────

def _precompute(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns in a single pass – no look-ahead bias."""
    df = df.copy()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    df["ema9"]       = ta.trend.ema_indicator(close, 9)
    df["ema21"]      = ta.trend.ema_indicator(close, 21)
    df["ema50"]      = ta.trend.ema_indicator(close, 50)
    df["rsi"]        = ta.momentum.RSIIndicator(close, 14).rsi()

    sr = ta.momentum.StochRSIIndicator(close, 14, smooth1=3, smooth2=3)
    df["stoch_k"]    = sr.stochrsi_k()
    df["stoch_d"]    = sr.stochrsi_d()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]       = macd.macd()
    df["macd_sig"]   = macd.macd_signal()
    df["macd_hist"]  = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_lower"]   = bb.bollinger_lband()
    df["bb_upper"]   = bb.bollinger_hband()

    df["atr"]        = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["vol_sma"]    = ta.trend.sma_indicator(volume, window=20)
    df["supertrend"] = _supertrend(df, period=10, multiplier=3.0)
    df["adx"]        = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    return df


def _score_at(df: pd.DataFrame, i: int) -> tuple[int, float]:
    """Return (TA score, ATR) at row i using pre-computed columns."""
    r    = df.iloc[i]
    prev = df.iloc[i - 1]
    score = 0

    # ── EMA crossover (±20) ───────────────────────────────────────────────────
    v9, v21, v50 = r["ema9"], r["ema21"], r["ema50"]
    if not any(pd.isna([v9, v21, v50])):
        if v9 > v21 > v50:   score += 20
        elif v9 > v21:        score += 10
        elif v9 < v21 < v50: score -= 20
        elif v9 < v21:        score -= 10

    # ── RSI (±15) ─────────────────────────────────────────────────────────────
    rsi = r["rsi"]
    if not pd.isna(rsi):
        if rsi < 25:   score += 15
        elif rsi < 35: score += 8
        elif rsi < 45: score += 3
        elif rsi > 75: score -= 15
        elif rsi > 65: score -= 8
        elif rsi > 55: score -= 3

    # ── Stochastic RSI (±20) ──────────────────────────────────────────────────
    k, k_prev = r["stoch_k"], prev["stoch_k"]
    if not pd.isna(k) and not pd.isna(k_prev):
        if k < 0.20 and k > k_prev:   score += 20
        elif k < 0.20:                  score += 10
        elif k > 0.80 and k < k_prev:  score -= 20
        elif k > 0.80:                  score -= 10

    # ── MACD (±20) ────────────────────────────────────────────────────────────
    ml, sl = r["macd"], r["macd_sig"]
    hist, p_hist = r["macd_hist"], prev["macd_hist"]
    if not any(pd.isna([ml, sl, hist, p_hist])):
        if ml > sl and p_hist <= 0:   score += 20
        elif ml > sl:                  score += 10
        elif ml < sl and p_hist >= 0: score -= 20
        elif ml < sl:                  score -= 10

    # ── Bollinger Bands (±15) ─────────────────────────────────────────────────
    bl, bu, price = r["bb_lower"], r["bb_upper"], r["close"]
    if not pd.isna(bl) and not pd.isna(bu):
        band = bu - bl
        if band > 0:
            pct = (price - bl) / band
            if pct < 0.10:   score += 15
            elif pct < 0.25: score += 7
            elif pct > 0.90: score -= 15
            elif pct > 0.75: score -= 7

    # ── Supertrend (±20) ──────────────────────────────────────────────────────
    st, st_prev = r["supertrend"], prev["supertrend"]
    if not pd.isna(st) and not pd.isna(st_prev):
        if st == 1 and st_prev == -1:   score += 20
        elif st == 1:                    score += 10
        elif st == -1 and st_prev == 1: score -= 20
        elif st == -1:                   score -= 10

    # ── Volume amplifier ──────────────────────────────────────────────────────
    vs = r["vol_sma"]
    if not pd.isna(vs) and vs > 0:
        vr = r["volume"] / vs
        if vr > 2.0:   score = int(score * 1.15)
        elif vr > 1.5: score = int(score * 1.08)

    score = max(-100, min(100, score))
    atr   = float(r["atr"]) if not pd.isna(r["atr"]) else 0.0
    adx   = float(r["adx"]) if not pd.isna(r["adx"]) else 25.0
    return score, atr, adx


# ── SL/TP helper ──────────────────────────────────────────────────────────────

def _make_sl_tp(side: str, price: float, atr: float) -> tuple[float, float]:
    if USE_ATR_SL_TP and atr > 0:
        if side == "long":
            return price - atr * ATR_SL_MULTIPLIER, price + atr * ATR_TP_MULTIPLIER
        return price + atr * ATR_SL_MULTIPLIER, price - atr * ATR_TP_MULTIPLIER
    if side == "long":
        return price * (1 - STOP_LOSS_PCT), price * (1 + TAKE_PROFIT_PCT)
    return price * (1 + STOP_LOSS_PCT), price * (1 - TAKE_PROFIT_PCT)


# ── Core backtest ─────────────────────────────────────────────────────────────

def backtest(symbol: str, granularity: str = "15m", limit: int = 1000,
             threshold: int = 50, months: int = 0) -> dict | None:
    df, source = _load_ohlcv(symbol, granularity, limit, months)
    if df.empty or len(df) < WARMUP + 10:
        return {"symbol": symbol, "error": f"only {len(df)} candles available"}

    df    = _precompute(df)
    trade = None   # currently open simulated trade
    trades: list  = []
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0

    for i in range(WARMUP, len(df)):
        row   = df.iloc[i]
        high  = float(row["high"])
        low   = float(row["low"])
        price = float(row["close"])

        # ── Manage open trade ─────────────────────────────────────────────────
        if trade:
            side = trade["side"]

            # Ratchet trailing stop before checking exit
            if side == "long":
                new_sl = round(price * (1 - TRAILING_STOP_PCT), 8)
                if new_sl > trade["sl"]:
                    trade["sl"] = new_sl
            else:
                new_sl = round(price * (1 + TRAILING_STOP_PCT), 8)
                if trade["sl"] == 0 or new_sl < trade["sl"]:
                    trade["sl"] = new_sl

            sl, tp = trade["sl"], trade["tp"]
            exit_price = result = None

            if side == "long":
                if low <= sl:     exit_price, result = sl, "sl"
                elif high >= tp:  exit_price, result = tp, "tp"
            else:
                if high >= sl:    exit_price, result = sl, "sl"
                elif low <= tp:   exit_price, result = tp, "tp"

            if exit_price:
                pnl_pct  = (exit_price - trade["entry"]) / trade["entry"]
                if side == "short":
                    pnl_pct = -pnl_pct
                pnl_usdt = POSITION_SIZE_USDT * pnl_pct * LEVERAGE

                # Trailing stop exits that are profitable → classify as "trail" (win)
                if result == "sl" and pnl_usdt > 0:
                    result = "trail"

                trade.update({
                    "exit_price": exit_price,
                    "result":     result,
                    "pnl":        round(pnl_usdt, 4),
                    "exit_ts":    str(row["timestamp"])[:16],
                    "duration":   i - trade["entry_idx"],
                })
                trades.append(trade)

                equity += pnl_usdt
                peak    = max(peak, equity)
                max_dd  = max(max_dd, peak - equity)
                trade   = None
            continue   # no new entry in same candle

        # ── Look for new entry ────────────────────────────────────────────────
        score, atr, adx = _score_at(df, i)

        # ADX filter – mirror live bot behaviour
        if adx < ADX_NO_TREND:
            continue
        if adx < ADX_WEAK_TREND:
            score = int(score * 0.70)
        elif adx > ADX_STRONG_TREND:
            score = min(100, int(score * 1.20))

        if score >= threshold:
            sl, tp = _make_sl_tp("long", price, atr)
            trade  = {
                "symbol": symbol, "side": "long",
                "entry": price, "sl": sl, "tp": tp,
                "score": score, "atr": atr,
                "entry_ts": str(row["timestamp"])[:16],
                "entry_idx": i,
            }
        elif score <= -threshold:
            sl, tp = _make_sl_tp("short", price, atr)
            trade  = {
                "symbol": symbol, "side": "short",
                "entry": price, "sl": sl, "tp": tp,
                "score": score, "atr": atr,
                "entry_ts": str(row["timestamp"])[:16],
                "entry_idx": i,
            }

    if not trades:
        return {"symbol": symbol, "granularity": granularity, "trades": 0,
                "error": "no trades triggered"}

    wins   = [t for t in trades if t["result"] in ("tp", "trail")]
    losses = [t for t in trades if t["result"] == "sl"]
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))

    return {
        "symbol":        symbol,
        "granularity":   granularity,
        "source":        source,
        "candles":       len(df),
        "from":          str(df["timestamp"].iloc[0])[:16],
        "to":            str(df["timestamp"].iloc[-1])[:16],
        "trades":        len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins) / len(trades) * 100, 1),
        "pnl":           round(equity, 2),
        "roi_pct":       round(equity / POSITION_SIZE_USDT * 100, 2),
        "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "max_drawdown":  round(max_dd, 2),
        "avg_win":       round(gp / len(wins), 2)    if wins   else 0,
        "avg_loss":      round(-gl / len(losses), 2) if losses else 0,
        "best_trade":    round(max(t["pnl"] for t in trades), 2),
        "worst_trade":   round(min(t["pnl"] for t in trades), 2),
        "avg_duration":  round(sum(t["duration"] for t in trades) / len(trades), 1),
        "all_trades":    trades,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def _bar(c="─", n=60): return c * n


def print_result(r: dict, show_trades: bool = False):
    if "error" in r and r.get("trades", 0) == 0:
        print(f"\n  [{r['symbol']}] {r['error']}")
        return

    sl_str = f"ATR×{ATR_SL_MULTIPLIER}" if USE_ATR_SL_TP else f"{STOP_LOSS_PCT*100:.1f}%"
    tp_str = f"ATR×{ATR_TP_MULTIPLIER}" if USE_ATR_SL_TP else f"{TAKE_PROFIT_PCT*100:.1f}%"

    print(f"\n{_bar()}")
    print(f"  {r['symbol']}  │  {r['granularity']}  │  {r.get('source','?')}  │  {r['from']} → {r['to']}")
    print(f"  {r['candles']} candles  │  SL: {sl_str}  TP: {tp_str}  │  {LEVERAGE}x leverage")
    print(_bar("·"))
    trails = sum(1 for t in r.get("all_trades", []) if t["result"] == "trail")
    print(f"  Trades        : {r['trades']}   (W: {r['wins']}  Trail: {trails}  L: {r['losses']}  WR: {r['win_rate']}%)")
    print(f"  Total PnL     : {r['pnl']:+.2f} USDT   ROI: {r['roi_pct']:+.2f}%")
    print(f"  Profit Factor : {r['profit_factor']:.2f}")
    print(f"  Max Drawdown  : {r['max_drawdown']:.2f} USDT")
    print(f"  Avg Win / Loss: {r['avg_win']:+.2f} / {r['avg_loss']:+.2f} USDT")
    print(f"  Best / Worst  : {r['best_trade']:+.2f} / {r['worst_trade']:+.2f} USDT")
    print(f"  Avg Duration  : {r['avg_duration']:.0f} candles")
    print(_bar())

    if show_trades and r.get("all_trades"):
        print(f"  {'Entry Time':<17} {'Side':<6} {'Entry':>12} {'Exit':>12} {'Result':<6} {'PnL':>9}")
        print(f"  {'─'*17} {'─'*5} {'─'*12} {'─'*12} {'─'*6} {'─'*9}")
        for t in r["all_trades"]:
            print(
                f"  {t['entry_ts']:<17} {t['side'].upper():<6}"
                f" {t['entry']:>12.6g} {t.get('exit_price', 0):>12.6g}"
                f" {t['result']:<6} {t['pnl']:>+9.2f}"
            )
        print(_bar())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CryptoBot Backtest")
    parser.add_argument("symbols", nargs="*",
                        help="Symbols to test (default: top coins from config)")
    parser.add_argument("--granularity", "-g", default="15m",
                        help="Candle interval, e.g. 15m, 1H, 4H (default: 15m)")
    parser.add_argument("--limit", "-l", type=int, default=1000,
                        help="Number of candles to fetch (max ~1000, default: 1000)")
    parser.add_argument("--trades", action="store_true",
                        help="Print every individual trade")
    parser.add_argument("--threshold", "-t", type=int, default=50,
                        help="TA score threshold for entry (default: 50).")
    parser.add_argument("--months", "-m", type=int, default=0,
                        help="Fetch N months of history from Binance (e.g. 3 or 6). "
                             "Falls back to Bitget if symbol not on Binance.")
    args = parser.parse_args()

    symbols = args.symbols
    if not symbols:
        print("Fetching top symbols from Bitget…")
        symbols = get_top_symbols(TOP_COINS_COUNT)
        if not symbols:
            print("Could not fetch symbols – check API connection.")
            sys.exit(1)

    source_label = f"Binance ({args.months}m)" if args.months else "Bitget (~10d)"
    print(f"\n{'═'*60}")
    print(f"  CryptoBot Backtest")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Granularity: {args.granularity}  │  Source: {source_label}")
    print(f"  Position: {POSITION_SIZE_USDT} USDT  │  Leverage: {LEVERAGE}x")
    print(f"  Note: TA-only (News/FG/Funding excluded – no historical data)")
    print(f"{'═'*60}")

    results = []
    for sym in symbols:
        print(f"\n  Backtesting {sym}…", end=" ", flush=True)
        r = backtest(sym, args.granularity, args.limit, args.threshold, args.months)
        if r:
            if r.get("trades", 0) > 0:
                print(f"{r['trades']} trades  PnL: {r['pnl']:+.2f} USDT  WR: {r['win_rate']}%")
            else:
                print(r.get("error", "no trades"))
            results.append(r)
            print_result(r, show_trades=args.trades)

    # Summary across all symbols
    valid = [r for r in results if r.get("trades", 0) > 0]
    if len(valid) > 1:
        total_pnl    = sum(r["pnl"]    for r in valid)
        total_trades = sum(r["trades"] for r in valid)
        total_wins   = sum(r["wins"]   for r in valid)
        print(f"\n{'═'*60}")
        print(f"  SUMMARY  │  {len(valid)} symbols  │  {total_trades} total trades")
        print(f"  Win Rate : {total_wins / total_trades * 100:.1f}%")
        print(f"  Total PnL: {total_pnl:+.2f} USDT")
        print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
