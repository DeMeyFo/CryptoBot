import logging

from config import ALLOWED_TRADE_SIDES, STRATEGY_TIMEFRAME, TRADE_DIRECTION
from database import save_signal
from technical_analysis import calculate_signals

logger = logging.getLogger(__name__)


def _result(symbol: str, ta_score: float, action: str, indicators: dict,
            atr=None, timestamp=None) -> dict:
    return {
        "symbol": symbol,
        "ta_score": round(ta_score, 2),
        "news_score": 0.0,
        "fear_greed_score": 0.0,
        "funding_rate": 0.0,
        "funding_score": 0.0,
        "oi_score": 0.0,
        "final_score": round(ta_score, 2),
        "action": action,
        "indicators": indicators,
        "atr": atr,
        "adx": indicators.get("adx", 0.0),
        "regime": "core-only",
        "timestamp": timestamp,
    }


def analyze_symbol(symbol: str, granularity: str | None = None) -> dict:
    """
    Evaluate only the validated technical core on completed candles.

    External news, sentiment, funding and LLM calls are intentionally absent from
    the time-critical order path. Their historical parity and latency were not part
    of the walk-forward validation.
    """
    ta_result = calculate_signals(symbol, granularity=granularity or STRATEGY_TIMEFRAME)
    ta_score = float(ta_result.get("score", 0.0))
    indicators = dict(ta_result.get("indicators", {}))
    atr = ta_result.get("atr")
    timestamp = ta_result.get("timestamp")
    entry_signal = ta_result.get("entry_signal", "hold")

    if ta_result.get("error"):
        indicators["filter_reason"] = ta_result["error"]
        save_signal(symbol, ta_score, 0.0, 0.0, "hold", indicators)
        logger.warning(f"{symbol:12s} data invalid -> HOLD ({ta_result['error']})")
        return _result(symbol, ta_score, "hold", indicators, atr, timestamp)

    if entry_signal == "hold":
        indicators["filter_reason"] = "trend-breakout conditions incomplete"
        save_signal(symbol, ta_score, 0.0, ta_score, "hold", indicators)
        logger.info(
            f"{symbol:12s} core={ta_score:+.0f} ADX={indicators.get('adx', 0):.1f} "
            f"RSI={indicators.get('rsi', 0):.1f} -> HOLD"
        )
        return _result(symbol, ta_score, "hold", indicators, atr, timestamp)

    # The setup is complete but the direction is disabled. Record it as a
    # filtered signal rather than dropping it, so the dashboard still shows
    # that a valid breakout occurred and why it was not traded.
    if entry_signal not in ALLOWED_TRADE_SIDES:
        indicators["filter_reason"] = (
            f"{entry_signal} disabled by TRADE_DIRECTION={TRADE_DIRECTION}"
        )
        save_signal(symbol, ta_score, 0.0, ta_score, "hold", indicators)
        logger.info(
            f"{symbol:12s} BREAKOUT {entry_signal.upper()} filtered "
            f"(direction={TRADE_DIRECTION}) core={ta_score:+.0f}"
        )
        return _result(symbol, ta_score, "hold", indicators, atr, timestamp)

    save_signal(symbol, ta_score, 0.0, ta_score, entry_signal, indicators)
    logger.info(
        f"{symbol:12s} BREAKOUT {entry_signal.upper()} core={ta_score:+.0f} "
        f"ADX={indicators.get('adx', 0):.1f} RSI={indicators.get('rsi', 0):.1f}"
    )
    return _result(symbol, ta_score, entry_signal, indicators, atr, timestamp)
