import logging
from config import (
    TA_WEIGHT, NEWS_WEIGHT, FEAR_GREED_WEIGHT, FUNDING_WEIGHT,
    LONG_THRESHOLD, SHORT_THRESHOLD, CONFIRM_TIMEFRAME,
    ADX_NO_TREND, ADX_WEAK_TREND, ADX_STRONG_TREND,
)
from technical_analysis import calculate_signals
from news_sentiment import get_news_sentiment, get_fear_greed_score, get_market_regime
from bitget_client import get_funding_rate
from database import save_signal

logger = logging.getLogger(__name__)

_FUNDING_EXTREME_HIGH =  0.001
_FUNDING_HIGH         =  0.0005
_FUNDING_EXTREME_LOW  = -0.001
_FUNDING_LOW          = -0.0005


def _funding_to_score(rate: float) -> float:
    if rate >= _FUNDING_EXTREME_HIGH:  return -100.0
    if rate >= _FUNDING_HIGH:          return  -50.0
    if rate <= _FUNDING_EXTREME_LOW:   return  100.0
    if rate <= _FUNDING_LOW:           return   50.0
    if rate > 0:
        return round(-50.0 * (rate / _FUNDING_HIGH), 1)
    return round(50.0 * (rate / _FUNDING_LOW), 1)


def _multitf_factor(symbol: str, primary_score: float) -> float:
    try:
        tf_result = calculate_signals(symbol, granularity=CONFIRM_TIMEFRAME)
        tf_score  = tf_result.get("score", 0)
        if abs(tf_score) < 20:
            return 1.0
        if (primary_score > 0 and tf_score > 0) or (primary_score < 0 and tf_score < 0):
            return 1.15
        return 0.70
    except Exception as e:
        logger.debug(f"Multi-TF check failed for {symbol}: {e}")
        return 1.0


def _adx_factor(adx: float) -> float | None:
    """
    Returns a multiplier for the final score based on ADX trend strength.
    Returns None to signal "no trend – force HOLD regardless of score".
    """
    if adx < ADX_NO_TREND:
        return None          # no measurable trend → skip
    if adx < ADX_WEAK_TREND:
        return 0.70          # weak trend → dampen confidence
    if adx > ADX_STRONG_TREND:
        return 1.20          # strong trend → boost confidence
    return 1.0               # normal trend → unchanged


def analyze_symbol(symbol: str) -> dict:
    """
    Full analysis combining:
      TA (55%) + News (15%) + Fear&Greed (10%) + Funding Rate (20%)

    Filters / adjustments applied in order:
      1. ADX filter  – no trend → HOLD
      2. Weighted score combination
      3. Multi-timeframe confirmation factor
    """
    # ── Technical Analysis (15m) ─────────────────────────────────────────────
    ta_result  = calculate_signals(symbol)
    ta_score   = ta_result.get("score", 0)
    indicators = ta_result.get("indicators", {})
    atr        = ta_result.get("atr")
    adx        = indicators.get("adx", 25.0)

    # ── ADX gate – if no clear trend, skip immediately ────────────────────────
    adx_mult = _adx_factor(adx)
    if adx_mult is None:
        logger.info(
            f"{symbol:12s}  ADX={adx:.1f} < {ADX_NO_TREND}  → HOLD (no trend)"
        )
        save_signal(symbol, ta_score, 0.0, 0.0, "hold", indicators)
        return {
            "symbol": symbol, "ta_score": round(ta_score, 2),
            "news_score": 0.0, "fear_greed_score": 0.0,
            "funding_rate": 0.0, "funding_score": 0.0,
            "final_score": 0.0, "action": "hold",
            "indicators": indicators, "atr": atr,
        }

    # ── News Sentiment ────────────────────────────────────────────────────────
    news_score       = get_news_sentiment(symbol)
    fear_greed_score = get_fear_greed_score()
    funding_rate     = get_funding_rate(symbol)
    funding_score    = _funding_to_score(funding_rate)

    # ── Weighted combination ──────────────────────────────────────────────────
    raw_score = (
        ta_score         * TA_WEIGHT +
        news_score       * NEWS_WEIGHT +
        fear_greed_score * FEAR_GREED_WEIGHT +
        funding_score    * FUNDING_WEIGHT
    )

    # ── ADX strength factor ───────────────────────────────────────────────────
    raw_score = raw_score * adx_mult

    # ── Multi-timeframe confirmation ──────────────────────────────────────────
    mtf_factor = _multitf_factor(symbol, raw_score)

    # ── Market regime factor (4h cached, no extra API cost) ───────────────────
    regime = get_market_regime()
    if regime == "bull":
        regime_factor = 1.15 if raw_score > 0 else 0.85   # favour longs
    elif regime == "bear":
        regime_factor = 1.15 if raw_score < 0 else 0.85   # favour shorts
    elif regime == "sideways":
        regime_factor = 0.85                               # dampen all signals
    else:
        regime_factor = 1.0

    final_score = round(raw_score * mtf_factor * regime_factor, 2)
    final_score = max(-100.0, min(100.0, final_score))

    if final_score >= LONG_THRESHOLD:
        action = "long"
    elif final_score <= SHORT_THRESHOLD:
        action = "short"
    else:
        action = "hold"

    save_signal(symbol, ta_score, news_score, final_score, action, indicators)

    logger.info(
        f"{symbol:12s}  TA={ta_score:+.1f}  ADX={adx:.1f}  News={news_score:+.1f}"
        f"  FG={fear_greed_score:+.1f}  Fund={funding_rate*100:+.4f}%"
        f"  MTF×{mtf_factor:.2f}  Regime={regime}×{regime_factor:.2f}"
        f"  Final={final_score:+.1f}  → {action.upper()}"
    )

    return {
        "symbol":           symbol,
        "ta_score":         round(ta_score, 2),
        "news_score":       round(news_score, 2),
        "fear_greed_score": round(fear_greed_score, 2),
        "funding_rate":     funding_rate,
        "funding_score":    round(funding_score, 2),
        "final_score":      final_score,
        "action":           action,
        "indicators":       indicators,
        "atr":              atr,
        "adx":              adx,
        "regime":           regime,
    }
