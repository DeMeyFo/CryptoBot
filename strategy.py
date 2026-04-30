import logging
from config import TA_WEIGHT, NEWS_WEIGHT, LONG_THRESHOLD, SHORT_THRESHOLD
from technical_analysis import calculate_signals
from news_sentiment import get_news_sentiment
from database import save_signal

logger = logging.getLogger(__name__)


def analyze_symbol(symbol: str) -> dict:
    """
    Full analysis combining TA (70 %) and news sentiment (30 %).

    Returns:
        {
            symbol, ta_score, news_score, final_score,
            action: 'long' | 'short' | 'hold',
            indicators: dict
        }
    """
    ta_result = calculate_signals(symbol)
    ta_score = ta_result.get("score", 0)
    indicators = ta_result.get("indicators", {})

    news_score = get_news_sentiment(symbol)

    final_score = round(ta_score * TA_WEIGHT + news_score * NEWS_WEIGHT, 2)

    if final_score >= LONG_THRESHOLD:
        action = "long"
    elif final_score <= SHORT_THRESHOLD:
        action = "short"
    else:
        action = "hold"

    save_signal(symbol, ta_score, news_score, final_score, action, indicators)

    logger.info(
        f"{symbol:12s}  TA={ta_score:+.1f}  News={news_score:+.1f}  "
        f"Final={final_score:+.1f}  -> {action.upper()}"
    )
    return {
        "symbol": symbol,
        "ta_score": round(ta_score, 2),
        "news_score": round(news_score, 2),
        "final_score": final_score,
        "action": action,
        "indicators": indicators,
    }
