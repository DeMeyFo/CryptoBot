import logging
from datetime import datetime
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Free RSS feeds – no API key required
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://cryptopanic.com/news/rss/",
]

COIN_KEYWORDS: dict[str, list[str]] = {
    "BTC":   ["bitcoin", "btc"],
    "ETH":   ["ethereum", "eth"],
    "SOL":   ["solana", "sol"],
    "BNB":   ["binance", "bnb"],
    "XRP":   ["ripple", "xrp"],
    "ADA":   ["cardano", "ada"],
    "DOGE":  ["dogecoin", "doge"],
    "AVAX":  ["avalanche", "avax"],
    "DOT":   ["polkadot", "dot"],
    "MATIC": ["polygon", "matic"],
    "LINK":  ["chainlink", "link"],
    "LTC":   ["litecoin", "ltc"],
    "NEAR":  ["near protocol", "near"],
    "OP":    ["optimism", " op "],
    "ARB":   ["arbitrum", "arb"],
    "SUI":   ["sui network", " sui "],
    "APT":   ["aptos", "apt"],
    "INJ":   ["injective", " inj "],
    "TIA":   ["celestia", " tia "],
    "WIF":   ["dogwifhat", " wif "],
}

_CACHE_TTL = 900  # 15 min

_analyzer = SentimentIntensityAnalyzer()
_articles: list = []
_articles_ts: datetime | None = None
_score_cache: dict[str, tuple[datetime, float]] = {}


def _refresh_articles():
    global _articles, _articles_ts
    now = datetime.utcnow()
    if _articles_ts and (now - _articles_ts).seconds < _CACHE_TTL:
        return
    collected = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:25]:
                collected.append(
                    entry.get("title", "") + " " + entry.get("summary", "")
                )
        except Exception as e:
            logger.debug(f"RSS {url}: {e}")
    _articles = collected
    _articles_ts = now
    logger.debug(f"News cache refreshed: {len(_articles)} articles")


def get_news_sentiment(symbol: str) -> float:
    """Return sentiment score in [-100, +100] for the given futures symbol."""
    coin = symbol.replace("USDT", "").replace("PERP", "").upper()

    # Return cached score if fresh
    if coin in _score_cache:
        ts, val = _score_cache[coin]
        if (datetime.utcnow() - ts).seconds < _CACHE_TTL:
            return val

    _refresh_articles()

    keywords = COIN_KEYWORDS.get(coin, [coin.lower()])
    scores = []
    for text in _articles:
        text_low = text.lower()
        if any(kw in text_low for kw in keywords):
            compound = _analyzer.polarity_scores(text)["compound"]
            scores.append(compound)

    result = round((sum(scores) / len(scores)) * 100, 1) if scores else 0.0
    _score_cache[coin] = (datetime.utcnow(), result)
    logger.debug(f"News sentiment {coin}: {result:+.1f} (from {len(scores)} articles)")
    return result
