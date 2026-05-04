import logging
import math
import re
from datetime import datetime, timezone

import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://cryptopanic.com/news/rss/",
    "https://www.theblock.co/rss.xml",
    "https://bitcoinist.com/feed/",
    "https://beincrypto.com/feed/",
    "https://cryptoslate.com/feed/",
    "https://ambcrypto.com/feed/",
    "https://newsbtc.com/feed/",
]

# Regex patterns per coin – word-boundary matching prevents "link" matching "blockchain"
COIN_PATTERNS: dict[str, list[str]] = {
    "BTC":   [r"\bbitcoin\b", r"\bbtc\b"],
    "ETH":   [r"\bethereum\b", r"\beth\b"],
    "SOL":   [r"\bsolana\b", r"\bsol\b"],
    "BNB":   [r"\bbinance\b", r"\bbnb\b"],
    "XRP":   [r"\bripple\b", r"\bxrp\b"],
    "ADA":   [r"\bcardano\b", r"\bada\b"],
    "DOGE":  [r"\bdogecoin\b", r"\bdoge\b"],
    "AVAX":  [r"\bavalanche\b", r"\bavax\b"],
    "DOT":   [r"\bpolkadot\b", r"\bdot\b"],
    "MATIC": [r"\bpolygon\b", r"\bmatic\b"],
    "LINK":  [r"\bchainlink\b", r"\blink token\b"],
    "LTC":   [r"\blitecoin\b", r"\bltc\b"],
    "NEAR":  [r"\bnear protocol\b", r"\bnear\b"],
    "OP":    [r"\boptimism\b", r"\b op \b"],
    "ARB":   [r"\barbitrum\b", r"\barb\b"],
    "SUI":   [r"\bsui network\b", r"\bsui\b"],
    "APT":   [r"\baptos\b", r"\bapt\b"],
    "INJ":   [r"\binjective\b", r"\binj\b"],
    "TIA":   [r"\bcelestia\b", r"\btia\b"],
    "WIF":   [r"\bdogwifhat\b", r"\bwif\b"],
}

_ARTICLE_TTL    = 900   # 15 min
_FEAR_GREED_TTL = 3600  # 1 hour

_analyzer = SentimentIntensityAnalyzer()
_articles: list = []
_articles_ts: datetime | None = None
_score_cache: dict[str, tuple[datetime, float]] = {}
_fear_greed_cache: tuple[datetime, float] | None = None

# Pre-compile all patterns
_compiled_patterns: dict[str, list[re.Pattern]] = {
    coin: [re.compile(p, re.IGNORECASE) for p in patterns]
    for coin, patterns in COIN_PATTERNS.items()
}


def _entry_age_seconds(entry) -> float:
    """Seconds since publication. Falls back to 3600 if unparseable."""
    for field in ("published_parsed", "updated_parsed"):
        t = entry.get(field)
        if t:
            try:
                pub = datetime(*t[:6], tzinfo=timezone.utc)
                return max(0.0, (datetime.now(timezone.utc) - pub).total_seconds())
            except Exception:
                pass
    return 3600.0


def _recency_weight(age_seconds: float) -> float:
    """Exponential decay: 1.0 now, ~0.45 at 4h, ~0.13 at 12h, ~0.02 at 24h."""
    return math.exp(-0.25 * age_seconds / 3600)


def _refresh_articles():
    global _articles, _articles_ts
    now = datetime.now(timezone.utc)
    if _articles_ts and (now - _articles_ts).total_seconds() < _ARTICLE_TTL:
        return
    collected = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                text = (entry.get("title", "") + " " + entry.get("summary", "")).strip()
                if text:
                    collected.append({
                        "text": text,
                        "age_s": _entry_age_seconds(entry),
                    })
        except Exception as e:
            logger.debug(f"RSS {url}: {e}")
    _articles = collected
    _articles_ts = now
    logger.debug(f"News cache refreshed: {len(_articles)} articles from {len(RSS_FEEDS)} feeds")


def get_fear_greed_score() -> float:
    """
    Fetch Alternative.me Fear & Greed Index and convert to signal score [-100, +100].
    Extreme fear (index=0) → +100 (buy signal)
    Extreme greed (index=100) → -100 (sell signal)
    Neutral (index=50) → 0
    """
    global _fear_greed_cache
    now = datetime.now(timezone.utc)
    if _fear_greed_cache:
        ts, val = _fear_greed_cache
        if (now - ts).total_seconds() < _FEAR_GREED_TTL:
            return val
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=5,
            headers={"User-Agent": "CryptoBot/1.0"},
        )
        resp.raise_for_status()
        index = int(resp.json()["data"][0]["value"])
        # Invert: fear = buy, greed = sell. Scale from 0-100 → ±100
        score = round(-(index - 50) * 2.0, 1)
        score = max(-100.0, min(100.0, score))
        _fear_greed_cache = (now, score)
        logger.debug(f"Fear & Greed Index: {index}/100 → signal {score:+.1f}")
        return score
    except Exception as e:
        logger.debug(f"Fear & Greed API error: {e}")
        return 0.0


def get_news_sentiment(symbol: str) -> float:
    """
    Weighted sentiment score in [-100, +100] for a futures symbol.
    Recent articles are exponentially up-weighted vs. older ones.
    """
    coin = symbol.replace("USDT", "").replace("PERP", "").upper()

    if coin in _score_cache:
        ts, val = _score_cache[coin]
        if (datetime.now(timezone.utc) - ts).total_seconds() < _ARTICLE_TTL:
            return val

    _refresh_articles()

    patterns = _compiled_patterns.get(
        coin, [re.compile(rf"\b{re.escape(coin.lower())}\b", re.IGNORECASE)]
    )

    weighted_sum = 0.0
    weight_total = 0.0
    for art in _articles:
        text = art["text"]
        if any(p.search(text) for p in patterns):
            compound = _analyzer.polarity_scores(text)["compound"]
            w = _recency_weight(art["age_s"])
            weighted_sum += compound * w
            weight_total += w

    result = round((weighted_sum / weight_total) * 100, 1) if weight_total > 0 else 0.0
    _score_cache[coin] = (datetime.now(timezone.utc), result)
    n = sum(1 for a in _articles if any(p.search(a["text"]) for p in patterns))
    logger.debug(f"News sentiment {coin}: {result:+.1f} ({n} articles, recency-weighted)")
    return result
