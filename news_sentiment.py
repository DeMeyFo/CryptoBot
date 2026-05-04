import json
import logging
import math
import re
from datetime import datetime, timezone

import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import CLAUDE_API_KEY, USE_CLAUDE_SENTIMENT

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
    "TAO":   [r"\bbittensor\b", r"\btao\b"],
    "CLU":   [r"\bclubhouse\b", r"\bclu\b"],
    "BUS":   [r"\bbus token\b", r"\bbus\b"],
    "RAVE":  [r"\brave token\b", r"\brave\b"],
}

_ARTICLE_TTL    = 900   # 15 min
_FEAR_GREED_TTL = 3600  # 1 hour

# ── VADER ──────────────────────────────────────────────────────────────────────
_analyzer = SentimentIntensityAnalyzer()
_compiled_patterns: dict[str, list[re.Pattern]] = {
    coin: [re.compile(p, re.IGNORECASE) for p in patterns]
    for coin, patterns in COIN_PATTERNS.items()
}

# ── Shared article cache ───────────────────────────────────────────────────────
_articles: list = []
_articles_ts: datetime | None = None

# ── Per-coin score cache (shared by both paths) ───────────────────────────────
_score_cache: dict[str, tuple[datetime, float]] = {}

# ── Fear & Greed cache ─────────────────────────────────────────────────────────
_fear_greed_cache: tuple[datetime, float] | None = None

# ── Claude batch-score cache ───────────────────────────────────────────────────
_claude_scores: dict[str, float] = {}
_claude_scores_ts: datetime | None = None
_anthropic_client = None

# ── Static system prompt for Claude (cached by the API via cache_control) ──────
_CLAUDE_SYSTEM = (
    "You are a professional crypto market analyst with deep expertise in blockchain technology, "
    "DeFi protocols, and cryptocurrency markets.\n\n"
    "TASK: Analyse the provided news article headlines and summaries, then score the market "
    "sentiment for each requested cryptocurrency.\n\n"
    "SCORING SCALE: -100 (extremely bearish / major negative event) to +100 (extremely bullish / "
    "major positive event). Use 0 for neutral or no relevant news.\n\n"
    "WEIGHTING: Recent articles (published within the last 1-2 hours) carry more weight than "
    "older ones. Prioritise: regulatory decisions, security incidents (hacks/exploits), protocol "
    "upgrades, institutional adoption, exchange listings, legal proceedings, partnership "
    "announcements, whale activity, and macro crypto trends.\n\n"
    "OUTPUT FORMAT: Return ONLY a valid JSON object. Keys = uppercase coin symbols. "
    "Values = integer scores. No markdown, no code fences, no explanation — pure JSON only.\n"
    "Example: {\"BTC\": 35, \"SOL\": -20, \"ETH\": 0}"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _entry_age_seconds(entry) -> float:
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
                    collected.append({"text": text, "age_s": _entry_age_seconds(entry)})
        except Exception as e:
            logger.debug(f"RSS {url}: {e}")
    _articles = collected
    _articles_ts = now
    logger.debug(f"News cache refreshed: {len(_articles)} articles")


# ── Fear & Greed Index ─────────────────────────────────────────────────────────

def get_fear_greed_score() -> float:
    """Alternative.me Fear & Greed Index → signal [-100, +100]. Cached 1h."""
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
        score = round(-(index - 50) * 2.0, 1)
        score = max(-100.0, min(100.0, score))
        _fear_greed_cache = (now, score)
        logger.debug(f"Fear & Greed Index: {index}/100 → signal {score:+.1f}")
        return score
    except Exception as e:
        logger.debug(f"Fear & Greed API error: {e}")
        return 0.0


# ── Claude AI sentiment ────────────────────────────────────────────────────────

def _get_anthropic_client():
    """Lazy-init the Anthropic client (only when Claude is enabled)."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic as _ant
            _anthropic_client = _ant.Anthropic(api_key=CLAUDE_API_KEY)
        except Exception as e:
            logger.warning(f"Could not initialise Anthropic client: {e}")
    return _anthropic_client


def _get_claude_scores(coins: list[str]) -> dict[str, float]:
    """
    ONE Claude API call per refresh cycle → scores all coins at once.
    Results are cached for _ARTICLE_TTL seconds (15 min).
    Falls back to an empty dict on any error; caller then falls back to VADER.
    """
    global _claude_scores, _claude_scores_ts

    now = datetime.now(timezone.utc)
    if _claude_scores_ts and (now - _claude_scores_ts).total_seconds() < _ARTICLE_TTL:
        return _claude_scores

    _refresh_articles()
    if not _articles:
        return {}

    client = _get_anthropic_client()
    if not client:
        return {}

    # Newest articles first, truncated to save tokens (~200 chars each, max 25)
    sorted_arts = sorted(_articles, key=lambda a: a["age_s"])
    article_text = "\n".join(
        f"{i + 1}. {a['text'][:200]}"
        for i, a in enumerate(sorted_arts[:25])
    )
    coin_list = ", ".join(coins)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=[{
                "type": "text",
                "text": _CLAUDE_SYSTEM,
                "cache_control": {"type": "ephemeral"},  # API caches when ≥4096 tokens
            }],
            messages=[{
                "role": "user",
                "content": (
                    f"Recent crypto news (newest first):\n{article_text}\n\n"
                    f"Score sentiment for these coins: {coin_list}\n"
                    f"JSON only."
                ),
            }],
        )

        raw_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        ).strip()

        # Extract JSON even if surrounded by whitespace or backtick fences
        match = re.search(r"\{[^{}]+\}", raw_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            _claude_scores = {
                k.upper().replace("USDT", "").replace("PERP", ""): max(-100.0, min(100.0, float(v)))
                for k, v in parsed.items()
            }
            _claude_scores_ts = now

            cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
            input_tokens = getattr(response.usage, "input_tokens", 0)
            logger.debug(
                f"Claude sentiment scored {len(_claude_scores)} coins "
                f"(input={input_tokens} cache_read={cache_read})"
            )
            return _claude_scores
        else:
            logger.warning(f"Claude returned non-JSON response: {raw_text[:120]}")

    except Exception as e:
        logger.warning(f"Claude sentiment API error: {e}")

    return {}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_news_sentiment(symbol: str) -> float:
    """
    Return sentiment score [-100, +100] for a futures symbol.

    Pipeline:
      1. Return cached score if still fresh.
      2. If USE_CLAUDE_SENTIMENT=true: one batch Claude call for all coins.
      3. Fall back to VADER (recency-weighted) if Claude is disabled or fails.
    """
    coin = symbol.replace("USDT", "").replace("PERP", "").upper()

    # Return cached score if fresh
    if coin in _score_cache:
        ts, val = _score_cache[coin]
        if (datetime.now(timezone.utc) - ts).total_seconds() < _ARTICLE_TTL:
            return val

    # ── Claude path ───────────────────────────────────────────────────────────
    if USE_CLAUDE_SENTIMENT and CLAUDE_API_KEY:
        all_coins = list(COIN_PATTERNS.keys())
        claude_scores = _get_claude_scores(all_coins)
        if coin in claude_scores:
            result = round(claude_scores[coin], 1)
            _score_cache[coin] = (datetime.now(timezone.utc), result)
            logger.debug(f"Claude sentiment {coin}: {result:+.1f}")
            return result
        # Coin not in Claude response → fall through to VADER

    # ── VADER path (default / fallback) ───────────────────────────────────────
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
    logger.debug(f"VADER sentiment {coin}: {result:+.1f} ({n} articles)")
    return result
