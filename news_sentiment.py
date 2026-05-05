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

_ARTICLE_TTL      = 900    # 15 min
_FEAR_GREED_TTL   = 3600   # 1 hour
_MARKET_REGIME_TTL = 4 * 3600  # 4 hours

# ── VADER setup ────────────────────────────────────────────────────────────────
_analyzer = SentimentIntensityAnalyzer()
_compiled_patterns: dict[str, list[re.Pattern]] = {
    coin: [re.compile(p, re.IGNORECASE) for p in patterns]
    for coin, patterns in COIN_PATTERNS.items()
}

# ── Shared caches ──────────────────────────────────────────────────────────────
_articles: list            = []
_articles_ts: datetime | None = None
_score_cache: dict[str, tuple[datetime, float]] = {}
_fear_greed_cache: tuple[datetime, float] | None = None
_claude_scores: dict[str, float]  = {}
_claude_scores_ts: datetime | None = None
_claude_exit_signals: dict[str, bool] = {}
_market_regime_cache: tuple[datetime, str] | None = None
_anthropic_client = None

# ── System prompt (static → cached by Anthropic API) ──────────────────────────
_CLAUDE_SYSTEM = (
    "You are a professional crypto market analyst with deep expertise in blockchain technology, "
    "DeFi protocols, and cryptocurrency markets.\n\n"
    "PRIMARY TASK – Sentiment scoring:\n"
    "Analyse the provided news headlines and score market sentiment for each requested "
    "cryptocurrency from -100 (extremely bearish) to +100 (extremely bullish). "
    "Use 0 for neutral or no relevant news. Weight recent articles more heavily.\n"
    "Consider: regulatory decisions, security incidents, protocol upgrades, institutional "
    "adoption, exchange listings, legal proceedings, partnerships, whale activity.\n\n"
    "SECONDARY TASK – Exit signals (only when open positions are provided):\n"
    "Flag any open position that needs urgent exit due to clearly negative breaking news "
    "(hack, exploit, regulatory ban, major scandal). Only flag true emergencies.\n\n"
    "RESPONSE FORMAT:\n"
    "With positions:    {\"sentiment\": {\"COIN\": score}, \"exits\": {\"SYMBOL\": true/false}}\n"
    "Without positions: {\"sentiment\": {\"COIN\": score}}\n"
    "Return ONLY valid JSON – no markdown, no explanation."
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


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic as _ant
            _anthropic_client = _ant.Anthropic(api_key=CLAUDE_API_KEY)
        except Exception as e:
            logger.warning(f"Could not initialise Anthropic client: {e}")
    return _anthropic_client


# ── Fear & Greed ───────────────────────────────────────────────────────────────

def get_fear_greed_score() -> float:
    global _fear_greed_cache
    now = datetime.now(timezone.utc)
    if _fear_greed_cache:
        ts, val = _fear_greed_cache
        if (now - ts).total_seconds() < _FEAR_GREED_TTL:
            return val
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=5, headers={"User-Agent": "CryptoBot/1.0"},
        )
        resp.raise_for_status()
        index = int(resp.json()["data"][0]["value"])
        score = round(-(index - 50) * 2.0, 1)
        score = max(-100.0, min(100.0, score))
        _fear_greed_cache = (now, score)
        logger.debug(f"Fear & Greed: {index}/100 → {score:+.1f}")
        return score
    except Exception as e:
        logger.debug(f"Fear & Greed API error: {e}")
        return 0.0


# ── Claude: sentiment + exit signals (one batch call per 15 min) ───────────────

def _get_claude_scores(coins: list[str]) -> dict[str, float]:
    global _claude_scores, _claude_scores_ts, _claude_exit_signals

    now = datetime.now(timezone.utc)
    if _claude_scores_ts and (now - _claude_scores_ts).total_seconds() < _ARTICLE_TTL:
        return _claude_scores

    _refresh_articles()
    if not _articles:
        return {}

    client = _get_anthropic_client()
    if not client:
        return {}

    # Fetch open positions for piggybacked exit signal check
    open_positions = []
    try:
        from database import get_open_trades
        from config import DRY_RUN
        open_positions = [
            {"symbol": t["symbol"], "side": t["side"]}
            for t in get_open_trades(dry_run=DRY_RUN)
            if not t.get("is_pyramid")
        ]
    except Exception:
        pass

    sorted_arts  = sorted(_articles, key=lambda a: a["age_s"])
    article_text = "\n".join(f"{i+1}. {a['text'][:200]}" for i, a in enumerate(sorted_arts[:25]))
    coin_list    = ", ".join(coins)

    position_context = ""
    if open_positions:
        pos_str = ", ".join(f"{p['side'].upper()} {p['symbol']}" for p in open_positions)
        position_context = f"\n\nOpen positions to monitor: {pos_str}"

    try:
        response = _get_anthropic_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=[{
                "type": "text",
                "text": _CLAUDE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": (
                    f"Recent crypto news (newest first):\n{article_text}\n\n"
                    f"Score sentiment for: {coin_list}"
                    f"{position_context}\n"
                    f"JSON only."
                ),
            }],
        )

        raw = next((b.text for b in response.content if b.type == "text"), "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            # Support both {"sentiment": {...}} and legacy flat {"BTC": score}
            scores_raw = parsed.get("sentiment", parsed)
            _claude_scores = {
                k.upper().replace("USDT", "").replace("PERP", ""): max(-100.0, min(100.0, float(v)))
                for k, v in scores_raw.items()
                if isinstance(v, (int, float))
            }
            _claude_exit_signals = {
                k.upper(): bool(v)
                for k, v in parsed.get("exits", {}).items()
            }
            _claude_scores_ts = now

            cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
            exits_flagged = [s for s, flag in _claude_exit_signals.items() if flag]
            logger.debug(
                f"Claude batch: {len(_claude_scores)} coins scored, "
                f"exits={exits_flagged or 'none'}, cache_read={cache_read}"
            )
            if exits_flagged:
                logger.warning(f"⚠ Claude flagged urgent exit for: {exits_flagged}")
            return _claude_scores
        else:
            logger.warning(f"Claude non-JSON response: {raw[:120]}")
    except Exception as e:
        logger.warning(f"Claude sentiment error: {e}")

    return {}


def get_claude_exit_signals() -> dict[str, bool]:
    """Return cached exit-signal flags from the last Claude batch call."""
    return _claude_exit_signals.copy()


# ── Market regime (every 4 hours) ─────────────────────────────────────────────

def get_market_regime() -> str:
    """
    Ask Claude to classify the current market: 'bull', 'bear', or 'sideways'.
    Cached for 4 hours. Returns 'unknown' when Claude is disabled or fails.
    """
    global _market_regime_cache

    now = datetime.now(timezone.utc)
    if _market_regime_cache:
        ts, regime = _market_regime_cache
        if (now - ts).total_seconds() < _MARKET_REGIME_TTL:
            return regime

    if not (USE_CLAUDE_SENTIMENT and CLAUDE_API_KEY):
        return "unknown"

    client = _get_anthropic_client()
    if not client:
        return "unknown"

    _refresh_articles()
    fg_score = get_fear_greed_score()
    fg_index = max(0, min(100, int(50 - fg_score / 2)))
    recent   = sorted(_articles, key=lambda a: a["age_s"])[:10]
    headlines = "\n".join(f"- {a['text'][:150]}" for a in recent)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{
                "role": "user",
                "content": (
                    f"Crypto market context:\n"
                    f"- Fear & Greed Index: {fg_index}/100\n"
                    f"Recent headlines:\n{headlines}\n\n"
                    f"Classify the current crypto market regime.\n"
                    f"Return JSON only: {{\"regime\": \"bull\"|\"bear\"|\"sideways\"}}"
                ),
            }],
        )
        text  = next((b.text for b in response.content if b.type == "text"), "").strip()
        match = re.search(r'"regime"\s*:\s*"(bull|bear|sideways)"', text)
        if match:
            regime = match.group(1)
            _market_regime_cache = (now, regime)
            logger.info(f"Market regime updated: {regime.upper()}")
            return regime
    except Exception as e:
        logger.debug(f"Market regime error: {e}")

    return "unknown"


# ── Trade validation (one call per trade signal) ───────────────────────────────

def validate_trade(symbol: str, action: str, analysis: dict) -> tuple[bool, str]:
    """
    Ask Claude to approve or reject a trade before execution.
    Returns (should_proceed: bool, reason: str).
    Always returns True when Claude is disabled so the bot still trades normally.
    """
    if not (USE_CLAUDE_SENTIMENT and CLAUDE_API_KEY):
        return True, "Claude disabled"

    client = _get_anthropic_client()
    if not client:
        return True, "Client unavailable"

    ind    = analysis.get("indicators", {})
    regime = get_market_regime()   # uses cached value – no extra API call

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Review this crypto futures trade signal:\n"
                    f"Trade:  {action.upper()} {symbol}\n"
                    f"Scores: TA={analysis.get('ta_score', 0):+.0f}  "
                    f"News={analysis.get('news_score', 0):+.0f}  "
                    f"FG={analysis.get('fear_greed_score', 0):+.0f}  "
                    f"Final={analysis.get('final_score', 0):+.1f}/100\n"
                    f"ADX={ind.get('adx', 0):.1f}  "
                    f"RSI={ind.get('rsi', 50):.1f}  "
                    f"Funding={analysis.get('funding_rate', 0)*100:+.4f}%\n"
                    f"Market regime: {regime}\n\n"
                    f"Approve or reject. "
                    f"JSON only: {{\"approve\": true/false, \"reason\": \"one sentence\"}}"
                ),
            }],
        )
        text  = next((b.text for b in response.content if b.type == "text"), "").strip()
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            parsed  = json.loads(match.group())
            approve = bool(parsed.get("approve", True))
            reason  = str(parsed.get("reason", ""))
            icon    = "✅" if approve else "❌"
            logger.info(f"Claude trade validation {action.upper()} {symbol}: {icon} {reason}")
            return approve, reason
    except Exception as e:
        logger.debug(f"Trade validation error: {e}")

    return True, "Validation error – proceeding"


# ── News sentiment (public entry point) ───────────────────────────────────────

def get_news_sentiment(symbol: str) -> float:
    """
    Return sentiment score [-100, +100] for a futures symbol.
    Uses Claude (batch call) when enabled, falls back to VADER.
    """
    coin = symbol.replace("USDT", "").replace("PERP", "").upper()

    if coin in _score_cache:
        ts, val = _score_cache[coin]
        if (datetime.now(timezone.utc) - ts).total_seconds() < _ARTICLE_TTL:
            return val

    # ── Claude path ───────────────────────────────────────────────────────────
    if USE_CLAUDE_SENTIMENT and CLAUDE_API_KEY:
        all_coins = list(COIN_PATTERNS.keys())
        scores    = _get_claude_scores(all_coins)
        if coin in scores:
            result = round(scores[coin], 1)
            _score_cache[coin] = (datetime.now(timezone.utc), result)
            return result

    # ── VADER fallback ────────────────────────────────────────────────────────
    _refresh_articles()
    patterns = _compiled_patterns.get(
        coin, [re.compile(rf"\b{re.escape(coin.lower())}\b", re.IGNORECASE)]
    )
    weighted_sum = weight_total = 0.0
    for art in _articles:
        if any(p.search(art["text"]) for p in patterns):
            compound = _analyzer.polarity_scores(art["text"])["compound"]
            w = _recency_weight(art["age_s"])
            weighted_sum += compound * w
            weight_total += w

    result = round((weighted_sum / weight_total) * 100, 1) if weight_total > 0 else 0.0
    _score_cache[coin] = (datetime.now(timezone.utc), result)
    n = sum(1 for a in _articles if any(p.search(a["text"]) for p in patterns))
    logger.debug(f"VADER sentiment {coin}: {result:+.1f} ({n} articles)")
    return result
