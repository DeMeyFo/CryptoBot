import os
from dotenv import load_dotenv

load_dotenv()

# --- API Credentials ---
API_KEY = os.getenv("BITGET_API_KEY", "")
SECRET_KEY = os.getenv("BITGET_SECRET_KEY", "")
PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")

# --- Mode ---
# Never switch this default to false. Live trading must be enabled explicitly.
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# --- Trading Parameters ---
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
# All position values are order notional, not margin. Leverage only changes margin usage.
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "50"))
MIN_POSITION_USDT = float(os.getenv("MIN_POSITION_USDT", "30"))
MAX_POSITION_USDT = float(os.getenv("MAX_POSITION_USDT", str(POSITION_SIZE_USDT)))
HIGH_CONVICTION_SCORE = float(os.getenv("HIGH_CONVICTION_SCORE", "90"))
# Dynamic universe rotation: scan top-volume Perps beyond the base universe.
# Default OFF: the scanner also picks up untested high-volume coins (e.g. newly
# listed tokens) that were never backtested and caused outsized losses. Only the
# 12 validated RESEARCH_UNIVERSE coins are traded unless explicitly re-enabled.
DYNAMIC_UNIVERSE_ENABLED = os.getenv("DYNAMIC_UNIVERSE_ENABLED", "false").lower() == "true"
DYNAMIC_UNIVERSE_MAX = int(os.getenv("DYNAMIC_UNIVERSE_MAX", "20"))
TOP_COINS_COUNT = int(os.getenv("TOP_COINS_COUNT", "12"))
LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL", "300"))

# --- Risk and execution costs ---
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.03"))
TAKER_FEE_RATE = float(os.getenv("TAKER_FEE_RATE", "0.0006"))
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.0002"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "8"))
DAILY_LOSS_LIMIT_USDT = float(os.getenv("DAILY_LOSS_LIMIT_USDT", "25"))

# --- Direction filter ---
# Portfolio study over 24 months, 8 symbols, 1H: shorts had a lower profit
# factor than longs in both independent halves (1.202 vs 1.318 and 1.043 vs
# 1.273), and trading both directions roughly doubled the drawdown in the
# second half (62.0% vs 37.5%). Long-only is therefore the validated default.
_TRADE_DIRECTIONS = {
    "long": ("long",),
    "short": ("short",),
    "both": ("long", "short"),
}
TRADE_DIRECTION = os.getenv("TRADE_DIRECTION", "long").strip().lower()
if TRADE_DIRECTION not in _TRADE_DIRECTIONS:
    raise ValueError(
        f"TRADE_DIRECTION must be one of {sorted(_TRADE_DIRECTIONS)}, "
        f"got {TRADE_DIRECTION!r}"
    )
ALLOWED_TRADE_SIDES = _TRADE_DIRECTIONS[TRADE_DIRECTION]

# --- Equity-proportional sizing ---
# Absolute USDT caps make the percentage return shrink as the account grows:
# a fixed 200 USDT notional is 20% of a 1k account but 4% of a 5k account, so
# the same strategy would report a falling monthly return on a larger balance.
# "percent" scales the ceiling and floor with equity; "absolute" keeps the
# legacy POSITION_SIZE_USDT / MAX_POSITION_USDT / MIN_POSITION_USDT behaviour.
SIZING_MODE = os.getenv("SIZING_MODE", "percent").strip().lower()
if SIZING_MODE not in ("percent", "absolute"):
    raise ValueError(
        f"SIZING_MODE must be 'percent' or 'absolute', got {SIZING_MODE!r}"
    )
# Ceiling for a single entry before margin and exposure limits apply. Risk per
# trade is normally the binding constraint; this only caps pathological cases
# such as a very tight stop.
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "1.0"))
# Entries below this fraction of equity are skipped instead of being rounded up.
MIN_POSITION_PCT = float(os.getenv("MIN_POSITION_PCT", "0.01"))
# Daily realised loss that blocks further entries until the next UTC day.
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.05"))

# --- Exposure ceilings, as a multiple of equity ---
# Risk-based sizing bounds the loss only while the stop fills at its price.
# These bound the loss when it does not: gap, liquidity hole, dead process.
# Measured cost over 24 months at 0.5% risk: none (peak was 2.22x gross and
# 0.67x per symbol). At 1% risk: 3.10% -> 3.08% monthly, drawdown 37.5% -> 36.9%.
MAX_SYMBOL_EXPOSURE_PCT = float(os.getenv("MAX_SYMBOL_EXPOSURE_PCT", "1.5"))
MAX_GROSS_EXPOSURE_PCT = float(os.getenv("MAX_GROSS_EXPOSURE_PCT", "3.5"))

# --- Pyramiding (adding to a winner) ---
# Default off. Over the full 24 months half-size adds raised the compounded
# monthly return from 3.10% to 4.35% with the best profit factor of the study
# (1.365), but the benefit is concentrated in trending regimes: it won two of
# four independent half-years and lost the other two, and the holdout half fell
# from 2.16% to 0.96% per month. Enable only as a deliberate regime bet.
PYRAMID_MAX_ADDS = max(0, int(os.getenv("PYRAMID_MAX_ADDS", "0")))
# Favourable excursion between adds, measured in units of the original entry
# risk (entry to initial stop).
PYRAMID_STEP_R = float(os.getenv("PYRAMID_STEP_R", "1.0"))
# Add size relative to the base risk per trade. Half size measured a better
# profit factor and a lower drawdown than full size.
PYRAMID_RISK_FRACTION = float(os.getenv("PYRAMID_RISK_FRACTION", "0.5"))

# --- Funding cashflow reconciliation ---
# The runtime clamps synchronization to at most once per minute. The overlap
# catches delayed bills and restarts; database event IDs provide final dedupe.
FUNDING_SYNC_INTERVAL_SECONDS = max(
    60, int(os.getenv("FUNDING_SYNC_INTERVAL_SECONDS", "60"))
)
FUNDING_SYNC_OVERLAP_SECONDS = max(
    0, int(os.getenv("FUNDING_SYNC_OVERLAP_SECONDS", "86400"))
)
FUNDING_SYNC_CLOSE_GRACE_SECONDS = max(
    FUNDING_SYNC_OVERLAP_SECONDS,
    int(os.getenv("FUNDING_SYNC_CLOSE_GRACE_SECONDS", "86400")),
)
FUNDING_SYNC_PAGE_SIZE = min(
    100, max(1, int(os.getenv("FUNDING_SYNC_PAGE_SIZE", "100")))
)
FUNDING_SYNC_MAX_PAGES = min(
    100, max(1, int(os.getenv("FUNDING_SYNC_MAX_PAGES", "10")))
)
FUNDING_MARK_WINDOW_MS = min(
    900_000, max(60_000, int(os.getenv("FUNDING_MARK_WINDOW_MS", "180000")))
)
FUNDING_RATE_MATCH_TOLERANCE_SECONDS = max(
    0, int(os.getenv("FUNDING_RATE_MATCH_TOLERANCE_SECONDS", "300"))
)

# --- Fallback exits ---
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.025"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))
# trend-breakout-v2 was validated without a fixed target. Do not allow an
# environment override to silently create a different, untested strategy.
USE_FIXED_TAKE_PROFIT = False

# --- Validated trend-breakout strategy ---
# A new variable intentionally avoids stale CANDLE_INTERVAL=15m values in old .env files.
STRATEGY_TIMEFRAME = os.getenv("STRATEGY_TIMEFRAME", "1H")
CANDLE_INTERVAL = STRATEGY_TIMEFRAME  # backwards-compatible alias
CANDLE_LIMIT = max(int(os.getenv("CANDLE_LIMIT", "1000")), 1000)
EMA_HISTORY_CANDLES = 750
EMA_FAST_PERIOD = int(os.getenv("EMA_FAST_PERIOD", "20"))
EMA_MID_PERIOD = int(os.getenv("EMA_MID_PERIOD", "50"))
EMA_SLOW_PERIOD = int(os.getenv("EMA_SLOW_PERIOD", "200"))
DONCHIAN_PERIOD = int(os.getenv("DONCHIAN_PERIOD", "55"))
# Trend-strength gate. Raised from 25 after measuring that losing months come
# from many small stop-outs in chop, not from giving back winners. Effect held
# in both independent halves of the 24-month study and across the whole 30-45
# range, so it is a plateau rather than a fitted spike:
#   train  : drawdown 17.5% -> 6.9%, profit factor 1.358 -> 2.190
#   holdout: drawdown 20.8% -> 6.5%, profit factor 1.327 -> 1.544,
#            worst rolling 6 months -15.7% -> -1.9%
# It roughly halves the trade count, which is the main statistical caveat.
ADX_ENTRY_THRESHOLD = float(os.getenv("ADX_ENTRY_THRESHOLD", "40"))
# Entries are accepted only shortly after the signal candle closed.
MAX_SIGNAL_AGE_SECONDS = int(os.getenv("MAX_SIGNAL_AGE_SECONDS", "180"))

# Retained aliases for older imports and dashboards.
ADX_NO_TREND = ADX_ENTRY_THRESHOLD
ADX_WEAK_TREND = float(os.getenv("ADX_WEAK_TREND", "25"))
ADX_STRONG_TREND = float(os.getenv("ADX_STRONG_TREND", "40"))
LONG_THRESHOLD = float(os.getenv("LONG_THRESHOLD", "65"))
SHORT_THRESHOLD = -LONG_THRESHOLD

# --- ATR risk management ---
USE_ATR_SL_TP = True
TREND_STOP_ATR_MULTIPLIER = float(os.getenv("TREND_STOP_ATR_MULTIPLIER", "2.5"))
TREND_TRAIL_ATR_MULTIPLIER = float(os.getenv("TREND_TRAIL_ATR_MULTIPLIER", "2.5"))
TRAILING_ACTIVATION_R = float(os.getenv("TRAILING_ACTIVATION_R", "1.0"))
ATR_SL_MULTIPLIER = TREND_STOP_ATR_MULTIPLIER
ATR_TP_MULTIPLIER = float(os.getenv("ATR_TP_MULTIPLIER", "4.0"))
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.02"))

# Compatibility weights. Alternative data is diagnostic only.
TA_WEIGHT = 1.0
NEWS_WEIGHT = 0.0
FEAR_GREED_WEIGHT = 0.0
FUNDING_WEIGHT = 0.0
OI_WEIGHT = 0.0
CONFIRM_TIMEFRAME = os.getenv("CONFIRM_TIMEFRAME", "4H")

# --- Limit-Order Entry ---
# "market" uses taker fees (0.06%) with slippage. "limit" places a limit
# order at close - pullback_atr * ATR, getting maker fees (0.02%) and better
# fills. The order expires after one candle if not filled.
# Measured effect: Sortino 2.13 -> 2.58, median month +3.14% -> +4.06%.
ENTRY_ORDER_TYPE = os.getenv("ENTRY_ORDER_TYPE", "limit").strip().lower()
if ENTRY_ORDER_TYPE not in ("market", "limit"):
    raise ValueError(f"ENTRY_ORDER_TYPE must be 'market' or 'limit', got {ENTRY_ORDER_TYPE!r}")
LIMIT_PULLBACK_ATR = float(os.getenv("LIMIT_PULLBACK_ATR", "0.3"))

# --- Volatility-Adaptive Risk ---
# Scales risk inversely with realised volatility: lower risk in volatile
# markets (wider tails, more gap risk), higher in calm markets.
# Measured effect: positive months 52% -> 56%, PF 1.776 -> 1.849.
VOL_ADAPTIVE_ENABLED = os.getenv("VOL_ADAPTIVE_ENABLED", "true").lower() == "true"
VOL_ADAPTIVE_LOOKBACK = int(os.getenv("VOL_ADAPTIVE_LOOKBACK", "336"))
VOL_ADAPTIVE_LOW_MULT = float(os.getenv("VOL_ADAPTIVE_LOW_MULT", "1.3"))
VOL_ADAPTIVE_HIGH_MULT = float(os.getenv("VOL_ADAPTIVE_HIGH_MULT", "0.7"))

# v3 changes the entry contract (direction filter) and the sizing contract
# (equity-proportional caps, exposure ceilings), so its signals and trades must
# not be pooled with v2 results.
# --- Secondary timeframe sleeve (80/20 diversification) ---
# The primary 1H sleeve runs with RISK_PER_TRADE_PCT. The secondary 4H sleeve
# runs with its own risk budget. Measured effect: Sortino 1.61 -> 2.13,
# Drawdown 18.3% -> 15.2%, median month +1.32% -> +3.14%.
SECONDARY_ENABLED = os.getenv("SECONDARY_ENABLED", "true").lower() == "true"
SECONDARY_TIMEFRAME = os.getenv("SECONDARY_TIMEFRAME", "4H")
SECONDARY_RISK_PER_TRADE_PCT = float(os.getenv("SECONDARY_RISK_PER_TRADE_PCT", "0.005"))

STRATEGY_VERSION = "trend-breakout-v3"

# --- Claude AI Sentiment (diagnostic only for this strategy) ---
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
USE_CLAUDE_SENTIMENT = os.getenv("USE_CLAUDE_SENTIMENT", "false").lower() == "true"

# --- Telegram Notifications ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Persistence ---
DB_PATH = "crypto_bot.db"
