import os
from dotenv import load_dotenv

load_dotenv()

# --- API Credentials ---
API_KEY    = os.getenv("BITGET_API_KEY", "")
SECRET_KEY = os.getenv("BITGET_SECRET_KEY", "")
PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")

# --- Mode ---
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# --- Trading Parameters ---
LEVERAGE             = int(os.getenv("LEVERAGE", "5"))
POSITION_SIZE_USDT   = float(os.getenv("POSITION_SIZE_USDT", "50"))   # default / mid tier
MIN_POSITION_USDT    = float(os.getenv("MIN_POSITION_USDT", "30"))    # low-conviction trades
MAX_POSITION_USDT    = float(os.getenv("MAX_POSITION_USDT", "100"))   # high-conviction trades
HIGH_CONVICTION_SCORE = float(os.getenv("HIGH_CONVICTION_SCORE", "80"))  # score → MAX size
TOP_COINS_COUNT      = int(os.getenv("TOP_COINS_COUNT", "20"))         # was 10
LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL", "300"))

# --- Risk Management ---
STOP_LOSS_PCT          = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT        = float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
MAX_OPEN_POSITIONS     = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
DAILY_LOSS_LIMIT_USDT  = float(os.getenv("DAILY_LOSS_LIMIT_USDT", "300"))

# --- Trailing Stop ---
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.02"))  # 2% trail distance

# --- Technical Analysis ---
CANDLE_INTERVAL = os.getenv("CANDLE_INTERVAL", "15m")
CANDLE_LIMIT    = 200

# --- ADX Trend Filter ---
ADX_NO_TREND    = float(os.getenv("ADX_NO_TREND", "18"))   # below → skip trade
ADX_WEAK_TREND  = float(os.getenv("ADX_WEAK_TREND", "25")) # below → dampen score 30%
ADX_STRONG_TREND = float(os.getenv("ADX_STRONG_TREND", "40")) # above → boost score 20%

# --- Signal Thresholds ---
LONG_THRESHOLD  = 45
SHORT_THRESHOLD = -45

# --- Signal Weights (must sum to 1.0) ---
TA_WEIGHT          = 0.50
NEWS_WEIGHT        = 0.15
FEAR_GREED_WEIGHT  = 0.10
FUNDING_WEIGHT     = 0.15
OI_WEIGHT          = 0.10

# --- Multi-Timeframe Confirmation ---
CONFIRM_TIMEFRAME = os.getenv("CONFIRM_TIMEFRAME", "1H")

# --- ATR-based dynamic SL/TP ---
USE_ATR_SL_TP     = os.getenv("USE_ATR_SL_TP", "true").lower() == "true"
ATR_SL_MULTIPLIER = float(os.getenv("ATR_SL_MULTIPLIER", "2.0"))
ATR_TP_MULTIPLIER = float(os.getenv("ATR_TP_MULTIPLIER", "3.5"))

# --- Claude AI Sentiment (optional) ---
CLAUDE_API_KEY        = os.getenv("CLAUDE_API_KEY", "")
USE_CLAUDE_SENTIMENT  = os.getenv("USE_CLAUDE_SENTIMENT", "false").lower() == "true"

# --- Telegram Notifications ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Persistence ---
DB_PATH = "crypto_bot.db"
