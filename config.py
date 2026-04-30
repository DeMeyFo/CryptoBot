import os
from dotenv import load_dotenv

load_dotenv()

# --- API Credentials ---
API_KEY = os.getenv("BITGET_API_KEY", "")
SECRET_KEY = os.getenv("BITGET_SECRET_KEY", "")
PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")

# --- Mode ---
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# --- Trading Parameters ---
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", "50"))
TOP_COINS_COUNT = int(os.getenv("TOP_COINS_COUNT", "10"))
LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL", "300"))  # 5 minutes

# --- Risk Management ---
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))    # 2%
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.04")) # 4%
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

# --- Technical Analysis ---
CANDLE_INTERVAL = os.getenv("CANDLE_INTERVAL", "15m")
CANDLE_LIMIT = 200

# --- Signal Thresholds ---
LONG_THRESHOLD = 50
SHORT_THRESHOLD = -50

# --- Weights TA vs. News ---
TA_WEIGHT = 0.70
NEWS_WEIGHT = 0.30

# --- Persistence ---
DB_PATH = "crypto_bot.db"
