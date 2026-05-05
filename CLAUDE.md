# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A crypto futures trading bot for Bitget exchange. Combines technical analysis, news sentiment (VADER + optional Claude AI), Fear & Greed Index, and funding rate signals to make long/short decisions on a fixed whitelist of coins. Runs 24/7 on a VPS via systemd.

## Running the bot

```bash
# Start (dry-run by default – no real orders)
python main.py

# Backtest against historical data (Binance as data source)
python backtest.py --months 3 --granularity 15m --threshold 50
python backtest.py BTCUSDT SOLUSDT --months 3
python backtest.py --trades   # also print each individual trade

# Dashboard
streamlit run dashboard.py
```

**VPS deployment:**
```bash
sudo systemctl start|stop|restart|status cryptobot
sudo journalctl -u cryptobot -f          # live logs
sqlite3 ~/CryptoBot/crypto_bot.db "SELECT symbol, side, status, pnl_usdt FROM trades ORDER BY opened_at DESC LIMIT 20;"
```

## Architecture

### Signal pipeline (`strategy.py → analyze_symbol`)

Every 5 minutes per symbol, in order:

1. **ADX gate** – if ADX < 20 (no trend), immediately return HOLD. No API calls wasted.
2. **TA score** (`technical_analysis.py`) – EMA crossover, RSI, Stochastic RSI, MACD, Bollinger Bands, Supertrend, ATR, Volume. Returns score ±100.
3. **News sentiment** (`news_sentiment.py`) – 10 RSS feeds, recency-weighted. Claude Haiku when `USE_CLAUDE_SENTIMENT=true`, else VADER.
4. **Fear & Greed** – alternative.me API, cached 1h. Extreme fear=+100, extreme greed=−100.
5. **Funding rate** (`bitget_client.get_funding_rate`) – from Bitget API. Crowded longs→bearish, crowded shorts→bullish.
6. **Weighted combination**: `TA×0.55 + News×0.15 + FG×0.10 + Funding×0.20`
7. **ADX strength factor** – weak trend dampens, strong trend boosts score.
8. **Multi-timeframe factor** – confirms 15m signal against 1H candles. Agreement ×1.15, conflict ×0.70.
9. **Market regime factor** – Claude detects bull/bear/sideways every 4h. Bull boosts longs ×1.15, bear boosts shorts ×1.15, sideways dampens all ×0.85.
10. **Entry threshold**: `|final_score| ≥ 55` triggers LONG or SHORT.

### Main loop (`main.py`)

Two independent intervals:
- **Every 60s**: `manage_open_trades()` – trailing stop, SL/TP check, Claude emergency exit, pyramiding
- **Every 300s**: `scan_new_entries()` – runs the full signal pipeline, Claude trade validation before placing order

### Coin selection (`bitget_client.WHITELIST`)

Fixed whitelist of 6 backtested coins ordered by Profit Factor. **Do not switch back to volume-based discovery** – it consistently picked micro-caps and meme coins with 0% win rates. To add a coin, backtest it first with `backtest.py --months 3`, require PF > 1.5 and WR > 40%.

### Claude AI integration (`news_sentiment.py`)

Four features, all optional via `USE_CLAUDE_SENTIMENT=true`:

| Function | When called | Cost |
|---|---|---|
| `_get_claude_scores()` | Every 15min (news refresh) | ~7€/month |
| Exit signals | Piggybacked on news call (same request) | +0€ |
| `get_market_regime()` | Every 4h | +0.50€/month |
| `validate_trade()` | Once per trade signal | +0.05€/month |

The batch news call includes open positions and asks Claude to flag urgent exits simultaneously – no separate API call. Model: `claude-haiku-4-5-20251001`.

### Position management

- **Dynamic sizing**: score 55–67 → `MIN_POSITION_USDT` (120), 67–80 → `POSITION_SIZE_USDT` (200), >80 → `MAX_POSITION_USDT` (400)
- **ATR-based SL/TP**: SL = entry ± 2×ATR, TP = entry ± 3.5×ATR
- **Trailing stop**: 2% trail, ratchets every 60s
- **Pyramiding**: when trade reaches 50% of TP distance AND latest signal score ≥ 70 → add 50% position, move original SL to break-even
- **Pyramid add-ons** do NOT count against `MAX_OPEN_POSITIONS`

### Database (`database.py`)

SQLite at `crypto_bot.db`. Two tables: `trades` and `signals`. The `trades` table has `is_pyramid`, `parent_trade_id`, `pyramid_count` columns for pyramiding. `update_trade_sl()` is used by both trailing stop and pyramiding.

### Backtest data sources

- **Bitget** (default): ~1000 candles ≈ 10 days on 15m
- **Binance** (`--months N`): paginated history up to 6 months. Falls back to Bitget for coins not listed on Binance (CLUSDT, BUSDT, RAVEUSDT are Bitget-specific).

## Key configuration (`.env`)

```
DRY_RUN=true                  # ALWAYS test dry-run first
LEVERAGE=5                    # do NOT increase – win rate ~40-50% makes higher leverage dangerous
POSITION_SIZE_USDT=200        # mid-conviction default
MIN_POSITION_USDT=120
MAX_POSITION_USDT=400
MAX_OPEN_POSITIONS=5
TRAILING_STOP_PCT=0.02
CONFIRM_TIMEFRAME=1H          # must be uppercase H for Bitget API
CLAUDE_API_KEY=...
USE_CLAUDE_SENTIMENT=true
```

## Critical constraints

- **Bitget granularity format**: hours require uppercase (`1H`, `4H`), minutes lowercase (`15m`). Wrong case → 400 error.
- **Signal weights must sum to 1.0** – `TA_WEIGHT + NEWS_WEIGHT + FEAR_GREED_WEIGHT + FUNDING_WEIGHT`.
- **Backtest threshold ≠ live threshold** – the backtest uses raw TA score (0–100) vs `--threshold 50`. The live bot applies the combined weighted score vs `LONG_THRESHOLD=55`. Always run backtest with `--threshold 50` for meaningful trade counts.
- **VADER fallback is always active** – if Claude is disabled or errors, `get_news_sentiment()` automatically falls back to VADER. Never remove this fallback.
- **Pyramid trades**: `is_pyramid=True` trades are excluded from `MAX_OPEN_POSITIONS` counting. Don't change `_open_position_count()` without accounting for this.
