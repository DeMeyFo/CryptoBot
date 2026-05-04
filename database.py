import sqlite3
import json
from datetime import datetime
from config import DB_PATH


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT    NOT NULL,
                side            TEXT    NOT NULL,
                entry_price     REAL    NOT NULL,
                size_usdt       REAL    NOT NULL,
                leverage        INTEGER NOT NULL,
                stop_loss       REAL,
                take_profit     REAL,
                status          TEXT    DEFAULT 'open',
                exit_price      REAL,
                pnl_usdt        REAL,
                pnl_pct         REAL,
                dry_run         INTEGER DEFAULT 1,
                opened_at       TEXT    NOT NULL,
                closed_at       TEXT,
                signal_score    REAL,
                order_id        TEXT,
                pyramid_count   INTEGER DEFAULT 0,
                is_pyramid      INTEGER DEFAULT 0,
                parent_trade_id INTEGER
            )
        """)
        # Migrate existing tables that are missing pyramid columns
        existing = _cols(c, "trades")
        for col, definition in [
            ("pyramid_count",   "INTEGER DEFAULT 0"),
            ("is_pyramid",      "INTEGER DEFAULT 0"),
            ("parent_trade_id", "INTEGER"),
        ]:
            if col not in existing:
                c.execute(f"ALTER TABLE trades ADD COLUMN {col} {definition}")
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                ta_score    REAL,
                news_score  REAL,
                final_score REAL,
                action      TEXT,
                indicators  TEXT
            )
        """)
        c.commit()


def _cols(conn, table: str) -> list:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def save_trade(symbol, side, entry_price, size_usdt, leverage,
               stop_loss, take_profit, dry_run, signal_score=None, order_id=None,
               is_pyramid=False, parent_trade_id=None) -> int:
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO trades
                (symbol, side, entry_price, size_usdt, leverage,
                 stop_loss, take_profit, dry_run, opened_at, signal_score, order_id,
                 is_pyramid, parent_trade_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (symbol, side, entry_price, size_usdt, leverage,
              stop_loss, take_profit, 1 if dry_run else 0,
              datetime.utcnow().isoformat(), signal_score, order_id,
              1 if is_pyramid else 0, parent_trade_id))
        c.commit()
        return cur.lastrowid


def update_trade_sl(trade_id: int, new_sl: float):
    """Move stop-loss to a new level (used for breakeven on pyramid entry)."""
    with _conn() as c:
        c.execute("UPDATE trades SET stop_loss=? WHERE id=?", (new_sl, trade_id))
        c.commit()


def increment_pyramid_count(trade_id: int):
    with _conn() as c:
        c.execute("UPDATE trades SET pyramid_count = pyramid_count + 1 WHERE id=?", (trade_id,))
        c.commit()


def get_latest_signal_score(symbol: str) -> float:
    """Return the most recently saved final_score for symbol, or 0 if none."""
    with _conn() as c:
        row = c.execute(
            "SELECT final_score FROM signals WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        return row[0] if row else 0.0


def close_trade(trade_id: int, exit_price: float, status: str) -> float:
    with _conn() as c:
        cols = _cols(c, "trades")
        row = c.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if not row:
            return 0.0
        t = dict(zip(cols, row))

        if t["side"] == "long":
            pnl_pct = (exit_price - t["entry_price"]) / t["entry_price"]
        else:
            pnl_pct = (t["entry_price"] - exit_price) / t["entry_price"]

        pnl_usdt = round(t["size_usdt"] * pnl_pct * t["leverage"], 4)

        c.execute("""
            UPDATE trades
            SET status=?, exit_price=?, pnl_usdt=?, pnl_pct=?, closed_at=?
            WHERE id=?
        """, (status, exit_price, pnl_usdt, round(pnl_pct * 100, 4),
              datetime.utcnow().isoformat(), trade_id))
        c.commit()
        return pnl_usdt


def get_open_trades(dry_run=None) -> list:
    with _conn() as c:
        cols = _cols(c, "trades")
        if dry_run is None:
            rows = c.execute("SELECT * FROM trades WHERE status='open'").fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM trades WHERE status='open' AND dry_run=?",
                (1 if dry_run else 0,)
            ).fetchall()
        return [dict(zip(cols, r)) for r in rows]


def get_all_trades(limit: int = 200) -> list:
    with _conn() as c:
        cols = _cols(c, "trades")
        rows = c.execute(
            "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(zip(cols, r)) for r in rows]


def get_total_pnl() -> float:
    with _conn() as c:
        result = c.execute(
            "SELECT SUM(pnl_usdt) FROM trades WHERE status != 'open'"
        ).fetchone()
        return result[0] or 0.0


def save_signal(symbol, ta_score, news_score, final_score, action, indicators):
    with _conn() as c:
        c.execute("""
            INSERT INTO signals (symbol, timestamp, ta_score, news_score, final_score, action, indicators)
            VALUES (?,?,?,?,?,?,?)
        """, (symbol, datetime.utcnow().isoformat(),
              ta_score, news_score, final_score, action, json.dumps(indicators)))
        c.commit()


def get_latest_signals(limit: int = 30) -> list:
    with _conn() as c:
        cols = _cols(c, "signals")
        rows = c.execute("""
            SELECT s.*
            FROM signals s
            INNER JOIN (
                SELECT symbol, MAX(timestamp) AS max_ts
                FROM signals
                GROUP BY symbol
            ) latest ON s.symbol = latest.symbol AND s.timestamp = latest.max_ts
            ORDER BY ABS(s.final_score) DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(zip(cols, r)) for r in rows]


def get_pnl_history() -> list:
    with _conn() as c:
        rows = c.execute("""
            SELECT closed_at, pnl_usdt
            FROM trades
            WHERE status != 'open' AND closed_at IS NOT NULL
            ORDER BY closed_at ASC
        """).fetchall()
        return [{"closed_at": r[0], "pnl_usdt": r[1]} for r in rows]
