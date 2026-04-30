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
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT    NOT NULL,
                side          TEXT    NOT NULL,
                entry_price   REAL    NOT NULL,
                size_usdt     REAL    NOT NULL,
                leverage      INTEGER NOT NULL,
                stop_loss     REAL,
                take_profit   REAL,
                status        TEXT    DEFAULT 'open',
                exit_price    REAL,
                pnl_usdt      REAL,
                pnl_pct       REAL,
                dry_run       INTEGER DEFAULT 1,
                opened_at     TEXT    NOT NULL,
                closed_at     TEXT,
                signal_score  REAL,
                order_id      TEXT
            )
        """)
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
               stop_loss, take_profit, dry_run, signal_score=None, order_id=None) -> int:
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO trades
                (symbol, side, entry_price, size_usdt, leverage,
                 stop_loss, take_profit, dry_run, opened_at, signal_score, order_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (symbol, side, entry_price, size_usdt, leverage,
              stop_loss, take_profit, 1 if dry_run else 0,
              datetime.utcnow().isoformat(), signal_score, order_id))
        c.commit()
        return cur.lastrowid


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
