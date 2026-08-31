import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone

from config import (
    DB_PATH,
    DRY_RUN,
    FUNDING_SYNC_INTERVAL_SECONDS,
    STRATEGY_VERSION,
    TAKER_FEE_RATE,
)
from ml_prediction_log import init_ml_tables
from risk_manager import effective_entry_price


def _conn():
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _cols(conn, table: str) -> list:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _mode(dry_run: bool) -> str:
    return "dry_run" if dry_run else "live"


def _closed_price_fee_expression(prefix: str = "") -> str:
    column = lambda name: f"{prefix}{name}"
    return (
        "CASE "
        f"WHEN COALESCE({column('pnl_model_version')}, 1) IN (2, 3) "
        f"AND {column('funding_usdt')} IS NOT NULL "
        f"THEN COALESCE({column('pnl_usdt')}, 0) - {column('funding_usdt')} "
        f"ELSE COALESCE({column('pnl_usdt')}, 0) END"
    )


def init_db():
    with _conn() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol             TEXT    NOT NULL,
                side               TEXT    NOT NULL,
                entry_price        REAL    NOT NULL,
                size_usdt          REAL    NOT NULL,
                quantity           REAL,
                leverage           INTEGER NOT NULL,
                stop_loss          REAL,
                initial_stop_loss  REAL,
                take_profit        REAL,
                status             TEXT    DEFAULT 'open',
                exit_price         REAL,
                pnl_usdt           REAL,
                pnl_pct            REAL,
                fees_usdt          REAL    DEFAULT 0,
                funding_usdt       REAL    DEFAULT 0,
                funding_synced_at  TEXT,
                dry_run            INTEGER DEFAULT 1,
                opened_at          TEXT    NOT NULL,
                closed_at          TEXT,
                signal_score       REAL,
                order_id           TEXT,
                exit_order_id      TEXT,
                best_price         REAL,
                entry_atr          REAL,
                last_trail_at      TEXT,
                strategy_version   TEXT,
                pnl_model_version  INTEGER DEFAULT 3,
                pyramid_count      INTEGER DEFAULT 0,
                is_pyramid         INTEGER DEFAULT 0,
                parent_trade_id    INTEGER
            )
        """)
        existing = _cols(connection, "trades")
        needs_pnl_migration = "pnl_model_version" not in existing
        for column, definition in [
            ("quantity", "REAL"),
            ("initial_stop_loss", "REAL"),
            ("fees_usdt", "REAL DEFAULT 0"),
            ("funding_usdt", "REAL"),
            ("funding_synced_at", "TEXT"),
            ("exit_order_id", "TEXT"),
            ("best_price", "REAL"),
            ("entry_atr", "REAL"),
            ("last_trail_at", "TEXT"),
            ("strategy_version", "TEXT"),
            ("pnl_model_version", "INTEGER DEFAULT 1"),
            ("pyramid_count", "INTEGER DEFAULT 0"),
            ("is_pyramid", "INTEGER DEFAULT 0"),
            ("parent_trade_id", "INTEGER"),
            # Volume-weighted entry after pyramid adds. NULL means "no adds", so
            # entry_price is the effective entry. entry_price itself is never
            # rewritten, because the trailing stop and the R measurement are
            # defined against the original entry.
            ("avg_entry_price", "REAL"),
        ]:
            if column not in existing:
                connection.execute(f"ALTER TABLE trades ADD COLUMN {column} {definition}")

        # Correct only the old leverage double-count. Historic fees and funding
        # are unknown, therefore these records deliberately remain model v1.
        if needs_pnl_migration:
            connection.execute("""
                UPDATE trades
                SET pnl_usdt=ROUND(size_usdt * pnl_pct / 100.0, 4),
                    fees_usdt=0,
                    pnl_model_version=1
                WHERE status!='open' AND pnl_pct IS NOT NULL
            """)
            connection.execute(
                "UPDATE trades SET pnl_model_version=1 WHERE status='open'"
            )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS funding_events (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                source           TEXT NOT NULL,
                external_id      TEXT NOT NULL,
                trade_id         INTEGER NOT NULL,
                symbol           TEXT NOT NULL,
                settlement_at    TEXT NOT NULL,
                funding_rate     REAL,
                mark_price       REAL,
                amount_usdt      REAL NOT NULL,
                mode             TEXT NOT NULL CHECK(mode IN ('dry_run', 'live')),
                raw_type         TEXT,
                created_at       TEXT NOT NULL,
                UNIQUE(source, external_id),
                FOREIGN KEY(trade_id) REFERENCES trades(id)
            )
        """)

        # Funding for a past settlement must be modeled with the quantity that
        # was actually held at that moment. Without this history a pyramid add
        # would retroactively change the amount of settlements that happened
        # before it, because only the current quantity would be known.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS trade_quantity_epochs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id       INTEGER NOT NULL,
                effective_from TEXT    NOT NULL,
                quantity       REAL    NOT NULL,
                size_usdt      REAL    NOT NULL,
                entry_price    REAL    NOT NULL,
                reason         TEXT    NOT NULL,
                created_at     TEXT    NOT NULL,
                UNIQUE(trade_id, effective_from, reason),
                FOREIGN KEY(trade_id) REFERENCES trades(id)
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol           TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                candle_timestamp TEXT,
                ta_score         REAL,
                news_score       REAL,
                final_score      REAL,
                action           TEXT,
                indicators       TEXT,
                dry_run          INTEGER,
                strategy_version TEXT
            )
        """)
        signal_columns = _cols(connection, "signals")
        for column, definition in [
            ("candle_timestamp", "TEXT"),
            ("dry_run", "INTEGER"),
            ("strategy_version", "TEXT"),
            ("ml_prediction_id", "INTEGER"),
        ]:
            if column not in signal_columns:
                connection.execute(f"ALTER TABLE signals ADD COLUMN {column} {definition}")

        connection.execute("""
            CREATE TABLE IF NOT EXISTS consumed_entry_signals (
                symbol           TEXT NOT NULL,
                candle_timestamp TEXT NOT NULL,
                dry_run          INTEGER NOT NULL,
                strategy_version TEXT NOT NULL,
                consumed_at      TEXT NOT NULL,
                order_id         TEXT,
                PRIMARY KEY (symbol, candle_timestamp, dry_run, strategy_version)
            )
        """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_status_mode ON trades(status, dry_run)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals(symbol, timestamp)"
        )
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_funding_events_trade_time
            ON funding_events(trade_id, settlement_at)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_funding_events_mode_time
            ON funding_events(mode, settlement_at)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_quantity_epochs_trade_time
            ON trade_quantity_epochs(trade_id, effective_from)
        """)
        connection.commit()

    # ML-Vorhersage-Tabellen anlegen (Anforderung 14.4)
    init_ml_tables()


def save_trade(symbol, side, entry_price, size_usdt, leverage,
               stop_loss, take_profit, dry_run, signal_score=None, order_id=None,
               is_pyramid=False, parent_trade_id=None, quantity=None,
               initial_stop_loss=None, best_price=None, entry_atr=None,
               last_trail_at=None, strategy_version=STRATEGY_VERSION) -> int:
    initial_stop_loss = stop_loss if initial_stop_loss is None else initial_stop_loss
    best_price = entry_price if best_price is None else best_price
    opened_at = _utc_now()
    with _conn() as connection:
        cursor = connection.execute("""
            INSERT INTO trades
                (symbol, side, entry_price, size_usdt, quantity, leverage,
                 stop_loss, initial_stop_loss, take_profit, dry_run, opened_at,
                 signal_score, order_id, best_price, entry_atr, last_trail_at,
                 strategy_version, pnl_model_version, funding_usdt,
                 funding_synced_at, is_pyramid, parent_trade_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            symbol, side, entry_price, size_usdt, quantity, leverage,
            stop_loss, initial_stop_loss, take_profit, 1 if dry_run else 0,
            opened_at, signal_score, order_id, best_price, entry_atr,
            last_trail_at, strategy_version, 3, 0.0, opened_at,
            1 if is_pyramid else 0, parent_trade_id,
        ))
        trade_id = cursor.lastrowid
        # The opening epoch is written in the same transaction as the trade, so
        # a trade can never exist with an incomplete quantity history.
        if quantity and float(quantity) > 0:
            connection.execute("""
                INSERT OR IGNORE INTO trade_quantity_epochs
                    (trade_id, effective_from, quantity, size_usdt, entry_price,
                     reason, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (
                trade_id, opened_at, float(quantity), float(size_usdt),
                float(entry_price), "open", opened_at,
            ))
        connection.commit()
        return trade_id


def quantity_at(trade_id: int, settlement_at: str) -> float | None:
    """
    Quantity held by this trade at `settlement_at`.

    Returns None when the trade has no recorded quantity history, which is the
    case for every trade opened before this table existed. Callers must then
    fall back to the current quantity; that is exact for any trade that never
    received a pyramid add.
    """
    with _conn() as connection:
        row = connection.execute("""
            SELECT quantity FROM trade_quantity_epochs
            WHERE trade_id=? AND effective_from<=?
            ORDER BY effective_from DESC, id DESC LIMIT 1
        """, (trade_id, settlement_at)).fetchone()
        if row is not None:
            return float(row[0])
        # A settlement earlier than the first epoch means the trade held nothing
        # yet. Distinguish that from "no history at all", which returns None.
        has_history = connection.execute(
            "SELECT 1 FROM trade_quantity_epochs WHERE trade_id=? LIMIT 1",
            (trade_id,),
        ).fetchone()
        return 0.0 if has_history else None


def get_quantity_epochs(trade_id: int) -> list:
    with _conn() as connection:
        columns = _cols(connection, "trade_quantity_epochs")
        rows = connection.execute("""
            SELECT * FROM trade_quantity_epochs
            WHERE trade_id=? ORDER BY effective_from, id
        """, (trade_id,)).fetchall()
        return [dict(zip(columns, row)) for row in rows]


def apply_pyramid_add(trade_id: int, add_quantity: float, add_price: float,
                      effective_from: str, max_adds: int) -> dict | None:
    """
    Add to an open position and record the resulting quantity epoch atomically.

    Returns the new position state, or None when the add was rejected because
    the trade is closed, its quantity is unknown, or it already holds the
    configured number of adds. Serialised against close_trade() and funding
    event inserts by the same immediate write lock, so a settlement can never
    be booked against a half-applied add.
    """
    add_quantity = float(add_quantity)
    add_price = float(add_price)
    if add_quantity <= 0 or add_price <= 0:
        return None

    connection = _conn()
    try:
        connection.execute("BEGIN IMMEDIATE")
        columns = _cols(connection, "trades")
        row = connection.execute(
            "SELECT * FROM trades WHERE id=? AND status='open'", (trade_id,)
        ).fetchone()
        if not row:
            connection.commit()
            return None
        trade = dict(zip(columns, row))

        held_quantity = float(trade.get("quantity") or 0)
        if held_quantity <= 0:
            connection.commit()
            return None
        if int(trade.get("pyramid_count") or 0) >= int(max_adds):
            connection.commit()
            return None

        previous_entry = float(
            trade.get("avg_entry_price") or trade.get("entry_price") or 0
        )
        if previous_entry <= 0:
            connection.commit()
            return None

        new_quantity = held_quantity + add_quantity
        new_avg_entry = (
            previous_entry * held_quantity + add_price * add_quantity
        ) / new_quantity
        new_size = float(trade.get("size_usdt") or 0) + add_quantity * add_price

        connection.execute("""
            UPDATE trades
            SET quantity=?, size_usdt=?, avg_entry_price=?,
                pyramid_count=pyramid_count+1
            WHERE id=? AND status='open'
        """, (new_quantity, new_size, new_avg_entry, trade_id))
        connection.execute("""
            INSERT OR IGNORE INTO trade_quantity_epochs
                (trade_id, effective_from, quantity, size_usdt, entry_price,
                 reason, created_at)
            VALUES (?,?,?,?,?,?,?)
        """, (
            trade_id, effective_from, new_quantity, new_size, new_avg_entry,
            "pyramid", _utc_now(),
        ))
        connection.commit()
        return {
            "trade_id": trade_id,
            "quantity": new_quantity,
            "size_usdt": new_size,
            "avg_entry_price": new_avg_entry,
            "added_quantity": add_quantity,
            "added_notional": add_quantity * add_price,
            "pyramid_count": int(trade.get("pyramid_count") or 0) + 1,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def update_trade_sl(trade_id: int, new_sl: float):
    with _conn() as connection:
        connection.execute("UPDATE trades SET stop_loss=? WHERE id=?", (new_sl, trade_id))
        connection.commit()


def update_trade_management(trade_id: int, new_sl: float, best_price: float,
                            last_trail_at: str):
    with _conn() as connection:
        connection.execute("""
            UPDATE trades
            SET stop_loss=?, best_price=?, last_trail_at=?
            WHERE id=? AND status='open'
        """, (new_sl, best_price, last_trail_at, trade_id))
        connection.commit()


def increment_pyramid_count(trade_id: int):
    with _conn() as connection:
        connection.execute(
            "UPDATE trades SET pyramid_count=pyramid_count+1 WHERE id=?", (trade_id,)
        )
        connection.commit()


def get_latest_signal_score(symbol: str) -> float:
    with _conn() as connection:
        row = connection.execute(
            "SELECT final_score FROM signals WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return row[0] if row else 0.0


def is_entry_signal_consumed(symbol: str, candle_timestamp: str,
                             dry_run: bool, strategy_version=STRATEGY_VERSION) -> bool:
    with _conn() as connection:
        row = connection.execute("""
            SELECT 1 FROM consumed_entry_signals
            WHERE symbol=? AND candle_timestamp=? AND dry_run=? AND strategy_version=?
        """, (symbol, candle_timestamp, 1 if dry_run else 0, strategy_version)).fetchone()
        return row is not None


def mark_entry_signal_consumed(symbol: str, candle_timestamp: str, dry_run: bool,
                               order_id: str | None,
                               strategy_version=STRATEGY_VERSION):
    with _conn() as connection:
        connection.execute("""
            INSERT OR IGNORE INTO consumed_entry_signals
                (symbol, candle_timestamp, dry_run, strategy_version, consumed_at, order_id)
            VALUES (?,?,?,?,?,?)
        """, (
            symbol, candle_timestamp, 1 if dry_run else 0,
            strategy_version, _utc_now(), order_id,
        ))
        connection.commit()


def close_trade(trade_id: int, exit_price: float, status: str,
                exit_order_id: str | None = None) -> float:
    connection = _conn()
    try:
        # Serialize the close snapshot with funding-event inserts. Without an
        # immediate write lock, an event could be added after this read but
        # before the close update and be omitted permanently from net PnL.
        connection.execute("BEGIN IMMEDIATE")
        columns = _cols(connection, "trades")
        row = connection.execute(
            "SELECT * FROM trades WHERE id=? AND status='open'", (trade_id,)
        ).fetchone()
        if not row:
            connection.commit()
            return 0.0
        trade = dict(zip(columns, row))

        # Volume-weighted entry once pyramid adds exist; the original otherwise.
        entry = effective_entry_price(trade)
        quantity = float(trade.get("quantity") or 0)
        notional = float(trade["size_usdt"])
        direction = 1.0 if trade["side"] == "long" else -1.0
        raw_return = ((exit_price - entry) / entry) * direction

        if quantity > 0:
            gross_pnl = (exit_price - entry) * quantity * direction
            entry_notional = entry * quantity
            exit_notional = exit_price * quantity
        else:
            gross_pnl = notional * raw_return
            entry_notional = notional
            exit_notional = max(0.0, notional + gross_pnl * direction)

        fees = (entry_notional + exit_notional) * TAKER_FEE_RATE
        funding_value = trade.get("funding_usdt")
        funding_known = funding_value is not None
        funding = float(funding_value) if funding_known else 0.0
        pnl_usdt = gross_pnl - fees + funding
        pnl_pct = pnl_usdt / entry_notional * 100 if entry_notional > 0 else 0.0
        closed_at = _utc_now()
        closed_time = _parse_utc(closed_at)
        synced_time = _parse_utc(trade.get("funding_synced_at"))
        coverage_known = (
            closed_time is not None
            and synced_time is not None
            and synced_time >= closed_time
        )
        model_version = 3 if funding_known and coverage_known else 2

        connection.execute("""
            UPDATE trades
            SET status=?, exit_price=?, pnl_usdt=?, pnl_pct=?, fees_usdt=?,
                exit_order_id=?, closed_at=?, pnl_model_version=?
            WHERE id=? AND status='open'
        """, (
            status, exit_price, pnl_usdt, pnl_pct, fees,
            exit_order_id, closed_at, model_version, trade_id,
        ))
        connection.commit()
        return pnl_usdt
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_trade(trade_id: int) -> dict | None:
    with _conn() as connection:
        columns = _cols(connection, "trades")
        row = connection.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        return dict(zip(columns, row)) if row else None


def get_open_trades(dry_run=None) -> list:
    with _conn() as connection:
        columns = _cols(connection, "trades")
        if dry_run is None:
            rows = connection.execute("SELECT * FROM trades WHERE status='open'").fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM trades WHERE status='open' AND dry_run=?",
                (1 if dry_run else 0,),
            ).fetchall()
        return [dict(zip(columns, row)) for row in rows]


def get_legacy_open_live_trades() -> list:
    return [
        trade for trade in get_open_trades(dry_run=False)
        if not trade.get("quantity")
    ]


def get_all_trades(limit: int = 200, dry_run=None) -> list:
    with _conn() as connection:
        columns = _cols(connection, "trades")
        if dry_run is None:
            rows = connection.execute(
                "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM trades WHERE dry_run=? ORDER BY opened_at DESC LIMIT ?",
                (1 if dry_run else 0, limit),
            ).fetchall()
        return [dict(zip(columns, row)) for row in rows]


def funding_event_exists(source: str, external_id: str) -> bool:
    with _conn() as connection:
        row = connection.execute("""
            SELECT 1 FROM funding_events WHERE source=? AND external_id=?
        """, (source, str(external_id))).fetchone()
        return row is not None


def record_funding_event(*, source: str, external_id: str, trade_id: int,
                         symbol: str, settlement_at: str, amount_usdt: float,
                         mode: str, funding_rate: float | None = None,
                         mark_price: float | None = None,
                         raw_type: str | None = None) -> bool:
    """Insert one cashflow and update its trade exactly once in one transaction."""
    if mode not in ("dry_run", "live"):
        raise ValueError(f"invalid funding mode: {mode}")
    amount = float(amount_usdt)
    if not math.isfinite(amount):
        raise ValueError("funding amount must be finite")
    if not source or not str(external_id) or _parse_utc(settlement_at) is None:
        raise ValueError("source, external_id and UTC settlement_at are required")

    connection = _conn()
    try:
        connection.execute("BEGIN IMMEDIATE")
        columns = _cols(connection, "trades")
        row = connection.execute(
            "SELECT * FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"trade {trade_id} does not exist")
        trade = dict(zip(columns, row))
        if trade["symbol"] != symbol:
            raise ValueError(f"funding symbol {symbol} does not match trade {trade_id}")
        if _mode(bool(trade["dry_run"])) != mode:
            raise ValueError(f"funding mode {mode} does not match trade {trade_id}")

        inserted = connection.execute("""
            INSERT OR IGNORE INTO funding_events
                (source, external_id, trade_id, symbol, settlement_at,
                 funding_rate, mark_price, amount_usdt, mode, raw_type, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            source, str(external_id), trade_id, symbol, settlement_at,
            funding_rate, mark_price, amount, mode, raw_type, _utc_now(),
        ))
        if inserted.rowcount == 0:
            connection.commit()
            return False

        previous_funding = trade.get("funding_usdt")
        new_funding = (float(previous_funding) if previous_funding is not None else 0.0) + amount
        updates = ["funding_usdt=?"]
        values = [new_funding]
        model_version = int(trade.get("pnl_model_version") or 1)
        if trade.get("status") != "open" and model_version in (2, 3):
            new_pnl = float(trade.get("pnl_usdt") or 0.0) + amount
            quantity = float(trade.get("quantity") or 0)
            entry_notional = (
                float(trade["entry_price"]) * quantity
                if quantity > 0 else float(trade["size_usdt"])
            )
            new_pct = new_pnl / entry_notional * 100 if entry_notional > 0 else 0.0
            updates.extend(["pnl_usdt=?", "pnl_pct=?"])
            values.extend([new_pnl, new_pct])
        values.append(trade_id)
        connection.execute(
            f"UPDATE trades SET {', '.join(updates)} WHERE id=?", values
        )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def find_trade_for_funding(symbol: str, settlement_at: str,
                           dry_run: bool) -> dict:
    """Return a match only when exactly one trade contains the settlement time."""
    settlement = _parse_utc(settlement_at)
    if settlement is None:
        return {"trade": None, "match_count": 0, "trade_ids": []}
    with _conn() as connection:
        columns = _cols(connection, "trades")
        rows = connection.execute("""
            SELECT * FROM trades WHERE symbol=? AND dry_run=?
        """, (symbol, 1 if dry_run else 0)).fetchall()
    matches = []
    for row in rows:
        trade = dict(zip(columns, row))
        opened = _parse_utc(trade.get("opened_at"))
        closed = _parse_utc(trade.get("closed_at"))
        if opened and opened <= settlement and (closed is None or settlement <= closed):
            matches.append(trade)
    return {
        "trade": matches[0] if len(matches) == 1 else None,
        "match_count": len(matches),
        "trade_ids": [trade["id"] for trade in matches],
    }


def get_funding_sync_candidates(dry_run: bool, limit: int = 500,
                                close_grace_seconds: int = 0) -> list:
    grace_days = max(0, int(close_grace_seconds)) / 86400.0
    # LIVE bills carry the authoritative cash amount and do not require a
    # modeled quantity. DRY-RUN can also recover quantity from notional/entry.
    quantity_clause = (
        "((quantity IS NOT NULL AND quantity>0) OR "
        "(size_usdt IS NOT NULL AND size_usdt>0 AND entry_price>0))"
        if dry_run else "1=1"
    )
    with _conn() as connection:
        columns = _cols(connection, "trades")
        rows = connection.execute(f"""
            SELECT * FROM trades
            WHERE dry_run=? AND {quantity_clause}
              AND (
                    status='open'
                    OR (
                        funding_synced_at IS NOT NULL
                        AND closed_at IS NOT NULL
                        AND julianday(funding_synced_at) < julianday(closed_at) + ?
                    )
                  )
            ORDER BY CASE WHEN status='open' THEN 0 ELSE 1 END,
                     COALESCE(funding_synced_at, opened_at) ASC, id ASC
            LIMIT ?
        """, (
            1 if dry_run else 0,
            grace_days,
            max(1, int(limit)),
        )).fetchall()
        return [dict(zip(columns, row)) for row in rows]


def mark_funding_synced(trade_ids: list[int], synced_at: str,
                        funding_known: bool = True) -> None:
    if not trade_ids:
        return
    if _parse_utc(synced_at) is None:
        raise ValueError("synced_at must be a UTC timestamp")
    unique_ids = sorted({int(trade_id) for trade_id in trade_ids})
    placeholders = ",".join("?" for _ in unique_ids)
    with _conn() as connection:
        if funding_known:
            connection.execute(f"""
                UPDATE trades
                SET funding_synced_at=?,
                    funding_usdt=COALESCE(funding_usdt, 0),
                    pnl_model_version=CASE
                        WHEN pnl_model_version=2
                         AND (status='open' OR closed_at IS NULL
                              OR julianday(?) >= julianday(closed_at))
                        THEN 3
                        ELSE pnl_model_version
                    END
                WHERE id IN ({placeholders})
                  AND (funding_synced_at IS NOT NULL OR status='open')
            """, [synced_at, synced_at, *unique_ids])
        else:
            connection.execute(f"""
                UPDATE trades SET funding_synced_at=?
                WHERE id IN ({placeholders})
                  AND (funding_synced_at IS NOT NULL OR status='open')
            """, [synced_at, *unique_ids])
        connection.commit()


def get_funding_events(trade_id: int | None = None, dry_run=None,
                       limit: int = 500) -> list:
    conditions = []
    params = []
    if trade_id is not None:
        conditions.append("f.trade_id=?")
        params.append(trade_id)
    if dry_run is not None:
        conditions.append("t.dry_run=?")
        params.append(1 if dry_run else 0)
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(max(1, int(limit)))
    with _conn() as connection:
        columns = _cols(connection, "funding_events")
        rows = connection.execute(f"""
            SELECT f.* FROM funding_events f
            JOIN trades t ON t.id=f.trade_id
            {where}
            ORDER BY f.settlement_at DESC, f.id DESC LIMIT ?
        """, params).fetchall()
        return [dict(zip(columns, row)) for row in rows]


def get_funding_summary(dry_run=None, open_only: bool | None = None,
                        stale_after_seconds: int | None = None) -> dict:
    event_conditions = []
    trade_conditions = []
    event_params = []
    trade_params = []
    if dry_run is not None:
        mode_value = 1 if dry_run else 0
        event_conditions.append("t.dry_run=?")
        trade_conditions.append("dry_run=?")
        event_params.append(mode_value)
        trade_params.append(mode_value)
    if open_only is True:
        event_conditions.append("t.status='open'")
        trade_conditions.append("status='open'")
    elif open_only is False:
        event_conditions.append("t.status!='open'")
        trade_conditions.append("status!='open'")
    event_where = f" WHERE {' AND '.join(event_conditions)}" if event_conditions else ""
    trade_where = f" WHERE {' AND '.join(trade_conditions)}" if trade_conditions else ""
    stale_seconds = (
        max(120, 2 * FUNDING_SYNC_INTERVAL_SECONDS)
        if stale_after_seconds is None
        else max(0, int(stale_after_seconds))
    )
    cursor_cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
    ).isoformat()
    unknown_clause = (
        "funding_usdt IS NULL OR COALESCE(pnl_model_version, 1) < 3 "
        "OR funding_synced_at IS NULL "
        "OR (status!='open' AND (closed_at IS NULL "
        "OR julianday(funding_synced_at) < julianday(closed_at))) "
        "OR (status='open' AND julianday(funding_synced_at) < julianday(?))"
    )
    with _conn() as connection:
        amount = connection.execute(f"""
            SELECT SUM(f.amount_usdt) FROM funding_events f
            JOIN trades t ON t.id=f.trade_id{event_where}
        """, event_params).fetchone()[0] or 0.0
        unknown = connection.execute(f"""
            SELECT COUNT(*) FROM trades{trade_where}
            {'AND' if trade_conditions else 'WHERE'} ({unknown_clause})
        """, [*trade_params, cursor_cutoff]).fetchone()[0]
    return {
        "amount": float(amount),
        "unknown_trades": int(unknown),
        "complete": int(unknown) == 0,
    }


def get_incomplete_recent_funding_trade_ids(
        dry_run: bool, day: str | None = None,
        stale_after_seconds: int | None = None) -> list[int]:
    """Return open or today-closed trades whose funding horizon is unproven."""
    day = day or datetime.now(timezone.utc).date().isoformat()
    stale_seconds = (
        max(120, 2 * FUNDING_SYNC_INTERVAL_SECONDS)
        if stale_after_seconds is None
        else max(0, int(stale_after_seconds))
    )
    cursor_cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
    ).isoformat()
    with _conn() as connection:
        rows = connection.execute("""
            SELECT id FROM trades
            WHERE dry_run=?
              AND (
                    (
                        status='open'
                        AND (
                            funding_usdt IS NULL
                            OR COALESCE(pnl_model_version, 1) < 3
                            OR funding_synced_at IS NULL
                            OR julianday(funding_synced_at) < julianday(?)
                        )
                    )
                    OR (
                        status!='open' AND substr(closed_at, 1, 10)=?
                        AND (
                            funding_usdt IS NULL
                            OR COALESCE(pnl_model_version, 1) < 3
                            OR funding_synced_at IS NULL
                            OR julianday(funding_synced_at) < julianday(closed_at)
                        )
                    )
                  )
            ORDER BY id
        """, (1 if dry_run else 0, cursor_cutoff, day)).fetchall()
    return [int(row[0]) for row in rows]


def get_total_funding(dry_run=None, open_only: bool | None = None) -> float:
    return get_funding_summary(dry_run=dry_run, open_only=open_only)["amount"]


def get_total_pnl(dry_run=None) -> float:
    trade_conditions = ["status!='open'", "pnl_usdt IS NOT NULL"]
    trade_params = []
    event_conditions = []
    event_params = []
    if dry_run is not None:
        mode_value = 1 if dry_run else 0
        trade_conditions.append("dry_run=?")
        trade_params.append(mode_value)
        event_conditions.append("t.dry_run=?")
        event_params.append(mode_value)
    trade_where = " AND ".join(trade_conditions)
    event_where = f" WHERE {' AND '.join(event_conditions)}" if event_conditions else ""
    with _conn() as connection:
        closed = connection.execute(
            f"SELECT SUM({_closed_price_fee_expression()}) FROM trades WHERE {trade_where}",
            trade_params,
        ).fetchone()[0] or 0.0
        funding = connection.execute(f"""
            SELECT SUM(f.amount_usdt) FROM funding_events f
            JOIN trades t ON t.id=f.trade_id{event_where}
        """, event_params).fetchone()[0] or 0.0
        return float(closed) + float(funding)


def save_signal(symbol, ta_score, news_score, final_score, action, indicators,
                dry_run=DRY_RUN, strategy_version=STRATEGY_VERSION,
                candle_timestamp=None):
    candle_timestamp = candle_timestamp or indicators.get("signal_timestamp")
    with _conn() as connection:
        connection.execute("""
            INSERT INTO signals
                (symbol, timestamp, candle_timestamp, ta_score, news_score,
                 final_score, action, indicators, dry_run, strategy_version)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            symbol, _utc_now(), candle_timestamp, ta_score, news_score,
            final_score, action, json.dumps(indicators),
            1 if dry_run else 0, strategy_version,
        ))
        connection.commit()


def get_latest_signals(limit: int = 30, dry_run=None,
                       strategy_version: str | None = None) -> list:
    conditions = []
    params = []
    if dry_run is not None:
        conditions.append("dry_run=?")
        params.append(1 if dry_run else 0)
    if strategy_version is not None:
        conditions.append("strategy_version=?")
        params.append(strategy_version)
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT s.* FROM signals s
        INNER JOIN (
            SELECT MAX(id) AS id FROM signals{where} GROUP BY symbol
        ) latest ON s.id=latest.id
        ORDER BY ABS(s.final_score) DESC
        LIMIT ?
    """
    params.append(limit)
    with _conn() as connection:
        columns = _cols(connection, "signals")
        rows = connection.execute(query, params).fetchall()
        return [dict(zip(columns, row)) for row in rows]


def get_daily_funding(dry_run=None, day: str | None = None) -> float:
    day = day or datetime.now(timezone.utc).date().isoformat()
    conditions = ["substr(f.settlement_at, 1, 10)=?"]
    params = [day]
    if dry_run is not None:
        conditions.append("t.dry_run=?")
        params.append(1 if dry_run else 0)
    with _conn() as connection:
        row = connection.execute(f"""
            SELECT SUM(f.amount_usdt) FROM funding_events f
            JOIN trades t ON t.id=f.trade_id
            WHERE {' AND '.join(conditions)}
        """, params).fetchone()
        return float(row[0] or 0.0)


def get_daily_pnl(dry_run=None, day: str | None = None) -> float:
    day = day or datetime.now(timezone.utc).date().isoformat()
    trade_conditions = [
        "status!='open'", "pnl_usdt IS NOT NULL", "substr(closed_at, 1, 10)=?"
    ]
    trade_params = [day]
    event_conditions = ["substr(f.settlement_at, 1, 10)=?"]
    event_params = [day]
    if dry_run is not None:
        mode_value = 1 if dry_run else 0
        trade_conditions.append("dry_run=?")
        trade_params.append(mode_value)
        event_conditions.append("t.dry_run=?")
        event_params.append(mode_value)
    with _conn() as connection:
        closed = connection.execute(
            f"SELECT SUM({_closed_price_fee_expression()}) FROM trades "
            f"WHERE {' AND '.join(trade_conditions)}",
            trade_params,
        ).fetchone()[0] or 0.0
        funding = connection.execute(f"""
            SELECT SUM(f.amount_usdt) FROM funding_events f
            JOIN trades t ON t.id=f.trade_id
            WHERE {' AND '.join(event_conditions)}
        """, event_params).fetchone()[0] or 0.0
        return float(closed) + float(funding)


def get_pnl_history(dry_run=None) -> list:
    trade_conditions = ["status!='open'", "closed_at IS NOT NULL", "pnl_usdt IS NOT NULL"]
    trade_params = []
    event_conditions = []
    event_params = []
    if dry_run is not None:
        mode_value = 1 if dry_run else 0
        trade_conditions.append("dry_run=?")
        trade_params.append(mode_value)
        event_conditions.append("t.dry_run=?")
        event_params.append(mode_value)

    expression = _closed_price_fee_expression()
    with _conn() as connection:
        closes = connection.execute(f"""
            SELECT id, symbol, closed_at, {expression} AS amount,
                   pnl_model_version
            FROM trades
            WHERE {' AND '.join(trade_conditions)}
        """, trade_params).fetchall()
        event_where = f" WHERE {' AND '.join(event_conditions)}" if event_conditions else ""
        funding = connection.execute(f"""
            SELECT f.id, f.trade_id, f.symbol, f.settlement_at, f.amount_usdt,
                   f.source, f.external_id
            FROM funding_events f
            JOIN trades t ON t.id=f.trade_id{event_where}
        """, event_params).fetchall()

    history = [
        {
            "timestamp": row[2],
            "closed_at": row[2],
            "pnl_usdt": float(row[3] or 0.0),
            "event_type": "close",
            "trade_id": row[0],
            "symbol": row[1],
            "pnl_model_version": row[4],
        }
        for row in closes
    ]
    history.extend({
        "timestamp": row[3],
        "closed_at": row[3],
        "pnl_usdt": float(row[4]),
        "event_type": "funding",
        "event_id": row[0],
        "trade_id": row[1],
        "symbol": row[2],
        "source": row[5],
        "external_id": row[6],
    } for row in funding)
    minimum_time = datetime.min.replace(tzinfo=timezone.utc)
    history.sort(key=lambda item: (
        _parse_utc(item["timestamp"]) or minimum_time,
        0 if item["event_type"] == "funding" else 1,
        item.get("event_id", item.get("trade_id", 0)),
    ))
    return history
