import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "alert_state.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_dispositions (
                user      TEXT    NOT NULL,
                day       TEXT    NOT NULL,
                status    TEXT    NOT NULL,
                note      TEXT    NOT NULL DEFAULT '',
                timestamp TEXT    NOT NULL,
                PRIMARY KEY (user, day)
            )
        """)


def upsert_disposition(user: str, day: str, status: str, note: str = "") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute("""
            INSERT INTO alert_dispositions (user, day, status, note, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user, day) DO UPDATE SET
                status    = excluded.status,
                note      = excluded.note,
                timestamp = excluded.timestamp
        """, (user, day, status, note, ts))


def get_disposition(user: str, day: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM alert_dispositions WHERE user = ? AND day = ?",
            (user, day),
        ).fetchone()


def get_all_dispositions() -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM alert_dispositions ORDER BY timestamp DESC"
        ).fetchall()
