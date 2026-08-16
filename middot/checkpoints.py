"""Production LangGraph checkpointer selection."""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any


_lock = threading.Lock()
_postgres_saver: Any | None = None
_postgres_pool: Any | None = None
_sqlite_connections: list[sqlite3.Connection] = []


def get_checkpointer(sqlite_path: str) -> Any:
    global _postgres_saver, _postgres_pool
    database_url = str(
        os.getenv("MIDDOT_CHECKPOINT_DATABASE_URL")
        or os.getenv("MIDDOT_DATABASE_URL")
        or ""
    ).strip()
    if database_url.startswith(("postgresql://", "postgres://")):
        if _postgres_saver is not None:
            return _postgres_saver
        with _lock:
            if _postgres_saver is None:
                from langgraph.checkpoint.postgres import PostgresSaver
                from psycopg_pool import ConnectionPool

                _postgres_pool = ConnectionPool(
                    conninfo=database_url,
                    min_size=1,
                    max_size=int(os.getenv("MIDDOT_CHECKPOINT_POOL_MAX", "10")),
                    kwargs={"autocommit": True, "prepare_threshold": 0},
                    open=True,
                )
                _postgres_saver = PostgresSaver(_postgres_pool)
                _postgres_saver.setup()
        return _postgres_saver

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(sqlite_path, timeout=5.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    _sqlite_connections.append(conn)
    return SqliteSaver(conn)
