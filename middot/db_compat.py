"""SQLite/PostgreSQL compatibility boundary for the legacy SQL surface.

The application historically uses sqlite3's qmark placeholders and Row API in
hundreds of small queries.  This adapter keeps that API stable while moving the
storage engine to PostgreSQL.  It is intentionally narrow: application SQL is
still visible and future modules should use native PostgreSQL/SQLAlchemy APIs.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any


DATABASE_URL = str(os.getenv("MIDDOT_DATABASE_URL") or "").strip()
IS_POSTGRES = DATABASE_URL.startswith(("postgresql://", "postgres://"))


class CompatRow(dict[str, Any]):
    def __init__(self, columns: Sequence[str], values: Sequence[Any]):
        super().__init__(zip(columns, values))
        self._values = tuple(values)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)


class _EmptyCursor:
    rowcount = 0
    lastrowid = None

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def __iter__(self):
        return iter(())


def _qmark_to_format(sql: str) -> str:
    out: list[str] = []
    quote = ""
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote:
            out.append(char)
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    out.append(sql[index + 1])
                    index += 1
                else:
                    quote = ""
        elif char in ("'", '"'):
            quote = char
            out.append(char)
        elif char == "?":
            out.append("%s")
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _postgres_sql(sql: str) -> str:
    value = sql.strip()
    value = re.sub(r"^BEGIN\s+IMMEDIATE\s*$", "BEGIN", value, flags=re.I)
    value = re.sub(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", "INSERT INTO", value, flags=re.I)
    was_insert_or_ignore = bool(re.search(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", sql, flags=re.I))
    value = re.sub(
        r"\bALTER\s+TABLE\s+([^\s]+)\s+ADD\s+COLUMN\s+(?!IF\s+NOT\s+EXISTS)",
        r"ALTER TABLE \1 ADD COLUMN IF NOT EXISTS ",
        value,
        flags=re.I,
    )
    value = re.sub(r"\bINTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT\b", "BIGSERIAL PRIMARY KEY", value, flags=re.I)
    value = re.sub(r"\bAUTOINCREMENT\b", "", value, flags=re.I)
    # SQLite INTEGER is a signed 64-bit storage class; PostgreSQL INTEGER is
    # only 32-bit, so BIGINT is the faithful target (trace timestamps are ms).
    value = re.sub(r"\bINTEGER\b", "BIGINT", value, flags=re.I)
    value = re.sub(r"\bREAL\b", "DOUBLE PRECISION", value, flags=re.I)
    value = re.sub(r"\bBLOB\b", "BYTEA", value, flags=re.I)
    value = re.sub(r"GROUP_CONCAT\(([^,]+),\s*('(?:[^']|'')*')\)", r"STRING_AGG(\1,\2)", value, flags=re.I)
    value = re.sub(r"instr\(([^,]+),\s*(%s)\)", r"POSITION(\2 IN \1)", value, flags=re.I)

    # SQLite's two-argument MAX/MIN are scalar functions; PostgreSQL names
    # these GREATEST/LEAST.  Aggregate MAX/MIN calls remain untouched.
    scalar_replacements = (
        ("MAX(memory_jobs.target_seq,excluded.target_seq)", "GREATEST(memory_jobs.target_seq,excluded.target_seq)"),
        ("MAX(memory_jobs.priority,excluded.priority)", "GREATEST(memory_jobs.priority,excluded.priority)"),
        ("MAX(priority,90)", "GREATEST(priority,90)"),
        ("MAX(confidence,excluded.confidence)", "GREATEST(confidence,excluded.confidence)"),
        ("MAX(1,COALESCE(", "GREATEST(1,COALESCE("),
        ("MAX(0,COALESCE(", "GREATEST(0,COALESCE("),
        ("MIN(0.99,", "LEAST(0.99,"),
    )
    for old, new in scalar_replacements:
        value = value.replace(old, new)

    if was_insert_or_ignore:
        value = value.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"

    # Preserve sqlite3.Cursor.lastrowid at the three production call sites.
    if re.match(r"INSERT\s+INTO\s+(favorites|memory_feedback)\b", value, flags=re.I) and " RETURNING " not in value.upper():
        value = value.rstrip().rstrip(";") + " RETURNING id"
    return _qmark_to_format(value)


def _postgres_script_statement(sql: str) -> str:
    return _postgres_sql(sql)


class PostgresCursor:
    def __init__(self, raw: Any):
        self.raw = raw
        self._lastrowid: int | None = None

    @property
    def rowcount(self) -> int:
        return int(self.raw.rowcount)

    @property
    def lastrowid(self) -> int | None:
        if self._lastrowid is None and self.raw.description:
            row = self.raw.fetchone()
            if row is not None:
                self._lastrowid = int(row[0])
        return self._lastrowid

    def _convert(self, row: Sequence[Any] | None) -> CompatRow | None:
        if row is None:
            return None
        columns = [str(item.name) for item in (self.raw.description or ())]
        return CompatRow(columns, row)

    def fetchone(self) -> CompatRow | None:
        return self._convert(self.raw.fetchone())

    def fetchall(self) -> list[CompatRow]:
        return [self._convert(row) for row in self.raw.fetchall()]  # type: ignore[list-item]

    def __iter__(self) -> Iterator[CompatRow]:
        for row in self.raw:
            converted = self._convert(row)
            if converted is not None:
                yield converted


class PostgresConnection:
    def __init__(self, pool: Any, raw: Any):
        self._pool = pool
        self.raw = raw
        self._closed = False
        self._schema_locked = False

    @property
    def in_transaction(self) -> bool:
        from psycopg.pq import TransactionStatus

        return self.raw.info.transaction_status != TransactionStatus.IDLE

    def acquire_schema_lock(self) -> None:
        if not self._schema_locked:
            self.raw.execute("SELECT pg_advisory_lock(724633681)")
            self._schema_locked = True

    def release_schema_lock(self) -> None:
        if self._schema_locked:
            if self.in_transaction:
                self.raw.commit()
            self.raw.execute("SELECT pg_advisory_unlock(724633681)")
            self.raw.commit()
            self._schema_locked = False

    def execute(self, sql: str, params: Sequence[Any] = ()) -> PostgresCursor | _EmptyCursor:
        stripped = sql.strip()
        pragma = re.fullmatch(r"PRAGMA\s+table_info\(([^)]+)\)", stripped, flags=re.I)
        if pragma:
            raw = self.raw.execute(
                "SELECT column_name AS name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name=%s ORDER BY ordinal_position",
                (pragma.group(1).strip("'\""),),
            )
            return PostgresCursor(raw)
        if re.fullmatch(r"PRAGMA\s+user_version", stripped, flags=re.I):
            self.raw.execute(
                "CREATE TABLE IF NOT EXISTS middot_schema_meta "
                "(singleton BOOLEAN PRIMARY KEY DEFAULT TRUE, user_version BIGINT NOT NULL DEFAULT 0, CHECK(singleton))"
            )
            self.raw.execute(
                "INSERT INTO middot_schema_meta(singleton,user_version) VALUES(TRUE,0) ON CONFLICT(singleton) DO NOTHING"
            )
            return PostgresCursor(self.raw.execute("SELECT user_version FROM middot_schema_meta WHERE singleton=TRUE"))
        version = re.fullmatch(r"PRAGMA\s+user_version\s*=\s*(\d+)", stripped, flags=re.I)
        if version:
            self.raw.execute(
                "UPDATE middot_schema_meta SET user_version=%s WHERE singleton=TRUE",
                (int(version.group(1)),),
            )
            return _EmptyCursor()
        if re.match(r"PRAGMA\s+(journal_mode|busy_timeout|foreign_keys)", stripped, flags=re.I):
            return _EmptyCursor()
        if re.fullmatch(r"BEGIN\s+IMMEDIATE", stripped, flags=re.I):
            if not self.in_transaction:
                self.raw.execute("BEGIN")
            # Preserve the old BEGIN IMMEDIATE critical-section semantics while
            # ordinary PostgreSQL writes remain concurrent and row-scoped.
            self.raw.execute("SELECT pg_advisory_xact_lock(724633682)")
            return _EmptyCursor()
        return PostgresCursor(self.raw.execute(_postgres_sql(sql), tuple(params)))

    def executescript(self, script: str) -> None:
        # The schema contains no procedures or semicolons inside string values.
        for statement in script.split(";"):
            if statement.strip():
                self.raw.execute(_postgres_script_statement(statement))

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._schema_locked:
                self.release_schema_lock()
            elif self.in_transaction:
                self.raw.rollback()
        finally:
            self._pool.putconn(self.raw)
            self._closed = True

    def __enter__(self) -> "PostgresConnection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del traceback
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False


_pool: Any | None = None
_pool_lock = threading.Lock()


def _postgres_pool() -> Any:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            from psycopg_pool import ConnectionPool

            _pool = ConnectionPool(
                conninfo=DATABASE_URL,
                min_size=int(os.getenv("MIDDOT_DB_POOL_MIN", "1")),
                max_size=int(os.getenv("MIDDOT_DB_POOL_MAX", "12")),
                timeout=float(os.getenv("MIDDOT_DB_POOL_TIMEOUT", "10")),
                kwargs={"autocommit": False, "prepare_threshold": None},
                open=True,
            )
    return _pool


def connect(sqlite_path: str) -> sqlite3.Connection | PostgresConnection:
    if not IS_POSTGRES:
        conn = sqlite3.connect(sqlite_path, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
    pool = _postgres_pool()
    return PostgresConnection(pool, pool.getconn())


def database_backend() -> str:
    return "postgresql" if IS_POSTGRES else "sqlite"


def acquire_schema_lock(conn: Any) -> None:
    method = getattr(conn, "acquire_schema_lock", None)
    if method is not None:
        method()


def release_schema_lock(conn: Any) -> None:
    method = getattr(conn, "release_schema_lock", None)
    if method is not None:
        method()


def integrity_error_types() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [sqlite3.IntegrityError]
    if IS_POSTGRES:
        import psycopg

        errors.append(psycopg.IntegrityError)
    return tuple(errors)
