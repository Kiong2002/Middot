#!/usr/bin/env python3
"""Migrate LangGraph checkpoints from SqliteSaver to PostgresSaver.

The two savers intentionally use different physical schemas.  Copying their
tables directly would therefore be incorrect; this script reads and writes
through LangGraph's public saver APIs so serialized channels and pending
writes are converted by the installed LangGraph version.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument(
        "--database-url",
        default=os.getenv("MIDDOT_CHECKPOINT_DATABASE_URL", ""),
        help="PostgreSQL URL; defaults to MIDDOT_CHECKPOINT_DATABASE_URL",
    )
    return parser.parse_args()


def _checkpoint_key(item: Any) -> tuple[str, str, str]:
    configurable = item.config["configurable"]
    return (
        str(configurable["thread_id"]),
        str(configurable.get("checkpoint_ns") or ""),
        str(configurable["checkpoint_id"]),
    )


def main() -> int:
    args = parse_args()
    if not args.sqlite.is_file():
        raise SystemExit(f"SQLite checkpoint database not found: {args.sqlite}")
    if not args.database_url.startswith(("postgresql://", "postgres://")):
        raise SystemExit("a PostgreSQL --database-url is required")

    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.sqlite import SqliteSaver

    sqlite_conn = sqlite3.connect(str(args.sqlite), check_same_thread=False)
    source = SqliteSaver(sqlite_conn)
    try:
        # Materialize before writing so source and target can be validated as
        # complete sets.  Oldest first keeps parent chains intuitive, although
        # the target schema does not require the parent row to exist first.
        items = list(source.list(None))
        items.sort(key=_checkpoint_key)
        source_keys = {_checkpoint_key(item) for item in items}
        expected_writes = sum(len(item.pending_writes or ()) for item in items)

        with PostgresSaver.from_conn_string(args.database_url) as target:
            target.setup()
            with psycopg.connect(args.database_url) as validation:
                existing = int(validation.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0])
            if existing:
                raise RuntimeError(
                    f"target checkpoint store is not empty ({existing} checkpoints); "
                    "use a fresh PostgreSQL database"
                )

            for item in items:
                checkpoint = item.checkpoint
                parent_config = item.parent_config or {
                    "configurable": {
                        "thread_id": item.config["configurable"]["thread_id"],
                        "checkpoint_ns": item.config["configurable"].get("checkpoint_ns", ""),
                    }
                }
                target.put(
                    parent_config,
                    checkpoint,
                    item.metadata,
                    checkpoint.get("channel_versions", {}),
                )

                writes_by_task: dict[str, list[tuple[str, Any]]] = defaultdict(list)
                for task_id, channel, value in item.pending_writes or ():
                    writes_by_task[str(task_id)].append((str(channel), value))
                for task_id, writes in writes_by_task.items():
                    target.put_writes(item.config, writes, task_id)

            migrated = list(target.list(None))
            migrated_keys = {_checkpoint_key(item) for item in migrated}
            actual_writes = sum(len(item.pending_writes or ()) for item in migrated)
            if migrated_keys != source_keys:
                missing = sorted(source_keys - migrated_keys)[:5]
                extra = sorted(migrated_keys - source_keys)[:5]
                raise RuntimeError(f"checkpoint validation failed: missing={missing}, extra={extra}")
            if actual_writes != expected_writes:
                raise RuntimeError(
                    f"pending-write validation failed: source={expected_writes}, target={actual_writes}"
                )

        print(
            f"checkpoint migration complete: {len(items)} checkpoints, "
            f"{expected_writes} pending writes"
        )
        return 0
    finally:
        sqlite_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
