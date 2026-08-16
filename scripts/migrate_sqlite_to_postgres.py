#!/usr/bin/env python3
"""Copy Middot's durable SQLite data into an initialized PostgreSQL database.

The target must be empty.  The script preserves primary keys, validates every
table count, and resets PostgreSQL sequences before committing.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument(
        "--database-url",
        default=os.getenv("MIDDOT_DATABASE_URL", ""),
        help="PostgreSQL URL; defaults to MIDDOT_DATABASE_URL",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.sqlite.is_file():
        raise SystemExit(f"SQLite database not found: {args.sqlite}")
    if not args.database_url.startswith(("postgresql://", "postgres://")):
        raise SystemExit("a PostgreSQL --database-url is required")

    # Importing app_v2 runs the versioned, advisory-locked schema initializer.
    os.environ["MIDDOT_DATABASE_URL"] = args.database_url
    import app_v2  # noqa: F401

    import psycopg
    from psycopg import sql

    source = sqlite3.connect(str(args.sqlite))
    source.row_factory = sqlite3.Row
    target = psycopg.connect(args.database_url, autocommit=False)
    try:
        source_tables = [
            str(row[0])
            for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        table_set = set(source_tables)
        dependencies = {
            table: {
                str(row[2])
                for row in source.execute(f'PRAGMA foreign_key_list("{table}")')
                if str(row[2]) in table_set
            }
            for table in source_tables
        }
        for table in source_tables:
            dependencies[table].discard(table)
        # Entity aliases point to their entity while the entity optionally
        # points back to its canonical alias. Insert the entity first with a
        # NULL canonical alias, then restore that pointer after both tables.
        dependencies.get("memory_entities", set()).discard("memory_entity_aliases")
        ordered: list[str] = []
        remaining = set(source_tables)
        while remaining:
            ready = sorted(table for table in remaining if dependencies[table] <= set(ordered))
            if not ready:
                raise RuntimeError(f"cyclic table dependencies: {sorted(remaining)}")
            ordered.extend(ready)
            remaining.difference_update(ready)
        source_tables = ordered
        copied: list[tuple[str, int]] = []
        for table in source_tables:
            target_columns = [
                str(row[0])
                for row in target.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema=current_schema() AND table_name=%s ORDER BY ordinal_position",
                    (table,),
                )
            ]
            if not target_columns:
                print(f"skip {table}: no target table")
                continue
            source_columns = [
                str(row[1]) for row in source.execute(f'PRAGMA table_info("{table}")')
            ]
            columns = [name for name in source_columns if name in target_columns]
            source_count = int(source.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            target_count = int(
                target.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
            )
            if target_count:
                raise RuntimeError(f"target table {table} is not empty ({target_count} rows)")
            if source_count:
                rows = [
                    tuple(
                        None
                        if table == "memory_entities" and name in {"canonical_alias_id", "merged_into"}
                        else row[name]
                        for name in columns
                    )
                    for row in source.execute(
                        f'SELECT {",".join(chr(34) + name + chr(34) for name in columns)} FROM "{table}"'
                    )
                ]
                statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(table),
                    sql.SQL(",").join(map(sql.Identifier, columns)),
                    sql.SQL(",").join(sql.Placeholder() for _ in columns),
                )
                with target.cursor() as cursor:
                    cursor.executemany(statement, rows)
            copied.append((table, source_count))

        if "memory_entities" in source_tables:
            entity_columns = {
                str(row[1]) for row in source.execute('PRAGMA table_info("memory_entities")')
            }
            if "canonical_alias_id" in entity_columns:
                canonical_rows = source.execute(
                    "SELECT id,canonical_alias_id FROM memory_entities WHERE canonical_alias_id IS NOT NULL"
                ).fetchall()
                with target.cursor() as cursor:
                    cursor.executemany(
                        "UPDATE memory_entities SET canonical_alias_id=%s WHERE id=%s",
                        [(row[1], row[0]) for row in canonical_rows],
                    )
            if "merged_into" in entity_columns:
                merged_rows = source.execute(
                    "SELECT id,merged_into FROM memory_entities WHERE merged_into IS NOT NULL"
                ).fetchall()
                with target.cursor() as cursor:
                    cursor.executemany(
                        "UPDATE memory_entities SET merged_into=%s WHERE id=%s",
                        [(row[1], row[0]) for row in merged_rows],
                    )

        for table, _ in copied:
            has_id = bool(target.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema=current_schema() "
                "AND table_name=%s AND column_name='id'",
                (table,),
            ).fetchone())
            if not has_id:
                continue
            sequence = target.execute(
                "SELECT pg_get_serial_sequence(%s, 'id')", (table,)
            ).fetchone()[0]
            if sequence:
                maximum = target.execute(
                    sql.SQL("SELECT MAX(id) FROM {}").format(sql.Identifier(table))
                ).fetchone()[0]
                if maximum is not None:
                    target.execute("SELECT setval(%s, %s, TRUE)", (sequence, maximum))

        failures: list[str] = []
        for table, expected in copied:
            actual = int(
                target.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
            )
            if actual != expected:
                failures.append(f"{table}: source={expected}, target={actual}")
            print(f"{table}: {actual} rows")
        if failures:
            raise RuntimeError("count validation failed: " + "; ".join(failures))
        target.commit()
        print(f"migration complete: {sum(count for _, count in copied)} rows")
        return 0
    except Exception:
        target.rollback()
        raise
    finally:
        target.close()
        source.close()


if __name__ == "__main__":
    sys.exit(main())
