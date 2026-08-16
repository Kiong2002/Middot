import sqlite3

from middot.db_compat import _postgres_sql


ENTITY_ALIAS_UPSERT = (
    "INSERT INTO memory_entity_aliases(device_id,entity_id,alias_norm,confidence,updated_at) "
    "VALUES(?,?,?,?,?) ON CONFLICT(device_id,entity_id,alias_norm) DO UPDATE SET "
    "confidence=MAX(memory_entity_aliases.confidence,excluded.confidence),"
    "updated_at=excluded.updated_at"
)


def test_entity_alias_upsert_is_unambiguous_in_postgres():
    translated = _postgres_sql(ENTITY_ALIAS_UPSERT)

    assert (
        "confidence=GREATEST(memory_entity_aliases.confidence,excluded.confidence)"
        in translated
    )
    assert "GREATEST(confidence,excluded.confidence)" not in translated


def test_entity_alias_upsert_keeps_highest_confidence_in_sqlite():
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE memory_entity_aliases("
            "device_id TEXT,entity_id TEXT,alias_norm TEXT,confidence REAL,updated_at INTEGER,"
            "UNIQUE(device_id,entity_id,alias_norm))"
        )
        connection.execute(ENTITY_ALIAS_UPSERT, ("device", "entity", "alias", 0.9, 1))
        connection.execute(ENTITY_ALIAS_UPSERT, ("device", "entity", "alias", 0.4, 2))
        row = connection.execute(
            "SELECT confidence,updated_at FROM memory_entity_aliases"
        ).fetchone()
        assert row == (0.9, 2)
    finally:
        connection.close()
