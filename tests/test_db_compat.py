import sqlite3

from middot.db_compat import _postgres_sql


ENTITY_ALIAS_UPSERT = (
    "INSERT INTO memory_entity_aliases(device_id,entity_id,alias_norm,confidence,updated_at) "
    "VALUES(?,?,?,?,?) ON CONFLICT(device_id,entity_id,alias_norm) DO UPDATE SET "
    "confidence=MAX(memory_entity_aliases.confidence,excluded.confidence),"
    "updated_at=excluded.updated_at"
)

PLACE_ALIAS_UPSERT = (
    "INSERT INTO place_alias_evidence(device_id,city,alias_norm,poi_id,confirmation_count) "
    "VALUES(?,?,?,?,1) ON CONFLICT(device_id,city,alias_norm,poi_id) DO UPDATE SET "
    "confirmation_count=place_alias_evidence.confirmation_count+1"
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


def test_place_alias_upsert_qualifies_current_row_for_both_databases():
    translated = _postgres_sql(PLACE_ALIAS_UPSERT)
    assert "place_alias_evidence.confirmation_count+1" in translated

    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE place_alias_evidence("
            "device_id TEXT,city TEXT,alias_norm TEXT,poi_id TEXT,confirmation_count INTEGER,"
            "UNIQUE(device_id,city,alias_norm,poi_id))"
        )
        values = ("device", "北京", "清华", "tsinghua")
        connection.execute(PLACE_ALIAS_UPSERT, values)
        connection.execute(PLACE_ALIAS_UPSERT, values)
        count = connection.execute(
            "SELECT confirmation_count FROM place_alias_evidence"
        ).fetchone()[0]
        assert count == 2
    finally:
        connection.close()
