"""
数据库连接和 Schema 初始化
==========================
提供数据库连接管理和所有表的创建逻辑
"""

import sqlite3
from flask import g
from ..config import MIDDOT_DB_PATH


def db_connect() -> sqlite3.Connection:
    """创建新的数据库连接"""
    conn = sqlite3.connect(MIDDOT_DB_PATH, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_db() -> sqlite3.Connection:
    """获取当前请求的数据库连接"""
    if "middot_db" not in g:
        g.middot_db = db_connect()
    return g.middot_db


def _db_close(_exc):
    """关闭当前请求的数据库连接"""
    conn = g.pop("middot_db", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def init_middot_db():
    """初始化数据库 schema"""
    conn = db_connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
  device_id     TEXT PRIMARY KEY,
  nickname      TEXT,
  created_at    INTEGER NOT NULL,
  last_seen_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS legacy_device_claims (
  old_device_id TEXT PRIMARY KEY,
  new_device_id TEXT NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS favorites (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id     TEXT NOT NULL,
  kind          TEXT NOT NULL,       -- 'location' | 'poi'
  label         TEXT,
  name          TEXT NOT NULL,
  address       TEXT,
  lng           REAL NOT NULL,
  lat           REAL NOT NULL,
  extra_json    TEXT,
  created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fav_device_kind ON favorites(device_id, kind, created_at DESC);
CREATE TABLE IF NOT EXISTS rooms (
  code           TEXT PRIMARY KEY,
  host_device_id TEXT NOT NULL,
  keyword        TEXT,
  anchor_json    TEXT,
  revision       INTEGER NOT NULL DEFAULT 0,
  status         TEXT NOT NULL DEFAULT 'active',   -- 'active' | 'closed'
  created_at     INTEGER NOT NULL,
  last_active_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS room_members (
  room_code     TEXT NOT NULL,
  device_id     TEXT NOT NULL,
  nickname      TEXT,
  role          TEXT NOT NULL DEFAULT 'member',    -- 'host' | 'member'
  location_json TEXT,
  prefer        TEXT DEFAULT 'auto',
  joined_at     INTEGER NOT NULL,
  PRIMARY KEY (room_code, device_id)
);
CREATE INDEX IF NOT EXISTS idx_rm_device ON room_members(device_id);
CREATE TABLE IF NOT EXISTS run_history (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id             TEXT NOT NULL,
  ran_at                INTEGER NOT NULL,
  anchor_json           TEXT,
  participants_json     TEXT NOT NULL,
  keyword               TEXT,
  city                  TEXT,
  results_summary_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_hist_dev ON run_history(device_id, ran_at DESC);
CREATE TABLE IF NOT EXISTS agent_memories (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id     TEXT NOT NULL,
  category      TEXT NOT NULL,
  memory_key    TEXT NOT NULL,
  memory_value  TEXT NOT NULL,
  source        TEXT NOT NULL DEFAULT 'explicit',
  status        TEXT NOT NULL DEFAULT 'confirmed',
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL,
  UNIQUE(device_id, category, memory_key)
);
CREATE INDEX IF NOT EXISTS idx_memory_device ON agent_memories(device_id, category, updated_at DESC);
CREATE TABLE IF NOT EXISTS memory_people (
  id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT NOT NULL,
  name TEXT NOT NULL, relation TEXT, usual_place TEXT, city TEXT,
  expires_at INTEGER, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
  UNIQUE(device_id, name)
);
CREATE TABLE IF NOT EXISTS memory_episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT NOT NULL,
  happened_at INTEGER NOT NULL, keyword TEXT, people_json TEXT,
  chosen_poi_json TEXT, summary TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episode_device ON memory_episodes(device_id, happened_at DESC);
CREATE TABLE IF NOT EXISTS memory_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT NOT NULL,
  poi_id TEXT, poi_name TEXT NOT NULL, signal TEXT NOT NULL,
  reason TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
  UNIQUE(device_id, poi_name, signal)
);
CREATE INDEX IF NOT EXISTS idx_feedback_device ON memory_feedback(device_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS memory_sources (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id       TEXT NOT NULL,
  source_type     TEXT NOT NULL,
  source_ref      TEXT NOT NULL,
  source_excerpt  TEXT,
  metadata_json   TEXT,
  created_at      INTEGER NOT NULL,
  UNIQUE(device_id, source_type, source_ref)
);
CREATE INDEX IF NOT EXISTS idx_memory_source_device ON memory_sources(device_id, created_at DESC);
CREATE TABLE IF NOT EXISTS memory_fact_events (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id           TEXT NOT NULL,
  kind                TEXT NOT NULL,
  entity_key          TEXT NOT NULL,
  record_id           INTEGER,
  action              TEXT NOT NULL,
  value_json          TEXT,
  changed_fields_json TEXT,
  source_id           INTEGER,
  happened_at         INTEGER NOT NULL,
  expires_at          INTEGER,
  idempotency_key     TEXT NOT NULL UNIQUE,
  FOREIGN KEY(source_id) REFERENCES memory_sources(id)
);
CREATE INDEX IF NOT EXISTS idx_memory_fact_history ON memory_fact_events(device_id, kind, entity_key, id DESC);
CREATE INDEX IF NOT EXISTS idx_memory_fact_record ON memory_fact_events(device_id, kind, record_id, id DESC);
CREATE TABLE IF NOT EXISTS conversations (
  id                    TEXT PRIMARY KEY,
  device_id             TEXT NOT NULL,
  title                 TEXT NOT NULL DEFAULT '新的对话',
  status                TEXT NOT NULL DEFAULT 'active',
  created_at            INTEGER NOT NULL,
  updated_at            INTEGER NOT NULL,
  last_activity_at      INTEGER NOT NULL,
  last_seq              INTEGER NOT NULL DEFAULT 0,
  last_compiled_seq     INTEGER NOT NULL DEFAULT 0,
  deleted_requested_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_conversation_device ON conversations(device_id, status, updated_at DESC);
CREATE TABLE IF NOT EXISTS conversation_events (
  conversation_id TEXT NOT NULL,
  seq             INTEGER NOT NULL,
  role            TEXT NOT NULL,
  event_type      TEXT NOT NULL DEFAULT 'message',
  visible_content TEXT NOT NULL,
  created_at      INTEGER NOT NULL,
  PRIMARY KEY(conversation_id, seq),
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_conversation_events_time ON conversation_events(conversation_id, seq);
CREATE TABLE IF NOT EXISTS memory_jobs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  job_type        TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  target_seq      INTEGER NOT NULL,
  priority        INTEGER NOT NULL DEFAULT 50,
  status          TEXT NOT NULL DEFAULT 'pending',
  attempts        INTEGER NOT NULL DEFAULT 0,
  run_after       INTEGER NOT NULL,
  lease_until     INTEGER,
  worker_id       TEXT,
  last_error      TEXT,
  created_at      INTEGER NOT NULL,
  started_at      INTEGER,
  finished_at     INTEGER,
  idempotency_key TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_memory_jobs_ready ON memory_jobs(status, run_after, priority DESC, id);
CREATE TABLE IF NOT EXISTS memory_compile_runs (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id   TEXT NOT NULL,
  from_seq          INTEGER NOT NULL,
  target_seq        INTEGER NOT NULL,
  reason            TEXT NOT NULL,
  status            TEXT NOT NULL,
  extracted_count   INTEGER NOT NULL DEFAULT 0,
  created_at        INTEGER NOT NULL,
  finished_at       INTEGER,
  error             TEXT,
  UNIQUE(conversation_id, from_seq, target_seq)
);
CREATE TABLE IF NOT EXISTS memory_candidates (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id         TEXT NOT NULL,
  kind              TEXT NOT NULL,
  entity_key        TEXT NOT NULL,
  field_name        TEXT NOT NULL,
  candidate_value   TEXT NOT NULL,
  confidence        REAL NOT NULL DEFAULT 0.5,
  evidence_summary  TEXT,
  source_conversation_id TEXT,
  source_from_seq   INTEGER,
  source_to_seq     INTEGER,
  status            TEXT NOT NULL DEFAULT 'candidate',
  created_at        INTEGER NOT NULL,
  updated_at        INTEGER NOT NULL,
  UNIQUE(device_id, kind, entity_key, field_name, candidate_value)
);
CREATE INDEX IF NOT EXISTS idx_memory_candidate_device ON memory_candidates(device_id, status, updated_at DESC);
CREATE TABLE IF NOT EXISTS memory_candidate_evidence (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id      INTEGER NOT NULL,
  conversation_id   TEXT,
  from_seq          INTEGER,
  to_seq            INTEGER,
  confidence        REAL NOT NULL,
  persistence_score REAL NOT NULL,
  evidence_summary  TEXT,
  created_at        INTEGER NOT NULL,
  UNIQUE(candidate_id,conversation_id,from_seq,to_seq),
  FOREIGN KEY(candidate_id) REFERENCES memory_candidates(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_candidate_evidence ON memory_candidate_evidence(candidate_id,created_at DESC);
CREATE TABLE IF NOT EXISTS memory_wiki_facts (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id       TEXT NOT NULL,
  subject_type    TEXT NOT NULL,
  subject_key     TEXT NOT NULL,
  predicate       TEXT NOT NULL,
  value           TEXT NOT NULL,
  confidence      REAL NOT NULL DEFAULT 1.0,
  status          TEXT NOT NULL DEFAULT 'confirmed',
  valid_from      INTEGER NOT NULL,
  expires_at      INTEGER,
  created_at      INTEGER NOT NULL,
  updated_at      INTEGER NOT NULL,
  UNIQUE(device_id,subject_type,subject_key,predicate)
);
CREATE INDEX IF NOT EXISTS idx_memory_wiki_device ON memory_wiki_facts(device_id,status,updated_at DESC);
CREATE TABLE IF NOT EXISTS memory_wiki_fact_versions (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id       TEXT NOT NULL,
  subject_type    TEXT NOT NULL,
  subject_key     TEXT NOT NULL,
  predicate       TEXT NOT NULL,
  value           TEXT NOT NULL,
  confidence      REAL NOT NULL,
  status          TEXT NOT NULL,
  valid_from      INTEGER,
  valid_to        INTEGER NOT NULL,
  change_reason   TEXT,
  created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_wiki_versions ON memory_wiki_fact_versions(device_id,subject_type,subject_key,predicate,valid_to DESC);
CREATE TABLE IF NOT EXISTS memory_wiki_fact_sources (
  fact_id         INTEGER NOT NULL,
  candidate_id    INTEGER NOT NULL,
  conversation_id TEXT,
  from_seq        INTEGER,
  to_seq          INTEGER,
  PRIMARY KEY(fact_id,candidate_id),
  FOREIGN KEY(fact_id) REFERENCES memory_wiki_facts(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS memory_entities (
  id                 TEXT PRIMARY KEY,
  device_id          TEXT NOT NULL,
  entity_type        TEXT NOT NULL,
  canonical_name     TEXT NOT NULL,
  canonical_norm     TEXT NOT NULL,
  external_key       TEXT,
  status             TEXT NOT NULL DEFAULT 'active',
  merged_into        TEXT,
  resolution_source  TEXT NOT NULL DEFAULT 'memory_compiler',
  created_at         INTEGER NOT NULL,
  updated_at         INTEGER NOT NULL,
  FOREIGN KEY(merged_into) REFERENCES memory_entities(id)
);
CREATE INDEX IF NOT EXISTS idx_memory_entity_lookup ON memory_entities(device_id,entity_type,canonical_norm,status);
CREATE TABLE IF NOT EXISTS memory_entity_aliases (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id      TEXT NOT NULL,
  entity_id      TEXT NOT NULL,
  alias          TEXT NOT NULL,
  alias_norm     TEXT NOT NULL,
  source         TEXT NOT NULL,
  confidence     REAL NOT NULL DEFAULT 1,
  status         TEXT NOT NULL DEFAULT 'confirmed',
  created_at     INTEGER NOT NULL,
  updated_at     INTEGER NOT NULL,
  UNIQUE(device_id,entity_id,alias_norm),
  FOREIGN KEY(entity_id) REFERENCES memory_entities(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_memory_alias_lookup ON memory_entity_aliases(device_id,alias_norm,status);
CREATE TABLE IF NOT EXISTS memory_entity_merge_events (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id          TEXT NOT NULL,
  source_entity_id   TEXT,
  target_entity_id   TEXT,
  action             TEXT NOT NULL,
  reason             TEXT NOT NULL,
  evidence_json      TEXT,
  reversible         INTEGER NOT NULL DEFAULT 1,
  created_at         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_entity_merge_device ON memory_entity_merge_events(device_id,created_at DESC);
"""
