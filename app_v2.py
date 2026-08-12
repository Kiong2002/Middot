"""
智能中间点推荐系统 v2 — 多 Agent 架构
======================================
Agent1（规划）:  LLM 理解需求 → 结构化搜索参数
Agent2（搜索）:  LLM + 受控工具 → 搜索候选地点（上下文精简）
路线计算:        纯 Python 直接调高德 API → A/B 分别计算
Agent3（总结）:  LLM 生成推荐文字

入口: python app_v2.py
"""

import os
import json
import uuid
import time
import re
import queue
import threading
import sqlite3
import secrets
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_from_directory, Response, g
from flask_cors import CORS
from openai import OpenAI

# 路线计算并发上限。每个 (POI, 参与者) 独立发一次 amap_get_best_route，
# 内部还会串行试多种交通方式。高德个人 key 默认 QPS 3/s，
# 付费 key 通常 ≥ 30/s。这里默认 6，可用 ROUTE_MAX_WORKERS 覆盖。
_ROUTE_MAX_WORKERS = int(os.getenv("ROUTE_MAX_WORKERS", "5"))
_ROUTE_LEG_RETRY   = int(os.getenv("ROUTE_LEG_RETRY", "1"))

from amap_client import (
    DEEPSEEK_API_KEY,
    AMAP_KEY,
    AMAP_JS_KEY,
    amap_geocode,
    amap_get_best_route,
    amap_search_nearby,
    haversine_distance,
    find_balanced_midpoint,
    fair_meeting_point,
    amap_district_polygon,
    amap_search_in_area,
)

app = Flask(__name__, static_folder="static")
CORS(app)

llm_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


# ──────────────────────────────────────────────────────
# Session 管理（内存缓存，TTL 1 小时）
# ──────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
SESSION_TTL = 3600


def session_create(data: dict) -> str:
    sid = str(uuid.uuid4())[:8]
    _sessions[sid] = {"data": data, "expires_at": time.time() + SESSION_TTL}
    _session_cleanup()
    return sid


def session_get(sid: str) -> dict | None:
    s = _sessions.get(sid)
    if not s or s["expires_at"] < time.time():
        return None
    return s["data"]


def session_update(sid: str, data: dict) -> bool:
    if sid not in _sessions:
        return False
    _sessions[sid]["data"].update(data)
    _sessions[sid]["expires_at"] = time.time() + SESSION_TTL
    return True


def _session_cleanup():
    now = time.time()
    expired = [k for k, v in _sessions.items() if v["expires_at"] < now]
    for k in expired:
        del _sessions[k]


# ══════════════════════════════════════════════════════
# 中点 Middot · 持久层（SQLite）
# ══════════════════════════════════════════════════════
# device_id 是匿名设备身份（cookie + localStorage 双备份），落到 devices 表；
# favorites 收藏的地点/POI；rooms/room_members 房间协作。所有跨会话数据的家。

MIDDOT_DB_PATH = os.getenv("MIDDOT_DB_PATH", os.path.join(os.path.dirname(__file__), "middot.db"))
DEVICE_COOKIE = "middot_did"
DEVICE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 年
ROOM_CODE_ALPHABET = "0123456789"      # 纯数字（PM 拍板：好读、好念、手机键盘友好）
ROOM_CODE_LEN = 6
ROOM_TTL_S = 60 * 60 * 24              # 未锁定：24h 无活跃回收
ROOM_LOCK_TTL_S = 60 * 60 * 24 * 7     # 锁定后暂存 7 天
ROOM_CODE_REUSE_COOLDOWN_S = 60 * 60 * 24  # 关闭后 24h 内不复用同 code
# 记忆锚点黑名单：太顺口 / 太常见 / 太像验证码
ROOM_CODE_BLACKLIST = frozenset({
    "000000", "111111", "222222", "333333", "444444",
    "555555", "666666", "777777", "888888", "999999",
    "123456", "234567", "345678", "456789", "567890",
    "654321", "121212", "112233", "998877", "520520",
})


def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(MIDDOT_DB_PATH, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _db() -> sqlite3.Connection:
    if "middot_db" not in g:
        g.middot_db = _db_connect()
    return g.middot_db


@app.teardown_appcontext
def _db_close(_exc):
    conn = g.pop("middot_db", None)
    if conn is not None:
        try: conn.close()
        except Exception: pass


def init_middot_db():
    conn = _db_connect()
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS devices (
          device_id     TEXT PRIMARY KEY,
          nickname      TEXT,
          created_at    INTEGER NOT NULL,
          last_seen_at  INTEGER NOT NULL
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
        """)
        # updated_by / last_ai_actions_json / locked_until 列：老库需要 in-place 迁移
        try:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(rooms)").fetchall()}
            if "updated_by" not in cols:
                conn.execute("ALTER TABLE rooms ADD COLUMN updated_by TEXT")
            if "last_ai_actions_json" not in cols:
                conn.execute("ALTER TABLE rooms ADD COLUMN last_ai_actions_json TEXT")
            if "locked_until" not in cols:
                conn.execute("ALTER TABLE rooms ADD COLUMN locked_until INTEGER")
        except Exception:
            pass
        conn.commit()
    finally:
        conn.close()


init_middot_db()


def _now() -> int:
    return int(time.time())


def _ensure_device(conn: sqlite3.Connection, device_id: str):
    now = _now()
    conn.execute(
        "INSERT INTO devices(device_id, created_at, last_seen_at) VALUES(?,?,?) "
        "ON CONFLICT(device_id) DO UPDATE SET last_seen_at=excluded.last_seen_at",
        (device_id, now, now),
    )
    conn.commit()


@app.before_request
def _middot_attach_device():
    # 只处理 API + SPA 路由；静态资源直接放行
    p = request.path
    if p.startswith("/static/") or p in ("/favicon.ico",):
        return None
    did = request.cookies.get(DEVICE_COOKIE)
    if not did or len(did) < 8:
        did = uuid.uuid4().hex
        g.middot_device_new = True
    else:
        g.middot_device_new = False
    g.device_id = did
    try:
        _ensure_device(_db(), did)
    except Exception as e:
        app.logger.warning("[middot] ensure_device failed: %s", e)


@app.after_request
def _middot_set_cookie(resp):
    did = getattr(g, "device_id", None)
    if did and getattr(g, "middot_device_new", False):
        resp.set_cookie(
            DEVICE_COOKIE, did,
            max_age=DEVICE_COOKIE_MAX_AGE,
            httponly=False,      # 前端 JS 也要读，做 localStorage 兜底
            samesite="Lax",
            secure=request.is_secure,
        )
    return resp


# ─────────────── /api/me ────────────────
@app.route("/api/me")
def api_me():
    conn = _db()
    fav_count = conn.execute(
        "SELECT COUNT(*) FROM favorites WHERE device_id=?", (g.device_id,),
    ).fetchone()[0]
    row = conn.execute(
        "SELECT nickname FROM devices WHERE device_id=?", (g.device_id,),
    ).fetchone()
    return jsonify({
        "device_id": g.device_id,
        "nickname":  row["nickname"] if row else None,
        "fav_count": fav_count,
    })


@app.route("/api/me/nickname", methods=["POST"])
def api_me_nickname():
    data = request.json or {}
    nick = (data.get("nickname") or "").strip()[:24]
    _db().execute("UPDATE devices SET nickname=? WHERE device_id=?", (nick or None, g.device_id))
    _db().commit()
    return jsonify({"ok": True, "nickname": nick or None})


# ─────────────── /api/favorites ────────────────
@app.route("/api/favorites")
def api_favorites_list():
    kind = request.args.get("kind")  # 可选：location / poi
    conn = _db()
    if kind:
        rows = conn.execute(
            "SELECT id,kind,label,name,address,lng,lat,extra_json,created_at "
            "FROM favorites WHERE device_id=? AND kind=? ORDER BY created_at DESC LIMIT 100",
            (g.device_id, kind),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id,kind,label,name,address,lng,lat,extra_json,created_at "
            "FROM favorites WHERE device_id=? ORDER BY created_at DESC LIMIT 200",
            (g.device_id,),
        ).fetchall()
    return jsonify({
        "favorites": [
            {
                "id": r["id"], "kind": r["kind"], "label": r["label"], "name": r["name"],
                "address": r["address"], "lng": r["lng"], "lat": r["lat"],
                "extra": json.loads(r["extra_json"]) if r["extra_json"] else None,
                "created_at": r["created_at"],
            } for r in rows
        ],
    })


@app.route("/api/favorites", methods=["POST"])
def api_favorites_add():
    data = request.json or {}
    kind = (data.get("kind") or "").strip()
    if kind not in ("location", "poi"):
        return jsonify({"error": "kind must be 'location' or 'poi'"}), 400
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    try:
        lng = float(data.get("lng"))
        lat = float(data.get("lat"))
    except (TypeError, ValueError):
        return jsonify({"error": "lng/lat must be numbers"}), 400
    label   = (data.get("label") or "").strip()[:32] or None
    address = (data.get("address") or "").strip()[:200] or None
    extra   = data.get("extra")
    extra_s = json.dumps(extra, ensure_ascii=False) if extra else None
    conn = _db()
    # 去重：同 device + kind + 相近坐标（<50m）不重复存
    dup = conn.execute(
        "SELECT id FROM favorites WHERE device_id=? AND kind=? "
        "AND ABS(lng-?)<0.0005 AND ABS(lat-?)<0.0005 LIMIT 1",
        (g.device_id, kind, lng, lat),
    ).fetchone()
    if dup:
        return jsonify({"ok": True, "id": dup["id"], "deduped": True})
    cur = conn.execute(
        "INSERT INTO favorites(device_id,kind,label,name,address,lng,lat,extra_json,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (g.device_id, kind, label, name, address, lng, lat, extra_s, _now()),
    )
    conn.commit()
    return jsonify({"ok": True, "id": cur.lastrowid})


@app.route("/api/favorites/<int:fid>", methods=["DELETE"])
def api_favorites_del(fid: int):
    conn = _db()
    cur = conn.execute("DELETE FROM favorites WHERE id=? AND device_id=?", (fid, g.device_id))
    conn.commit()
    if cur.rowcount == 0:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


# ─────────────── /api/v2/rooms ────────────────
# 房间协作：任何成员可改 anchor/keyword/自己 location；用 revision 做 last-write-wins。
# 6 位纯数字 code，避开记忆锚点黑名单，且关闭后 24h 内不复用。

def _gen_room_code(conn: sqlite3.Connection) -> str:
    """生成不冲突的 6 位数字房间 code。
    过滤：黑名单顺口码、当前 active、24h 内刚关闭的 code。
    """
    cooldown_threshold = _now() - ROOM_CODE_REUSE_COOLDOWN_S
    for _ in range(40):
        code = "".join(secrets.choice(ROOM_CODE_ALPHABET) for _ in range(ROOM_CODE_LEN))
        if code in ROOM_CODE_BLACKLIST:
            continue
        row = conn.execute(
            "SELECT status, last_active_at FROM rooms WHERE code=?", (code,)
        ).fetchone()
        if not row:
            return code
        if row["status"] == "active":
            continue
        # closed 房间：24h 冷却期内不复用（避免刚离开的人误进老友新房）
        if (row["last_active_at"] or 0) >= cooldown_threshold:
            continue
        # 冷却期过了：删掉旧 room 让新房占用同 code
        conn.execute("DELETE FROM room_members WHERE room_code=?", (code,))
        conn.execute("DELETE FROM rooms WHERE code=?", (code,))
        return code
    raise RuntimeError("room code allocation exhausted")


def _sweep_stale_rooms(conn: sqlite3.Connection):
    """懒清理：
    - 未锁定房间：24h 无活动 → closed
    - 锁定房间：locked_until 到期 → closed（哪怕最近还有活动，锁本身是显式 TTL）
    """
    now = _now()
    unlocked_threshold = now - ROOM_TTL_S
    conn.execute(
        "UPDATE rooms SET status='closed' WHERE status='active' AND "
        "(locked_until IS NULL OR locked_until=0) AND last_active_at<?",
        (unlocked_threshold,),
    )
    conn.execute(
        "UPDATE rooms SET status='closed' WHERE status='active' AND "
        "locked_until IS NOT NULL AND locked_until>0 AND locked_until<?",
        (now,),
    )


def _begin_immediate(conn: sqlite3.Connection):
    """
    在 Python sqlite3 默认 isolation_level='' 下，DML 会隐式开 deferred 事务。
    再次显式 BEGIN IMMEDIATE 会报 'cannot start a transaction within a transaction'。
    这里先把可能的隐式事务提交掉，再手动开 IMMEDIATE。
    """
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")


def _room_snapshot(conn: sqlite3.Connection, code: str, me_did: str) -> dict | None:
    """一次读 rooms + room_members，按 device_id 打 is_me 标。"""
    row = conn.execute(
        "SELECT code, host_device_id, keyword, anchor_json, revision, status, "
        "created_at, last_active_at, updated_by, last_ai_actions_json, locked_until "
        "FROM rooms WHERE code=?",
        (code,),
    ).fetchone()
    if not row or row["status"] != "active":
        return None
    members = conn.execute(
        "SELECT device_id, nickname, role, location_json, prefer, joined_at "
        "FROM room_members WHERE room_code=? ORDER BY joined_at ASC",
        (code,),
    ).fetchall()
    try:
        last_ai = json.loads(row["last_ai_actions_json"] or "[]")
    except (TypeError, ValueError):
        last_ai = []
    return {
        "code":            row["code"],
        "revision":        row["revision"],
        "host_device_id":  row["host_device_id"],
        "keyword":         row["keyword"],
        "anchor":          json.loads(row["anchor_json"]) if row["anchor_json"] else None,
        "updated_by":      row["updated_by"],
        "created_at":      row["created_at"],
        "last_active_at":  row["last_active_at"],
        "locked_until":    row["locked_until"] or 0,
        "last_ai_actions": last_ai,
        "members": [
            {
                "device_id": m["device_id"],
                "nickname":  m["nickname"],
                "role":      m["role"],
                "location":  json.loads(m["location_json"]) if m["location_json"] else None,
                "prefer":    m["prefer"] or "auto",
                "joined_at": m["joined_at"],
                "is_me":     (m["device_id"] == me_did),
            } for m in members
        ],
    }


def _room_bump_revision(conn: sqlite3.Connection, code: str, updated_by: str | None):
    """rooms.revision += 1（同事务原子），并更新 last_active_at + updated_by。"""
    conn.execute(
        "UPDATE rooms SET revision=revision+1, last_active_at=?, updated_by=? WHERE code=?",
        (_now(), updated_by, code),
    )


# ── AI 归属日志 + 写限流 ────────────────────────────────────
_AI_ACTION_LOG_MAX = 20
_AI_WRITE_LAST: dict[str, float] = {}       # device_id → monotonic()
_AI_WRITE_COOLDOWN_S = 10.0
_AI_WRITE_TOOLS = {"shift_center", "set_participant_location", "set_keyword", "set_radius", "add_participant"}


def _append_ai_action(
    conn: sqlite3.Connection,
    code: str,
    actor_did: str,
    action: dict,
    actor_type: str = "ai",
) -> None:
    """把一条动作 append 进 rooms.last_ai_actions_json（滚动最多 20 条）。
    actor_type='ai'（默认）由小 Mid 触发；actor_type='human' 是成员手动改锚点/关键词。
    前端 banner 用这个字段区分文案："X 的小 Mid 改了…" vs "X 改了…"。"""
    row = conn.execute(
        "SELECT last_ai_actions_json FROM rooms WHERE code=?", (code,)
    ).fetchone()
    try:
        log = json.loads((row["last_ai_actions_json"] if row else None) or "[]")
    except (TypeError, ValueError):
        log = []
    actor_name_row = conn.execute(
        "SELECT nickname FROM room_members WHERE room_code=? AND device_id=?",
        (code, actor_did),
    ).fetchone()
    actor_name = (actor_name_row["nickname"] if actor_name_row else None) or "某人"
    entry = {
        "id":         f"aia_{_now()}_{len(log)}",
        "ts":         _now(),
        "actor_did":  actor_did,
        "actor_name": actor_name,
        "actor_type": actor_type,
        "tool":       action.get("tool"),
        "before":     action.get("before"),
        "after":      action.get("after"),
        "undone":     False,
    }
    # 合并：同一 actor + 同 tool + 同 actor_type 且 10s 内没被撤销 → 替换掉旧条，只保留最新 after
    # 防止用户逐字打关键词时 banner 出现 "→ 火" "→ 火锅" "→ 火锅店" 一串刷屏
    if log:
        last = log[-1]
        if (last.get("actor_did") == actor_did
            and last.get("tool") == entry["tool"]
            and last.get("actor_type") == actor_type
            and not last.get("undone")
            and _now() - int(last.get("ts") or 0) <= 10):
            # 合并：沿用旧 id / 旧 before，只更新 ts 和 after
            entry["id"] = last["id"]
            entry["before"] = last.get("before")
            log[-1] = entry
        else:
            log.append(entry)
    else:
        log.append(entry)
    log = log[-_AI_ACTION_LOG_MAX:]
    conn.execute(
        "UPDATE rooms SET last_ai_actions_json=? WHERE code=?",
        (json.dumps(log, ensure_ascii=False), code),
    )


def _ai_write_gate(did: str, tool_name: str) -> tuple[bool, str]:
    """跨 stream 冷却：同一 device 10s 内只允许**一个** AI stream 写。
    同一 stream 里 N 个 tool_call 应该被视为一次原子操作——
    多人 one-shot『我在北大，Lisa在对外经贸，吃火锅』要一次调 3 个工具，
    别让 10s 限流把它拆成半成品。调用侧用 stream-local flag 保证一 stream 只 gate 一次。"""
    if not did or tool_name not in _AI_WRITE_TOOLS:
        return True, ""
    last = _AI_WRITE_LAST.get(did, 0.0)
    now = time.monotonic()
    if now - last < _AI_WRITE_COOLDOWN_S:
        remain = int(_AI_WRITE_COOLDOWN_S - (now - last)) + 1
        return False, f"请稍等 {remain}s，你或房间里刚有人改过（AI 写限流 10s/人）。"
    return True, ""


def _ai_write_touch(did: str, tool_name: str) -> None:
    if did and tool_name in _AI_WRITE_TOOLS:
        _AI_WRITE_LAST[did] = time.monotonic()


@app.route("/api/v2/rooms/<code>/undo_action", methods=["POST"])
def api_rooms_undo_action(code: str):
    """撤销一条 AI 归属动作：把 before 值写回房间字段。任何成员都能撤销。"""
    code = (code or "").strip().upper()
    data = request.json or {}
    action_id = (data.get("action_id") or "").strip()
    if not action_id:
        return jsonify({"error": "缺少 action_id"}), 400
    conn = _db()
    try:
        _begin_immediate(conn)
        row = conn.execute(
            "SELECT status, last_ai_actions_json, host_device_id FROM rooms WHERE code=?", (code,)
        ).fetchone()
        if not row or row["status"] != "active":
            conn.rollback()
            return jsonify({"error": "房间不存在或已关闭"}), 404
        member = conn.execute(
            "SELECT 1 FROM room_members WHERE room_code=? AND device_id=?",
            (code, g.device_id),
        ).fetchone()
        if not member:
            conn.rollback()
            return jsonify({"error": "你不是房间成员"}), 403
        try:
            log = json.loads(row["last_ai_actions_json"] or "[]")
        except (TypeError, ValueError):
            log = []
        target = next((a for a in log if a.get("id") == action_id), None)
        if not target:
            conn.rollback()
            return jsonify({"error": "找不到该动作，可能已过期"}), 404
        if target.get("undone"):
            conn.rollback()
            return jsonify({"ok": True, "already_undone": True})

        tool = target.get("tool")
        before = target.get("before") or {}

        # 权限：set_participant_location 只允许 actor / 地址所有者 / 房主 撤销
        # （不能让 Bob 一键回滚 Alice 的地址）
        if tool == "set_participant_location":
            actor_did = target.get("actor_did")
            owner_did = before.get("participant_did") or actor_did
            host_did = row["host_device_id"]
            if g.device_id not in (actor_did, owner_did, host_did):
                conn.rollback()
                return jsonify({"error": "只有触发者、地址所有者或房主可以撤销这条"}), 403

        # 应用 before 值
        if tool == "shift_center" and "anchor" in before:
            a = before["anchor"]
            conn.execute(
                "UPDATE rooms SET anchor_json=? WHERE code=?",
                (json.dumps(a, ensure_ascii=False) if a else None, code),
            )
        elif tool == "set_radius" and "anchor" in before:
            # 半径也存在 anchor.radius_m 里，走 anchor 整体覆盖
            a = before["anchor"]
            conn.execute(
                "UPDATE rooms SET anchor_json=? WHERE code=?",
                (json.dumps(a, ensure_ascii=False) if a else None, code),
            )
        elif tool == "set_keyword" and "keyword" in before:
            kw = (before.get("keyword") or "")[:120] or None
            conn.execute("UPDATE rooms SET keyword=? WHERE code=?", (kw, code))
        elif tool == "set_participant_location" and "location" in before:
            pid = before.get("participant_did") or target.get("actor_did")
            loc = before.get("location")
            loc_s = json.dumps(loc, ensure_ascii=False) if loc else None
            conn.execute(
                "UPDATE room_members SET location_json=? WHERE room_code=? AND device_id=?",
                (loc_s, code, pid),
            )

        # 标记 undone + append 一条 "由 X 撤销" 系统消息
        for a in log:
            if a.get("id") == action_id:
                a["undone"] = True
                a["undone_by"] = g.device_id
                a["undone_ts"] = _now()
                break
        conn.execute(
            "UPDATE rooms SET last_ai_actions_json=? WHERE code=?",
            (json.dumps(log, ensure_ascii=False), code),
        )
        _room_bump_revision(conn, code, g.device_id)
        conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.warning("[rooms] undo_action failed: %s", e)
        return jsonify({"error": "撤销失败"}), 500
    return jsonify({"ok": True, "snapshot": _room_snapshot(conn, code, g.device_id)})


@app.route("/api/v2/rooms", methods=["POST"])
def api_rooms_create():
    data = request.json or {}
    nickname = (data.get("nickname") or "").strip()[:24] or "房主"
    conn = _db()
    _sweep_stale_rooms(conn)
    try:
        _begin_immediate(conn)
        code = _gen_room_code(conn)
        now = _now()
        anchor = data.get("anchor")
        anchor_s = json.dumps(anchor, ensure_ascii=False) if anchor else None
        keyword = (data.get("keyword") or "").strip()[:120] or None
        conn.execute(
            "INSERT INTO rooms(code, host_device_id, keyword, anchor_json, revision, "
            "status, created_at, last_active_at, updated_by) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (code, g.device_id, keyword, anchor_s, 1, "active", now, now, g.device_id),
        )
        location = data.get("location")
        loc_s = json.dumps(location, ensure_ascii=False) if location else None
        prefer = (data.get("prefer") or "auto").strip()[:16] or "auto"
        conn.execute(
            "INSERT INTO room_members(room_code, device_id, nickname, role, "
            "location_json, prefer, joined_at) VALUES(?,?,?,?,?,?,?)",
            (code, g.device_id, nickname, "host", loc_s, prefer, now),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.warning("[rooms] create failed: %s", e)
        return jsonify({"error": "创建房间失败"}), 500
    return jsonify({"code": code, "snapshot": _room_snapshot(conn, code, g.device_id)})


@app.route("/api/v2/rooms/join", methods=["POST"])
def api_rooms_join():
    data = request.json or {}
    code = (data.get("code") or "").strip().upper()
    if not code:
        return jsonify({"error": "缺少 code"}), 400
    nickname = (data.get("nickname") or "").strip()[:24] or "访客"
    location = data.get("location")
    loc_s = json.dumps(location, ensure_ascii=False) if location else None
    prefer = (data.get("prefer") or "auto").strip()[:16] or "auto"
    conn = _db()
    try:
        _begin_immediate(conn)
        _sweep_stale_rooms(conn)
        room = conn.execute(
            "SELECT status, locked_until FROM rooms WHERE code=?", (code,)
        ).fetchone()
        if not room or room["status"] != "active":
            conn.rollback()
            return jsonify({"error": "房间不存在或已关闭"}), 404
        now = _now()
        # UPSERT 自己那行；已在 → 更新 nickname/location/prefer 但不改 joined_at/role
        existed = conn.execute(
            "SELECT 1 FROM room_members WHERE room_code=? AND device_id=?",
            (code, g.device_id),
        ).fetchone()
        locked_until = room["locked_until"] or 0
        if locked_until and locked_until > now and not existed:
            conn.rollback()
            return jsonify({"error": "房间已锁定，不再接受新成员"}), 403
        if existed:
            conn.execute(
                "UPDATE room_members SET nickname=?, location_json=COALESCE(?, location_json), "
                "prefer=COALESCE(?, prefer) WHERE room_code=? AND device_id=?",
                (nickname, loc_s, prefer, code, g.device_id),
            )
        else:
            conn.execute(
                "INSERT INTO room_members(room_code, device_id, nickname, role, "
                "location_json, prefer, joined_at) VALUES(?,?,?,?,?,?,?)",
                (code, g.device_id, nickname, "member", loc_s, prefer, now),
            )
        _room_bump_revision(conn, code, g.device_id)
        conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.warning("[rooms] join failed: %s", e)
        return jsonify({"error": "加入失败"}), 500
    return jsonify({"snapshot": _room_snapshot(conn, code, g.device_id)})


@app.route("/api/v2/rooms/<code>")
def api_rooms_get(code: str):
    code = (code or "").strip().upper()
    try:
        since_rev = int(request.args.get("since_rev", "0"))
    except ValueError:
        since_rev = 0
    conn = _db()
    _sweep_stale_rooms(conn)
    row = conn.execute(
        "SELECT revision, status FROM rooms WHERE code=?", (code,)
    ).fetchone()
    if not row or row["status"] != "active":
        return jsonify({"error": "房间不存在或已关闭"}), 404
    if row["revision"] <= since_rev:
        return jsonify({"revision": row["revision"], "unchanged": True})
    snap = _room_snapshot(conn, code, g.device_id)
    if snap is None:
        return jsonify({"error": "房间不存在"}), 404
    return jsonify({"snapshot": snap})


@app.route("/api/v2/rooms/<code>/update", methods=["POST"])
def api_rooms_update(code: str):
    code = (code or "").strip().upper()
    data = request.json or {}
    conn = _db()
    try:
        _begin_immediate(conn)
        room = conn.execute(
            "SELECT status FROM rooms WHERE code=?", (code,)
        ).fetchone()
        if not room or room["status"] != "active":
            conn.rollback()
            return jsonify({"error": "房间不存在或已关闭"}), 404
        member = conn.execute(
            "SELECT 1 FROM room_members WHERE room_code=? AND device_id=?",
            (code, g.device_id),
        ).fetchone()
        if not member:
            conn.rollback()
            return jsonify({"error": "你不是房间成员，请先加入"}), 403

        # 人工改动 attribution 用：先把当前 anchor/keyword 拍下来，等改完再对比
        prev_row = conn.execute(
            "SELECT anchor_json, keyword FROM rooms WHERE code=?", (code,)
        ).fetchone()
        try:
            prev_anchor = json.loads(prev_row["anchor_json"] or "null") if prev_row else None
        except (TypeError, ValueError):
            prev_anchor = None
        prev_keyword = prev_row["keyword"] if prev_row else None

        # 房间级字段：任何成员可改
        if "anchor" in data:
            a = data["anchor"]
            anchor_s = json.dumps(a, ensure_ascii=False) if a else None
            conn.execute("UPDATE rooms SET anchor_json=? WHERE code=?", (anchor_s, code))
        if "keyword" in data:
            kw = (data.get("keyword") or "").strip()[:120] or None
            conn.execute("UPDATE rooms SET keyword=? WHERE code=?", (kw, code))

        # 成员自己的字段：只能改自己那行
        member_dirty = False
        if "my_location" in data:
            loc = data["my_location"]
            loc_s = json.dumps(loc, ensure_ascii=False) if loc else None
            conn.execute(
                "UPDATE room_members SET location_json=? WHERE room_code=? AND device_id=?",
                (loc_s, code, g.device_id),
            )
            member_dirty = True
        if "my_prefer" in data:
            pref = (data.get("my_prefer") or "auto").strip()[:16] or "auto"
            conn.execute(
                "UPDATE room_members SET prefer=? WHERE room_code=? AND device_id=?",
                (pref, code, g.device_id),
            )
            member_dirty = True
        if "my_nickname" in data:
            nick = (data.get("my_nickname") or "").strip()[:24] or None
            conn.execute(
                "UPDATE room_members SET nickname=? WHERE room_code=? AND device_id=?",
                (nick, code, g.device_id),
            )
            member_dirty = True

        # AI 归属日志：单独一条 append，前端广播时用
        ai_action = data.get("ai_action")
        is_ai = isinstance(ai_action, dict) and ai_action.get("tool")
        if is_ai:
            _append_ai_action(conn, code, g.device_id, ai_action)

        # 人工改动 attribution：只在【非 AI 触发】且【anchor/keyword 真变了】时才 log。
        # 让其他成员知道"小明改了锚点到国贸"——跟 AI 攻击面走同一条广播链。
        if not is_ai:
            if "anchor" in data:
                new_a = data["anchor"]
                # 规范化对比：dumps sort_keys 消除字段顺序差异
                if json.dumps(new_a, sort_keys=True, ensure_ascii=False) != json.dumps(prev_anchor, sort_keys=True, ensure_ascii=False):
                    _append_ai_action(
                        conn, code, g.device_id,
                        {"tool": "shift_center",
                         "before": {"anchor": prev_anchor},
                         "after":  {"anchor": new_a}},
                        actor_type="human",
                    )
            if "keyword" in data:
                new_kw = (data.get("keyword") or "").strip()[:120] or None
                if new_kw != prev_keyword:
                    _append_ai_action(
                        conn, code, g.device_id,
                        {"tool": "set_keyword",
                         "before": {"keyword": prev_keyword},
                         "after":  {"keyword": new_kw}},
                        actor_type="human",
                    )

        _room_bump_revision(conn, code, g.device_id)
        conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.warning("[rooms] update failed: %s", e)
        return jsonify({"error": "更新失败"}), 500
    # 返回新 revision，前端可用来跳过下一次自己刚发出的 poll
    new_rev = conn.execute(
        "SELECT revision FROM rooms WHERE code=?", (code,)
    ).fetchone()["revision"]
    return jsonify({"ok": True, "revision": new_rev, "updated_by": g.device_id})


@app.route("/api/v2/rooms/<code>/leave", methods=["POST"])
def api_rooms_leave(code: str):
    code = (code or "").strip().upper()
    conn = _db()
    try:
        _begin_immediate(conn)
        room = conn.execute(
            "SELECT host_device_id, status FROM rooms WHERE code=?", (code,)
        ).fetchone()
        if not room:
            conn.rollback()
            return jsonify({"ok": True, "note": "房间不存在"}), 200
        conn.execute(
            "DELETE FROM room_members WHERE room_code=? AND device_id=?",
            (code, g.device_id),
        )
        # host 走了 → 移交给最早 joined 的成员；没人了 → close
        if room["host_device_id"] == g.device_id:
            heir = conn.execute(
                "SELECT device_id FROM room_members WHERE room_code=? "
                "ORDER BY joined_at ASC LIMIT 1",
                (code,),
            ).fetchone()
            if heir:
                conn.execute(
                    "UPDATE rooms SET host_device_id=? WHERE code=?",
                    (heir["device_id"], code),
                )
                conn.execute(
                    "UPDATE room_members SET role='host' WHERE room_code=? AND device_id=?",
                    (code, heir["device_id"]),
                )
            else:
                conn.execute(
                    "UPDATE rooms SET status='closed' WHERE code=?", (code,)
                )
        _room_bump_revision(conn, code, g.device_id)
        conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.warning("[rooms] leave failed: %s", e)
        return jsonify({"error": "退出失败"}), 500
    return jsonify({"ok": True})


@app.route("/api/v2/rooms/<code>/lock", methods=["POST"])
def api_rooms_lock(code: str):
    """房主锁定/解锁房间。锁定后 locked_until = now + 7d；解锁清 0。
    body: {"locked": true|false}
    """
    code = (code or "").strip().upper()
    data = request.json or {}
    want_lock = bool(data.get("locked"))
    conn = _db()
    try:
        _begin_immediate(conn)
        row = conn.execute(
            "SELECT host_device_id, status FROM rooms WHERE code=?", (code,)
        ).fetchone()
        if not row or row["status"] != "active":
            conn.rollback()
            return jsonify({"error": "房间不存在或已关闭"}), 404
        if row["host_device_id"] != g.device_id:
            conn.rollback()
            return jsonify({"error": "只有房主可以锁定/解锁房间"}), 403
        new_lock = (_now() + ROOM_LOCK_TTL_S) if want_lock else 0
        conn.execute(
            "UPDATE rooms SET locked_until=? WHERE code=?", (new_lock, code)
        )
        _room_bump_revision(conn, code, g.device_id)
        conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.warning("[rooms] lock failed: %s", e)
        return jsonify({"error": "操作失败"}), 500
    return jsonify({"ok": True, "snapshot": _room_snapshot(conn, code, g.device_id)})


@app.route("/api/v2/rooms/<code>/kick", methods=["POST"])
def api_rooms_kick(code: str):
    """房主把某人踢出房间。body: {"device_id": "..."}"""
    code = (code or "").strip().upper()
    data = request.json or {}
    target = (data.get("device_id") or "").strip()
    if not target:
        return jsonify({"error": "缺少 device_id"}), 400
    conn = _db()
    try:
        _begin_immediate(conn)
        row = conn.execute(
            "SELECT host_device_id, status FROM rooms WHERE code=?", (code,)
        ).fetchone()
        if not row or row["status"] != "active":
            conn.rollback()
            return jsonify({"error": "房间不存在或已关闭"}), 404
        if row["host_device_id"] != g.device_id:
            conn.rollback()
            return jsonify({"error": "只有房主可以踢人"}), 403
        if target == g.device_id:
            conn.rollback()
            return jsonify({"error": "不能踢自己，请用『退出房间』"}), 400
        cur = conn.execute(
            "DELETE FROM room_members WHERE room_code=? AND device_id=?",
            (code, target),
        )
        if cur.rowcount == 0:
            conn.rollback()
            return jsonify({"error": "此成员不在房间"}), 404
        # 踢人自动锁房 7 天，防止对方拿同一链接立刻回来（社交事故）
        lock_until = _now() + ROOM_LOCK_TTL_S
        conn.execute(
            "UPDATE rooms SET locked_until=? WHERE code=?",
            (lock_until, code),
        )
        _room_bump_revision(conn, code, g.device_id)
        conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.warning("[rooms] kick failed: %s", e)
        return jsonify({"error": "操作失败"}), 500
    return jsonify({"ok": True, "snapshot": _room_snapshot(conn, code, g.device_id)})


# ─────────────── /api/v2/history ────────────────
# 我的历史：每次 run_pipeline 成功后写一条精简摘要，restore 只回填参数不重跑。

def _compact_poi(p: dict) -> dict:
    """把一张 POI enriched 卡压缩成历史摘要用的最小形态。"""
    return {
        "name":    p.get("name"),
        "address": p.get("address"),
        "lng":     p.get("lng"),
        "lat":     p.get("lat"),
        "rating":  p.get("rating"),
        "avg_min": p.get("avg_min"),
        "worst_min": p.get("worst_min"),
    }


def _persist_run_history(anchor, participants, keyword, city, enriched):
    """在 run_pipeline 成功后调，把这次搜索存到 run_history。"""
    try:
        did = getattr(g, "device_id", None)
        if not did:
            return
        conn = _db()
        # 参与者也精简，避免存太多冗余
        parts_min = [
            {
                "name":    p.get("name"),
                "address": p.get("address"),
                "lng":     p.get("lng"),
                "lat":     p.get("lat"),
                "prefer":  p.get("prefer", "auto"),
            } for p in participants
        ]
        top5 = [_compact_poi(x) for x in (enriched or [])[:5]]
        conn.execute(
            "INSERT INTO run_history(device_id, ran_at, anchor_json, participants_json, "
            "keyword, city, results_summary_json) VALUES(?,?,?,?,?,?,?)",
            (
                did, _now(),
                json.dumps(anchor, ensure_ascii=False) if anchor else None,
                json.dumps(parts_min, ensure_ascii=False),
                keyword or None,
                city or None,
                json.dumps(top5, ensure_ascii=False),
            ),
        )
        # 限制每 device 最多 100 条，超过删最老的
        conn.execute(
            "DELETE FROM run_history WHERE device_id=? AND id NOT IN "
            "(SELECT id FROM run_history WHERE device_id=? ORDER BY ran_at DESC LIMIT 100)",
            (did, did),
        )
        people_names = [p.get("name") for p in participants if p.get("name") and p.get("name") != "我"]
        top = (enriched or [{}])[0] if enriched else {}
        summary = f"和{'、'.join(people_names) or '朋友'}找了{keyword or '会面地点'}"
        if top.get("name"): summary += f"，首选推荐是{top['name']}"
        conn.execute(
            "INSERT INTO memory_episodes(device_id,happened_at,keyword,people_json,chosen_poi_json,summary) VALUES(?,?,?,?,?,?)",
            (did,_now(),keyword or None,json.dumps(people_names,ensure_ascii=False),
             json.dumps(_compact_poi(top),ensure_ascii=False) if top else None,summary),
        )
        conn.execute(
            "DELETE FROM memory_episodes WHERE device_id=? AND id NOT IN "
            "(SELECT id FROM memory_episodes WHERE device_id=? ORDER BY happened_at DESC LIMIT 100)",
            (did,did),
        )
        conn.commit()
    except Exception as e:
        app.logger.warning("[history] persist failed: %s", e)


@app.route("/api/v2/history")
def api_history_list():
    try:
        limit = min(50, max(1, int(request.args.get("limit", "20"))))
    except ValueError:
        limit = 20
    conn = _db()
    rows = conn.execute(
        "SELECT id, ran_at, anchor_json, participants_json, keyword, city, "
        "results_summary_json FROM run_history WHERE device_id=? "
        "ORDER BY ran_at DESC LIMIT ?",
        (g.device_id, limit),
    ).fetchall()
    items = []
    for r in rows:
        try:
            parts = json.loads(r["participants_json"] or "[]")
        except Exception:
            parts = []
        try:
            top = json.loads(r["results_summary_json"] or "[]")
        except Exception:
            top = []
        try:
            anchor = json.loads(r["anchor_json"]) if r["anchor_json"] else None
        except Exception:
            anchor = None
        items.append({
            "id":                r["id"],
            "ran_at":            r["ran_at"],
            "keyword":           r["keyword"],
            "city":              r["city"],
            "anchor":            anchor,
            "participants_count": len(parts),
            "top_pois":          top[:3],
        })
    return jsonify({"items": items})


@app.route("/api/v2/history/<int:hid>/restore", methods=["POST"])
def api_history_restore(hid: int):
    conn = _db()
    row = conn.execute(
        "SELECT anchor_json, participants_json, keyword, city, results_summary_json "
        "FROM run_history WHERE id=? AND device_id=?",
        (hid, g.device_id),
    ).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    def _safe_json(s, default):
        if not s: return default
        try: return json.loads(s)
        except Exception: return default
    return jsonify({
        "anchor":       _safe_json(row["anchor_json"], None),
        "participants": _safe_json(row["participants_json"], []),
        "keyword":      row["keyword"],
        "city":         row["city"],
        "top_pois":     _safe_json(row["results_summary_json"], []),
    })


@app.route("/api/v2/history/<int:hid>", methods=["DELETE"])
def api_history_delete(hid: int):
    conn = _db()
    cur = conn.execute(
        "DELETE FROM run_history WHERE id=? AND device_id=?", (hid, g.device_id)
    )
    conn.commit()
    if cur.rowcount == 0:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


# ──────────────────────────────────────────────────────
# 通用工具函数
# ──────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | list | None:
    """从 LLM 回答中提取第一个合法 JSON 对象或数组"""
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _compact_poi(poi: dict) -> dict:
    """压缩 POI 字段，减少传给 LLM 的 token 数量"""
    return {
        "name":    poi.get("name", ""),
        "address": poi.get("address", ""),
        "lng":     poi.get("lng", 0),
        "lat":     poi.get("lat", 0),
        "rating":  poi.get("rating", 0),
    }


def _format_route(route: dict) -> dict:
    success = route.get("success", True)
    # 查询失败时保留 error 字段，duration_text 置为 None（前端据此显示友好提示）
    return {
        "mode":             route.get("mode", "unknown"),
        "success":          success,
        "error":            route.get("error") if not success else None,
        "duration_text":    route.get("duration_text") if success else None,
        "distance_text":    route.get("distance_text") if success else None,
        "line_summary":     route.get("line_summary", ""),
        "duration_minutes": route.get("duration_minutes", 999),
        "all_modes":        route.get("all_modes", {}),
    }


# ──────────────────────────────────────────────────────
# Agent 1：规划 Agent
# 职责：理解用户自然语言需求 → 结构化搜索参数
# 特点：无工具调用，上下文极短（~200 tokens in/out）
# ──────────────────────────────────────────────────────

_PLAN_SYSTEM = """\
你是搜索参数提取专家。根据用户的自然语言需求，提取结构化的搜索参数。
只输出一个JSON对象，不要有任何解释：

{
  "keyword": "高德地图搜索关键词（如：鲁菜、火锅、咖啡馆、电影院、KTV、酒吧）",
  "keyword_fallbacks": ["备选词1", "备选词2"],
  "min_rating": 4.0,
  "top_n": 20,
  "poi_category": "餐厅/咖啡馆/酒吧/电影院/KTV/购物/其他",
  "notes": "其他特殊需求，如价格范围、环境要求等，没有则空字符串",
  "sort_weights": {
    "rating": 0.4,
    "max_time": 0.4,
    "fairness": 0.2
  }
}

规则：
- keyword 要精准（"鲁菜"而非"好吃的鲁菜餐厅"）
- min_rating：用户提到"评分高"→4.5，"评分4.5以上"→4.5，未提及→4.0
- top_n 固定为20
- keyword_fallbacks 从窄到宽（例如"鲁菜"的备选：["山东菜","中餐"]）
- sort_weights 三项之和必须等于1.0，语义：
  * rating: 评分权重（越大越偏向高分店）
  * max_time: 最长通勤时长权重（越大越保护"最惨的人"，min-max 公平）
  * fairness: 通勤时长方差惩罚（越大越强调"每个人都差不多远"）
  * 默认：rating=0.4, max_time=0.4, fairness=0.2
  * 用户强调"近"/"方便"：rating=0.2, max_time=0.7, fairness=0.1
  * 用户强调"评分"/"好吃"：rating=0.7, max_time=0.2, fairness=0.1
  * 用户强调"公平"/"每个人差不多"：rating=0.3, max_time=0.3, fairness=0.4
"""


def agent_plan(user_query: str) -> dict:
    """
    Agent 1：规划 Agent
    输入：用户需求文字
    输出：结构化搜索参数
    """
    print(f"[Agent1/规划] 分析需求: {user_query}")
    try:
        resp = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user",   "content": user_query},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        content = resp.choices[0].message.content or ""
        plan = _extract_json(content)
        if isinstance(plan, dict) and "keyword" in plan:
            print(f"[Agent1/规划] 结果: {plan}")
            return plan
    except Exception as e:
        print(f"[Agent1/规划] 错误: {e}")

    # 降级：直接用用户输入作为 keyword
    return {
        "keyword": user_query,
        "keyword_fallbacks": [],
        "min_rating": 4.0,
        "top_n": 20,
        "poi_category": "未知",
        "notes": "",
        "sort_weights": {"rating": 0.4, "max_time": 0.4, "fairness": 0.2},
    }


# ──────────────────────────────────────────────────────
# Agent 2：搜索 Agent
# 职责：调用地图 API 搜索候选地点
# 特点：
#   - 只暴露 find_midpoint + search_pois_nearby 两个工具
#   - 工具返回给 LLM 的是压缩版，完整数据存 Python 侧 search_ctx
#   - 最多 10 轮工具调用
# ──────────────────────────────────────────────────────

_SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fair_center",
            "description": "计算 N 个参与者的几何中心点和建议搜索半径（在没有目标区域约束时使用）",
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lng": {"type": "number"},
                                "lat": {"type": "number"},
                            },
                            "required": ["lng", "lat"],
                        },
                        "description": "所有参与者的坐标",
                    },
                },
                "required": ["points"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pois_nearby",
            "description": "在中心点周边圆形范围内搜索地点。若结果<3，可扩大radius或换keyword重试。最多重试2次。",
            "parameters": {
                "type": "object",
                "properties": {
                    "center_lng": {"type": "number"},
                    "center_lat": {"type": "number"},
                    "keyword":    {"type": "string"},
                    "radius":     {"type": "integer", "description": "搜索半径（米）"},
                },
                "required": ["center_lng", "center_lat", "keyword", "radius"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_area",
            "description": "解析目标区域（城市或行政区）为地理边界，用于区域约束搜索。当用户明确指定'去朝阳'/'在海淀'/'去北京'等时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":     {"type": "string", "description": "城市名，如'北京'"},
                    "district": {"type": "string", "description": "行政区，如'朝阳区'（可选）"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pois_in_area",
            "description": "在已解析的行政区/城市范围内搜索地点。参数直接使用 resolve_area 返回的 bbox。",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_lng": {"type": "number"},
                    "min_lat": {"type": "number"},
                    "max_lng": {"type": "number"},
                    "max_lat": {"type": "number"},
                    "keyword": {"type": "string"},
                },
                "required": ["min_lng", "min_lat", "max_lng", "max_lat", "keyword"],
            },
        },
    },
]

_SEARCH_SYSTEM = """\
你是地图POI搜索专家，负责为多个参与者找到候选会面地点。

## 决策流程
1. 用 fair_center 算所有参与者的几何中心和建议半径。
2. 用 search_pois_nearby 以该中心为圆心做圆形搜索。

## 兜底策略
- 如果结果 < 3 个：
  * 用 keyword_fallbacks 里的备选词重试
  * search_pois_nearby 可以把 radius 扩大 1.5 倍再试
- 完成后输出 JSON：{"found": true, "count": N, "center": {...}}
  或 {"found": false, "reason": "..."}

## 注意
- 不要自行排名或过滤结果，只负责搜索
- 每次工具调用后看返回，判断是否需要下一步
"""


def agent_search(
    participants: list, plan: dict, city: str, search_ctx: dict,
    anchor_hint: dict | None = None,
    progress_cb=None,
) -> dict:
    """
    Agent 2：搜索 Agent（N 人 + 可选锚点）
    """
    cb = progress_cb or (lambda *a, **k: None)
    keyword   = plan.get("keyword", "餐厅")
    fallbacks = plan.get("keyword_fallbacks", [])

    # ── 有锚点：绕过 LLM，直接周边搜索 ──
    if anchor_hint and anchor_hint.get("lng") is not None and anchor_hint.get("lat") is not None:
        center = {"lng": float(anchor_hint["lng"]), "lat": float(anchor_hint["lat"])}
        radius_m = int(anchor_hint.get("radius_m") or 5000)
        aname = anchor_hint.get("name") or f"({center['lng']:.3f},{center['lat']:.3f})"
        r_km = round(radius_m / 1000, 1)
        cb("search", f"在「{aname}」周边 {r_km} km 内搜索 {keyword}")
        print(f"[Agent2/搜索] 锚点模式: {aname} r={radius_m}m, keyword={keyword}")
        pois: list[dict] = []
        used_kw = keyword
        for kw in [keyword] + list(fallbacks):
            r = amap_search_nearby(center["lng"], center["lat"], kw, radius=radius_m, sort_by="distance")
            if r.get("success") and r.get("pois"):
                pois = r["pois"]
                used_kw = kw
                print(f"[Agent2/搜索] 锚点周边命中 {len(pois)} 个 (kw={kw})")
                break
            if kw != keyword:
                cb("search", f"「{kw}」再试一次…")
        search_ctx["pois"] = pois
        search_ctx["last_center"] = center
        search_ctx["last_radius"] = radius_m
        return {
            "success": len(pois) > 0,
            "center":  center,
            "search_radius_m": radius_m,
            "anchor":  anchor_hint,
            "used_keyword": used_kw,
        }

    # ── 无锚点：走 LLM 循环 + fair_center ──
    cb("search", f"未指定锚点，让 Agent 找{keyword}的合适区域…")
    print(f"[Agent2/搜索] N={len(participants)}, keyword={keyword}, "
          f"fallbacks={fallbacks}（无锚点，用几何中心）")

    points_desc = "\n".join(
        f"- {p.get('name', f'P{i+1}')}: ({p['lng']}, {p['lat']})"
        for i, p in enumerate(participants)
    )

    messages = [
        {"role": "system", "content": _SEARCH_SYSTEM},
        {
            "role": "user",
            "content": (
                f"参与者（{len(participants)}人）:\n{points_desc}\n"
                f"搜索关键词: {keyword}\n"
                f"备选关键词: {', '.join(fallbacks) or '无'}\n"
                f"城市: {city}\n"
                f"请按决策流程完成搜索。"
            ),
        },
    ]

    for round_i in range(10):
        resp = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=_SEARCH_TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=1000,
        )
        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        if finish == "stop" or not msg.tool_calls:
            center = search_ctx.get("last_center") or {
                "lng": sum(p["lng"] for p in participants) / len(participants),
                "lat": sum(p["lat"] for p in participants) / len(participants),
            }
            radius = search_ctx.get("last_radius", 3000)
            found_count = len(search_ctx.get("pois", []))
            print(f"[Agent2/搜索] 完成，找到 {found_count} 个POI，共 {round_i + 1} 轮")
            return {
                "success": found_count > 0,
                "center":  center,
                "search_radius_m": radius,
                "anchor":  None,
            }

        messages.append(msg)
        tool_results = []
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = _exec_search_tool(name, args, search_ctx)
            print(f"[Agent2/搜索] 工具 {name}: {str(result)[:120]}")
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        messages.extend(tool_results)

    return {
        "success": len(search_ctx.get("pois", [])) > 0,
        "center":  search_ctx.get("last_center", {}),
        "search_radius_m": search_ctx.get("last_radius", 3000),
        "anchor":  None,
    }


def _exec_search_tool(name: str, args: dict, search_ctx: dict) -> dict:
    """执行搜索工具：完整数据存 search_ctx，压缩版返回给 LLM"""

    if name == "fair_center":
        pts = args.get("points", [])
        result = fair_meeting_point(pts)
        search_ctx["last_center"] = result.get("midpoint", {})
        search_ctx["last_radius"] = result.get("suggested_search_radius_m", 3000)
        return result

    # 保留旧名字兼容
    if name == "find_midpoint":
        result = find_balanced_midpoint(
            args["lng1"], args["lat1"], args["lng2"], args["lat2"],
        )
        search_ctx["last_center"] = result.get("midpoint", {})
        search_ctx["last_radius"] = result.get("suggested_search_radius_m", 3000)
        return result

    if name == "resolve_area":
        result = amap_district_polygon(args["city"], args.get("district"))
        if result.get("success"):
            search_ctx["last_center"] = result["center"]
            search_ctx["last_area"]   = result
        return result

    if name == "search_pois_in_area":
        bbox = {k: args[k] for k in ("min_lng", "min_lat", "max_lng", "max_lat")}
        full_result = amap_search_in_area(bbox, args["keyword"])
        full_pois = full_result.get("pois", [])
        if full_pois:
            existing = {p["name"] for p in search_ctx.get("pois", [])}
            for p in full_pois:
                if p["name"] not in existing:
                    search_ctx.setdefault("pois", []).append(p)
        return {
            "success": full_result.get("success", False),
            "count":   len(full_pois),
            "keyword": args["keyword"],
            "sample":  [_compact_poi(p) for p in full_pois[:5]],
        }

    if name == "search_pois_nearby":
        full_result = amap_search_nearby(
            args["center_lng"], args["center_lat"],
            args["keyword"], args.get("radius", 3000),
        )
        full_pois = full_result.get("pois", [])
        if full_pois:
            existing = {p["name"] for p in search_ctx.get("pois", [])}
            for p in full_pois:
                if p["name"] not in existing:
                    search_ctx.setdefault("pois", []).append(p)
        return {
            "success": full_result.get("success", False),
            "count":   len(full_pois),
            "keyword": args["keyword"],
            "radius":  args.get("radius", 3000),
            "sample":  [_compact_poi(p) for p in full_pois[:5]],
        }

    return {"success": False, "error": f"未知工具: {name}"}


# ──────────────────────────────────────────────────────
# 路线计算（纯 Python，不走 LLM）
# 职责：A/B 分别独立计算，支持不同出行方式，可重复调用
# ──────────────────────────────────────────────────────

def calculate_routes(
    pois: list,
    participants: list,
    city: str = "北京",
    departure_time: str | None = None,
    sort_weights: dict | None = None,
    progress_cb=None,
) -> list:
    """
    对每个 POI 逐个参与者算最优路线，得到 N 组 (mode, duration)，
    然后按 rating / max_time / fairness(std) 综合打分排序。
    """
    cb = progress_cb or (lambda *a, **k: None)
    w = sort_weights or {}
    w_rating   = float(w.get("rating",   0.4))
    w_max_time = float(w.get("max_time", 0.4))
    w_fairness = float(w.get("fairness", 0.2))
    # 兼容老字段
    if "total_time" in w:  w_max_time = float(w["total_time"])
    if "time_diff"  in w:  w_fairness = float(w["time_diff"])
    w_sum = w_rating + w_max_time + w_fairness or 1.0
    w_rating   /= w_sum
    w_max_time /= w_sum
    w_fairness /= w_sum

    print(f"[排序权重] 评分×{w_rating:.2f}  最长×{w_max_time:.2f}  公平×{w_fairness:.2f}")

    total = len(pois)
    n_part = len(participants)

    def _one_leg(pidx, i, person, poi):
        prefer = person.get("prefer", "auto") or "auto"
        route = amap_get_best_route(
            person["lng"], person["lat"],
            poi["lng"],    poi["lat"],
            city, prefer, departure_time,
        )
        # QPS/瞬时抖动导致全模式失败时重试
        attempts = 0
        while attempts < _ROUTE_LEG_RETRY and not route.get("success"):
            time.sleep(0.3 + 0.2 * attempts)
            attempts += 1
            route = amap_get_best_route(
                person["lng"], person["lat"],
                poi["lng"],    poi["lat"],
                city, prefer, departure_time,
            )
        leg = _format_route(route)
        leg["name"] = person.get("name", f"P{i + 1}")
        return pidx, i, leg

    # 展开所有 (POI, 参与者) 组合，一次性丢进线程池
    poi_legs: dict[int, list] = {pidx: [None] * n_part for pidx in range(total)}
    poi_remaining = {pidx: n_part for pidx in range(total)}
    enriched_by_idx: dict[int, dict] = {}
    done_count = 0

    with ThreadPoolExecutor(max_workers=_ROUTE_MAX_WORKERS) as pool:
        futs = [
            pool.submit(_one_leg, pidx, i, person, poi)
            for pidx, poi in enumerate(pois)
            for i, person in enumerate(participants)
        ]
        for f in as_completed(futs):
            pidx, i, leg = f.result()
            poi_legs[pidx][i] = leg
            poi_remaining[pidx] -= 1
            if poi_remaining[pidx] != 0:
                continue

            poi = pois[pidx]
            p = dict(poi)
            legs = poi_legs[pidx]
            p["legs"] = legs
            if len(legs) >= 1:
                p["transport_from_a"] = legs[0]
            if len(legs) >= 2:
                p["transport_from_b"] = legs[1]

            durations = [l.get("duration_minutes", 999) for l in legs]
            p["max_time_minutes"]    = max(durations)
            p["min_time_minutes"]    = min(durations)
            p["mean_time_minutes"]   = round(sum(durations) / len(durations), 1)
            p["time_spread_minutes"] = max(durations) - min(durations)
            p["total_time_minutes"]  = sum(durations)
            p["time_diff_minutes"]   = p["time_spread_minutes"]

            mean = p["mean_time_minutes"]
            variance = sum((d - mean) ** 2 for d in durations) / len(durations)
            p["time_std_minutes"] = round(variance ** 0.5, 1)

            enriched_by_idx[pidx] = p
            done_count += 1
            cb("routes_progress",
               f"{done_count}/{total} · {poi.get('name','')[:16]} · 最长 {p['max_time_minutes']} 分钟",
               {"done": done_count, "total": total})

    enriched = [enriched_by_idx[i] for i in range(total) if i in enriched_by_idx]

    # 归一化后加权
    max_max  = max((p["max_time_minutes"]  for p in enriched), default=1) or 1
    max_std  = max((p["time_std_minutes"]  for p in enriched), default=1) or 1
    max_rating = max((p.get("rating", 0)   for p in enriched), default=5) or 5

    for p in enriched:
        p["_score"] = round(
            (p.get("rating", 0)   / max_rating) *  w_rating
            - (p["max_time_minutes"] / max_max) *  w_max_time
            - (p["time_std_minutes"] / max_std) *  w_fairness,
            4,
        )

    enriched.sort(key=lambda x: x["_score"], reverse=True)
    return enriched


def filter_and_rank_pois(
    pois: list, min_rating: float = 4.0, top_n: int = 20
) -> list:
    """筛选评分 >= min_rating，按评分降序，取前 top_n 个（最多 20）"""
    top_n = min(top_n, 20)
    filtered = [p for p in pois if p.get("rating", 0) >= min_rating]
    filtered.sort(key=lambda x: x.get("rating", 0), reverse=True)
    if len(filtered) < 3 and pois:
        filtered = sorted(pois, key=lambda x: x.get("rating", 0), reverse=True)
    return filtered[:top_n]


# ──────────────────────────────────────────────────────
# Agent 3：总结 Agent
# 职责：根据结构化数据生成自然语言推荐文字
# 特点：无工具，上下文可控（只传摘要）
# ──────────────────────────────────────────────────────

_SUMMARY_SYSTEM = """\
你是一个简洁友好的推荐助手。根据 N 位参与者的会面地点候选数据，用2-4句话概括推荐结果。
重点提炼：
- 在哪个区域找到了什么类型的地点（如果指定了行政区，明确点出来）
- 评分最高的是哪家，全员通勤是否合理均衡
- 如果某地"最长通勤"明显偏大，简短提醒一下最惨的是谁
- 参与者多于 2 人时，用"三位/四位"这样的表达而不是"A/B"
语气活泼简洁，不要罗列所有细节，不要用"首先其次"等套话。
"""


def agent_summarize(
    query: str,
    participants: list[dict],
    enriched_pois: list,
    anchor: dict | None = None,
) -> str:
    """
    Agent 3：总结 Agent（N 人）
    """
    print("[Agent3/总结] 生成推荐文字...")
    mode_label = {
        "auto": "最快", "transit": "地铁公交",
        "driving": "驾车", "cycling": "骑行", "walking": "步行",
    }

    pois_summary = []
    for i, p in enumerate(enriched_pois[:5]):
        legs = p.get("legs", [])
        leg_txt = " / ".join(
            f"{l.get('name', '?')}{mode_label.get(l.get('mode', '?'), l.get('mode', '?'))}"
            f"{l.get('duration_text', '?')}"
            for l in legs
        )
        pois_summary.append(
            f"{i + 1}. {p['name']}（评分{p.get('rating', 0):.1f}）"
            f" - {leg_txt}"
            f"，最长{p.get('max_time_minutes', '?')}分钟"
        )

    participants_desc = "、".join(f"{p.get('name', f'P{i+1}')}({p.get('prefer', 'auto')})"
                                   for i, p in enumerate(participants))
    anchor_desc = ""
    if anchor and anchor.get("name"):
        r_km = round((anchor.get("radius_m") or 0) / 1000, 1)
        anchor_desc = f"\n会面锚点：{anchor['name']} 半径 {r_km}km"

    user_msg = (
        f"用户需求：{query}\n"
        f"参与者（{len(participants)}人）：{participants_desc}"
        f"{anchor_desc}\n"
        f"找到地点：\n" + "\n".join(pois_summary)
    )

    try:
        resp = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=300,
        )
        return resp.choices[0].message.content or "已找到推荐地点，请查看地图标注。"
    except Exception as e:
        print(f"[Agent3/总结] 错误: {e}")
        return f"已找到 {len(enriched_pois)} 个推荐地点。"


# ──────────────────────────────────────────────────────
# 完整流水线（供 /api/v2/search 调用）
# ──────────────────────────────────────────────────────

def _normalize_participants(data: dict) -> list[dict]:
    """
    统一入口：新格式 `participants:[...]` 优先，缺失时降级为老的 location_a/b。
    每个参与者最终结构：{"name", "lng", "lat", "prefer"}
    """
    raw = data.get("participants") or []
    if not raw:
        # 老接口回落
        la, lb = data.get("location_a"), data.get("location_b")
        if la and lb:
            raw = [
                {**la, "name": la.get("name", "A"), "prefer": data.get("prefer_a", "auto")},
                {**lb, "name": lb.get("name", "B"), "prefer": data.get("prefer_b", "auto")},
            ]

    out = []
    for i, p in enumerate(raw):
        if "lng" not in p or "lat" not in p:
            continue
        out.append({
            "name":   p.get("name") or f"P{i + 1}",
            "lng":    float(p["lng"]),
            "lat":    float(p["lat"]),
            "prefer": p.get("prefer", "auto") or "auto",
        })
    return out


def run_pipeline(
    user_query: str,
    participants: list[dict],
    city: str = "北京",
    departure_time: str | None = None,
    anchor_hint: dict | None = None,
    progress_cb=None,
) -> dict:
    """
    运行完整多 Agent 流水线（N 人 + 可选锚点）。
    """
    cb = progress_cb or (lambda *a, **k: None)
    if len(participants) < 2:
        return {"success": False, "error": "至少需要 2 位参与者"}

    # ── Agent 1：规划 ──
    cb("plan", "解析你的需求…")
    plan = agent_plan(user_query)
    kw = plan.get("keyword", "?")
    fbs = plan.get("keyword_fallbacks") or []
    fb_txt = f" · 备选 {'/'.join(fbs[:3])}" if fbs else ""
    cb("plan_done",
       f"关键词 {kw}（评分 ≥ {plan.get('min_rating', 4.0)}，取前 {plan.get('top_n', 10)}）{fb_txt}",
       {"plan": plan})

    # ── Agent 2：搜索（锚点优先） ──
    search_ctx: dict = {"pois": []}
    search_result = agent_search(
        participants, plan, city, search_ctx,
        anchor_hint=anchor_hint, progress_cb=cb,
    )

    if not search_result.get("success") or not search_ctx.get("pois"):
        cb("error", "未找到符合条件的地点，请换关键词或扩大锚点半径")
        return {"success": False, "error": "未能找到符合条件的地点，请尝试修改关键词或扩大范围"}

    center          = search_result.get("center", {})
    search_radius_m = search_result.get("search_radius_m", 3000)
    anchor          = search_result.get("anchor")
    found_n         = len(search_ctx.get("pois", []))
    cb("search_done", f"找到 {found_n} 个候选地点", {"count": found_n})

    # ── 筛选排名 ──
    top_pois = filter_and_rank_pois(
        search_ctx["pois"],
        min_rating=plan.get("min_rating", 4.0),
        top_n=plan.get("top_n", 20),
    )
    cb("rank", f"评分筛选后取 {len(top_pois)} 个候选", {"count": len(top_pois)})
    print(f"[Pipeline] 筛选后 {len(top_pois)} 个POI")

    # ── 路线计算 + 打分 ──
    cb("routes_start",
       f"为 {len(top_pois)} 个地点计算 {len(participants)} 人的最优路线…",
       {"total": len(top_pois), "participants": len(participants)})
    enriched = calculate_routes(
        top_pois, participants, city, departure_time,
        sort_weights=plan.get("sort_weights"),
        progress_cb=cb,
    )
    cb("routes_done", f"路线计算完成", {"count": len(enriched)})

    # ── Agent 3：总结 ──
    cb("summary", "生成推荐总结…")
    summary_text = agent_summarize(
        user_query, participants, enriched, anchor
    )
    cb("summary_done", "总结完成")

    # ── Session ──
    session_id = session_create({
        "participants":    participants,
        "city":            city,
        "query":           user_query,
        "plan":            plan,
        "center":          center,
        "search_radius_m": search_radius_m,
        "anchor":          anchor,
        "pois_base":       top_pois,
        "last_pois":       enriched,
        "departure_time":  departure_time,
        "chat_history":    [],
    })

    result = {
        "success":         True,
        "session_id":      session_id,
        "summary":         summary_text,
        "plan":            plan,
        "participants":    participants,
        "center":          center,
        "search_radius_m": search_radius_m,
        "anchor":          anchor,
        "pois":            enriched,
    }
    # 老前端兼容字段
    if len(participants) >= 2:
        result["midpoint"] = center
        result["prefer_a"] = participants[0].get("prefer", "auto")
        result["prefer_b"] = participants[1].get("prefer", "auto")
    return result


# ──────────────────────────────────────────────────────
# Flask 路由
# ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config")
def get_config():
    return jsonify({
        "amap_key":     AMAP_JS_KEY,
        "amap_js_code": os.getenv("AMAP_JS_CODE", ""),
        "has_amap_key": bool(AMAP_JS_KEY),
        "version":      "v2",
    })


@app.route("/api/v2/search", methods=["POST"])
def api_v2_search():
    """
    主搜索接口：多 Agent 流水线（N 人 + 可选锚点）
    请求：{
      participants: [{name,lng,lat,prefer?}, ...],   // 或老字段 location_a/b
      anchor:       {lng, lat, name?, radius_m} | null,  // 可选，前端选定的会面锚点
      query, city, departure_time
    }
    """
    data           = request.json or {}
    user_query     = data.get("query", "")
    city           = data.get("city", "北京")
    departure_time = data.get("departure_time") or None
    anchor         = data.get("anchor") or None
    # 老字段兼容：如果传的还是 target_area，忽略并打日志（新模型下无意义）
    if anchor is None and data.get("target_area"):
        print(f"[Compat] 收到旧字段 target_area={data.get('target_area')}，已忽略（请改传 anchor）")

    participants = _normalize_participants(data)

    if len(participants) < 2:
        return jsonify({"success": False, "error": "至少需要 2 位参与者"}), 400
    if not user_query:
        return jsonify({"success": False, "error": "请描述您的需求"}), 400
    if not AMAP_KEY:
        return jsonify({"success": False, "error": "高德地图 API Key 未配置"}), 500
    if not DEEPSEEK_API_KEY:
        return jsonify({"success": False, "error": "DeepSeek API Key 未配置"}), 500

    try:
        result = run_pipeline(
            user_query, participants,
            city=city,
            departure_time=departure_time,
            anchor_hint=anchor,
        )
        return jsonify(result), (200 if result.get("success") else 500)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/v2/search-stream", methods=["POST"])
def api_v2_search_stream():
    """
    SSE 流式搜索接口。请求体同 /api/v2/search，响应为 text/event-stream，
    每条 `data: {json}\\n\\n` 为一个进度事件。
    事件 stage: plan | plan_done | search | search_done | rank
                | routes_start | routes_progress | routes_done
                | summary | summary_done | result | error
    """
    data           = request.json or {}
    user_query     = data.get("query", "")
    city           = data.get("city", "北京")
    departure_time = data.get("departure_time") or None
    anchor         = data.get("anchor") or None
    if anchor is None and data.get("target_area"):
        print(f"[Compat] search-stream 收到旧字段 target_area={data.get('target_area')}，已忽略")

    participants = _normalize_participants(data)

    def err_stream(msg: str, code: int = 400):
        payload = json.dumps({"stage": "error", "msg": msg, "data": None}, ensure_ascii=False)
        return Response(f"data: {payload}\n\n", mimetype="text/event-stream",
                        status=code, headers={"Cache-Control": "no-cache",
                                              "X-Accel-Buffering": "no"})

    if len(participants) < 2:
        return err_stream("至少需要 2 位参与者", 400)
    if not user_query:
        return err_stream("请描述您的需求", 400)
    if not AMAP_KEY:
        return err_stream("高德地图 API Key 未配置", 500)
    if not DEEPSEEK_API_KEY:
        return err_stream("DeepSeek API Key 未配置", 500)

    events: "queue.Queue" = queue.Queue()
    SENTINEL = object()

    def cb(stage: str, msg: str = "", extra=None):
        events.put({"stage": stage, "msg": msg, "data": extra})

    def worker():
        try:
            r = run_pipeline(
                user_query, participants,
                city=city,
                departure_time=departure_time,
                anchor_hint=anchor,
                progress_cb=cb,
            )
            if r.get("success"):
                events.put({"stage": "result", "msg": "完成", "data": r})
            else:
                events.put({"stage": "error", "msg": r.get("error", "未知错误"), "data": None})
        except Exception as e:
            import traceback; traceback.print_exc()
            events.put({"stage": "error", "msg": str(e), "data": None})
        finally:
            events.put(SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        # SSE 连接心跳的初始注释行
        yield ": stream open\n\n"
        while True:
            try:
                evt = events.get(timeout=15)
            except queue.Empty:
                # 心跳保活（Nginx / 代理）
                yield ": ping\n\n"
                continue
            if evt is SENTINEL:
                break
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


@app.route("/api/v2/routes", methods=["POST"])
def api_v2_routes():
    """
    路线重算接口：复用 session 的 POI，重新按当前出行方式算路。
    """
    data           = request.json or {}
    session_id     = data.get("session_id")
    departure_time = data.get("departure_time") or None
    # 新接口：允许前端更新每个参与者的 prefer；也接受旧的 prefer_a/prefer_b
    prefer_overrides = data.get("prefer_overrides") or {}
    prefer_a       = data.get("prefer_a")
    prefer_b       = data.get("prefer_b")

    if not session_id:
        return jsonify({"success": False, "error": "缺少 session_id，请先执行搜索"}), 400

    session = session_get(session_id)
    if not session:
        return jsonify({"success": False, "error": "会话已过期，请重新搜索"}), 404

    pois_base = session.get("pois_base", [])
    if not pois_base:
        return jsonify({"success": False, "error": "缓存的搜索结果为空"}), 400

    participants = [dict(p) for p in session.get("participants", [])]
    # 合并 prefer 覆盖
    for i, person in enumerate(participants):
        override = prefer_overrides.get(person.get("name")) \
                or prefer_overrides.get(str(i))
        if override:
            person["prefer"] = override
        elif i == 0 and prefer_a:
            person["prefer"] = prefer_a
        elif i == 1 and prefer_b:
            person["prefer"] = prefer_b

    if departure_time is None:
        departure_time = session.get("departure_time")

    try:
        enriched = calculate_routes(
            pois_base,
            participants,
            session.get("city", "北京"),
            departure_time,
            sort_weights=session.get("plan", {}).get("sort_weights"),
        )
        session_update(session_id, {
            "participants":   participants,
            "departure_time": departure_time,
        })
        resp = {
            "success":         True,
            "session_id":      session_id,
            "pois":            enriched,
            "participants":    participants,
            "center":          session.get("center", {}),
            "search_radius_m": session.get("search_radius_m", 3000),
            "anchor":          session.get("anchor"),
        }
        # 老字段兼容
        if len(participants) >= 2:
            resp["midpoint"] = session.get("center", {})
            resp["prefer_a"] = participants[0].get("prefer", "auto")
            resp["prefer_b"] = participants[1].get("prefer", "auto")
        return jsonify(resp)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/v2/session/<session_id>", methods=["GET"])
def api_v2_session_info(session_id):
    """查看 Session 摘要（调试用）"""
    s = session_get(session_id)
    if not s:
        return jsonify({"exists": False}), 404
    return jsonify({
        "exists":       True,
        "query":        s.get("query"),
        "city":         s.get("city"),
        "poi_count":    len(s.get("pois_base", [])),
        "participants": s.get("participants", []),
        "anchor":       s.get("anchor"),
        "plan":         s.get("plan"),
    })


@app.route("/api/geocode", methods=["POST"])
def api_geocode():
    data = request.json or {}
    address = data.get("address", "")
    if not address:
        return jsonify({"success": False, "error": "请提供地址"}), 400
    return jsonify(amap_geocode(address))


@app.route("/api/resolve-area", methods=["POST"])
def api_resolve_area():
    """{city, district?} → 行政区 bbox/center。前端目标区域实时预览用。"""
    data = request.json or {}
    city = (data.get("city") or "").strip()
    district = (data.get("district") or "").strip() or None
    if not city and not district:
        return jsonify({"success": False, "error": "请提供 city 或 district"}), 400
    result = amap_district_polygon(city or district, district if city else None)
    result["city"] = city
    result["district"] = district
    return jsonify(result)


@app.route("/api/nearby-search", methods=["POST"])
def api_nearby_search():
    """
    查询某坐标附近的 POI（用于卡片内"附近搜索"功能）。
    请求：{ lng, lat, keyword, radius_m (可选，默认 1000) }
    返回：{ success, pois: [{name, address, distance_m, rating, lng, lat, type}] }
    """
    data     = request.json or {}
    lng      = data.get("lng")
    lat      = data.get("lat")
    keyword  = data.get("keyword", "").strip()
    radius_m = min(int(data.get("radius_m", 1000)), 5000)

    if not lng or not lat or not keyword:
        return jsonify({"success": False, "error": "缺少参数 lng/lat/keyword"}), 400

    url = "https://restapi.amap.com/v3/place/around"
    params = {
        "key":       AMAP_KEY,
        "location":  f"{lng},{lat}",
        "keywords":  keyword,
        "radius":    radius_m,
        "offset":    10,
        "page":      1,
        "extensions": "base",
        "output":    "json",
    }
    try:
        resp   = requests.get(url, params=params, timeout=8)
        result = resp.json()
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "pois": []})

    if result.get("status") != "1":
        return jsonify({"success": False, "error": result.get("info", "搜索失败"), "pois": []})

    pois = []
    for p in result.get("pois", []):
        loc = p.get("location", "").split(",")
        if len(loc) != 2:
            continue
        try:
            plng, plat = float(loc[0]), float(loc[1])
            dist       = int(haversine_distance(lng, lat, plng, plat) * 1000)
            biz_ext    = p.get("biz_ext") or {}
            rating     = float(biz_ext.get("rating", 0) or 0) if isinstance(biz_ext, dict) else 0.0
            pois.append({
                "name":       p.get("name", ""),
                "address":    p.get("address", "") if isinstance(p.get("address"), str) else "",
                "type":       p.get("type", ""),
                "distance_m": dist,
                "rating":     rating,
                "lng":        plng,
                "lat":        plat,
            })
        except (ValueError, TypeError):
            continue

    pois.sort(key=lambda x: x["distance_m"])
    return jsonify({"success": True, "pois": pois[:10]})


_GEOCODE_SUGGEST_CACHE: dict[str, tuple[float, list[dict]]] = {}
_GEOCODE_SUGGEST_TTL = 300  # 5 min

@app.route("/api/geocode-suggest", methods=["POST"])
def api_geocode_suggest():
    """
    地点输入提示接口 — 前端搜索框下拉候选。
    调用高德 /v3/assistant/inputtips，返回带坐标的候选列表。
    带 LRU + TTL 缓存，同 keyword 5 分钟内直返，避免每次敲键都打高德。
    """
    import time as _time
    data    = request.json or {}
    keyword = data.get("keyword", "").strip()
    if not keyword:
        return jsonify({"tips": []})

    now = _time.time()
    key = keyword.lower()
    cached = _GEOCODE_SUGGEST_CACHE.get(key)
    if cached and (now - cached[0]) < _GEOCODE_SUGGEST_TTL:
        return jsonify({"tips": cached[1]})

    url = "https://restapi.amap.com/v3/assistant/inputtips"
    params = {
        "key":      AMAP_KEY,
        "keywords": keyword,
        "datatype": "all",
        "output":   "json",
    }
    try:
        resp   = requests.get(url, params=params, timeout=8)
        result = resp.json()
    except Exception as e:
        return jsonify({"tips": [], "error": str(e)})

    tips = []
    if result.get("status") == "1":
        for tip in result.get("tips", []):
            location = tip.get("location", "")
            if not location or location == "[]":
                continue
            try:
                lng_s, lat_s = location.split(",")
                tips.append({
                    "name":     tip.get("name", ""),
                    "district": tip.get("district", ""),
                    "address":  tip.get("address", "") if isinstance(tip.get("address"), str) else "",
                    "lng":      float(lng_s),
                    "lat":      float(lat_s),
                })
            except (ValueError, TypeError):
                continue

    tips = tips[:6]
    _GEOCODE_SUGGEST_CACHE[key] = (now, tips)
    # 简易 LRU：超上限踢最旧
    if len(_GEOCODE_SUGGEST_CACHE) > 500:
        oldest_key = min(_GEOCODE_SUGGEST_CACHE, key=lambda k: _GEOCODE_SUGGEST_CACHE[k][0])
        _GEOCODE_SUGGEST_CACHE.pop(oldest_key, None)

    return jsonify({"tips": tips})


# ══════════════════════════════════════════════════════
# AI 助手：SSE 流式 + DeepSeek Tool Calling
# ══════════════════════════════════════════════════════

def _sse(event: dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


_FALSE_MISSING_LOCATION_MARKERS = (
    "你的位置还没设", "你的位置还没填", "你的位置没有设置", "你的位置还未设置",
    "你的位置还不知道", "不知道你的位置", "告诉我你在哪", "告诉我你在哪里",
    "点地图上的“定位到我”", "点地图上的「定位到我」", "请再定位", "需要你的当前位置",
)


def _guard_assistant_location_claim(text: str, me_has_location: bool) -> str:
    """快照已有本人坐标时，不允许模型向用户陈述相反事实。"""
    if not me_has_location or not text or not any(x in text for x in _FALSE_MISSING_LOCATION_MARKERS):
        return text
    sentences = re.split(r"(?<=[。！？!?])\s*|\n+", text)
    kept = [s for s in sentences if s and not any(x in s for x in _FALSE_MISSING_LOCATION_MARKERS)]
    clean = "".join(kept).strip()
    if clean:
        return clean + "\n\n你的位置已经设置好了，我会直接使用左侧当前地点继续规划。"
    return "你的位置已经设置好了，我会直接使用左侧当前地点继续规划。"


ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "offer_choices",
            "description": "当缺少的信息适合用户点击选择时，展示2到5个候选项。用户可以选择后补充文字再发送，也可以忽略选项直接输入。只用于澄清/选择，不要在信息已足够时滥用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type":"string","description":"简短问题"},
                    "mode": {"type":"string","enum":["single","multiple"],"description":"互斥答案必须用single：同一人的交通方式、是否采用记忆、预算区间。只有可同时成立的条件（如多个忌口，或分别为不同人物选择）才用multiple。"},
                    "options": {"type":"array","minItems":2,"maxItems":5,"items":{"type":"object","properties":{
                        "label":{"type":"string","description":"按钮短标签"},
                        "value":{"type":"string","description":"合并进用户下一条消息的自然语言内容"}
                    },"required":["label","value"]}}
                },
                "required":["question","mode","options"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": "保存用户明确要求长期记住的个人偏好。仅限交通、饮食、预算；一次性的‘今天/这次’不要保存，位置和朋友资料禁止保存。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["transport", "food", "budget"]},
                    "key": {"type": "string", "description": "规范键：交通用 default_mode；饮食用具体偏好键；预算用 per_person_max。"},
                    "value": {"type": "string", "description": "简洁、用户可读的记忆内容。交通统一用 公交/骑行/驾车/步行/最快；预算只写金额数字。"},
                },
                "required": ["category", "key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "读取小 Mid 已为当前用户保存的长期偏好。用户问‘你记得我什么’时调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": "按类别或具体键忘记当前用户的长期偏好。用户说忘掉、删除、不再记住时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["transport", "food", "budget", "all"]},
                    "key": {"type": "string", "description": "可选；省略则删除整个类别。"},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_person",
            "description": "在用户明确要求记住时保存人物关系及常用出发地。必须是用户主动提供；不得推断。地点默认90天过期。",
            "parameters": {"type":"object","properties":{
                "name":{"type":"string"}, "relation":{"type":"string"},
                "usual_place":{"type":"string"}, "city":{"type":"string"},
                "days":{"type":"integer","description":"有效天数，默认90，最大365"}
            },"required":["name"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_feedback",
            "description": "保存用户对某家店的明确反馈：喜欢、去过、不喜欢。只有用户明确表达时调用。",
            "parameters": {"type":"object","properties":{
                "poi_name":{"type":"string"},
                "signal":{"type":"string","enum":["liked","visited","disliked"]},
                "reason":{"type":"string"}
            },"required":["poi_name","signal"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pois",
            "description": "在指定坐标周围搜索 POI（餐厅/咖啡/景点等）。会替换掉当前的推荐结果列表并把新结果传回前端展示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword":    {"type": "string", "description": "搜索关键词，如『咖啡』『川菜』『安静的酒吧』"},
                    "center_lng": {"type": "number", "description": "中心点经度。若省略则用当前锚点。"},
                    "center_lat": {"type": "number", "description": "中心点纬度。若省略则用当前锚点。"},
                    "radius_m":   {"type": "integer", "description": "搜索半径（米）。默认 3000，最大 50000。"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shift_center",
            "description": "【草稿】提议把会面锚点移到某个坐标或地名。**不会立刻生效**——只是把提议扔到用户面前的草稿卡里等他确认。可以放心调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "地名，如『三里屯』『国贸』；给了就会自动地理编码"},
                    "city": {"type": "string", "description": "地名所在城市，如『杭州』『上海』。用户明确提到跨城市地名（例『杭州文三路』『上海外滩』）时必须传。同城可省略。"},
                    "lng":  {"type": "number", "description": "经度（如果 name 已经能定位则可省略）"},
                    "lat":  {"type": "number", "description": "纬度"},
                    "radius_m": {"type": "integer", "description": "同时调整半径（可选）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_participant_location",
            "description": "【草稿】提议改某个参与者的位置**或/和昵称**。用 index (1-based) 或 participant_name 定位。可以只改昵称（省略 place_name）、只改位置（省略 new_nickname）、或两个一起改。**不会立刻生效**，进草稿卡等用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "index":            {"type": "integer", "description": "参与者序号（1-based）。和 participant_name 二选一。用户说『我 / 我自己 / 咱』时，**必须**用 [当前会话快照] 里的 `me_index`——**永远别硬编码 1**，房间模式下你可能是 index=2 或更后。"},
                    "participant_name": {"type": "string",  "description": "参与者姓名精确匹配"},
                    "place_name":       {"type": "string",  "description": "新位置的地名，如『望京SOHO』。只改昵称时可省略。"},
                    "city":             {"type": "string",  "description": "跨城市时必填"},
                    "lng":               {"type": "number"},
                    "lat":               {"type": "number"},
                    "new_nickname":     {"type": "string",  "description": "新昵称。用户提到朋友名字（如『我闺蜜 Lisa』『同事小王』）时，把对应参与者的昵称改成那个人名字（Lisa / 小王）。"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_participant",
            "description": "【草稿】提议**新增**一位参与者。用户说『我闺蜜 Lisa 在对外经贸』『再加个从望京来的同事王小明』时用这个。**不会立刻生效**，进草稿卡等用户确认。房间模式下禁止：让用户分享房间码让本人加入。",
            "parameters": {
                "type": "object",
                "properties": {
                    "nickname":   {"type": "string", "description": "新参与者昵称，如『Lisa』『王小明』；用户没提名字时用『朋友』『同事』等占位。"},
                    "place_name": {"type": "string", "description": "地名，如『对外经济贸易大学』『望京SOHO』。可省略——那样只加占位人由用户后填。"},
                    "city":       {"type": "string", "description": "跨城市时必填"},
                    "lng":        {"type": "number"},
                    "lat":        {"type": "number"},
                    "prefer":     {"type": "string", "enum": ["auto","transit","driving","walking","cycling"], "description": "交通偏好，默认 auto"},
                },
                "required": ["nickname"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_keyword",
            "description": "【草稿】提议改搜索关键词（如从『咖啡』改成『日料』）。**不会立刻生效**，进草稿卡等用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "新的搜索关键词"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_radius",
            "description": "【草稿】提议改搜索半径（米）。**不会立刻生效**，进草稿卡等用户确认。范围 300~60000。",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius_m": {"type": "integer", "description": "半径（米），300~60000"},
                },
                "required": ["radius_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recompute_routes",
            "description": "基于当前的 POI 列表和参与者位置，重新算每人到每个 POI 的路线时长。适合改了锚点/参与者后调用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_result",
            "description": "获取当前会话的锚点、参与者、Top POI 概览。用于回答用户关于当前结果的问题。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _compute_me_index(participants: list, my_did: str) -> int:
    """算出'我'在参与者列表中的 1-based index。
    房间模式：找 id == 'room-<my_did>' 的那位。
    Solo 模式（或找不到）：默认 index=1。
    """
    if my_did:
        target = f"room-{my_did}"
        for i, p in enumerate(participants, start=1):
            if str(p.get("id") or "") == target:
                return i
    return 1


def _assistant_get_state(sid: str) -> dict:
    """AI 助手拿到的会话状态镜像。"""
    s = session_get(sid) or {}
    return {
        "anchor":       s.get("anchor"),
        "participants": s.get("participants") or [],
        "pois":         s.get("last_pois") or s.get("pois") or [],
        "query":        s.get("query", ""),
        "city":         s.get("city", "北京"),
        "my_did":       s.get("my_did") or "",
    }


def _tool_search_pois(sid: str, args: dict) -> tuple[dict, dict | None]:
    st = _assistant_get_state(sid)
    keyword = (args.get("keyword") or "").strip()
    if not keyword:
        return {"ok": False, "error": "缺少 keyword"}, None
    center_lng = args.get("center_lng")
    center_lat = args.get("center_lat")
    if center_lng is None or center_lat is None:
        anchor = st["anchor"] or {}
        center_lng = anchor.get("lng")
        center_lat = anchor.get("lat")
    # 无锚点时用参与者中点兜底（对齐"自动搜"场景：AI 刚设完位置就搜，此时锚点通常还没定）
    if center_lng is None or center_lat is None:
        pts = [p for p in (st.get("participants") or []) if p.get("lng") is not None and p.get("lat") is not None]
        if len(pts) >= 1:
            mp = fair_meeting_point(pts)
            mid = mp.get("midpoint") or {}
            if mid.get("lng") is not None and mid.get("lat") is not None:
                center_lng = mid["lng"]
                center_lat = mid["lat"]
    if center_lng is None or center_lat is None:
        return {"ok": False, "error": "没有 center 坐标，且当前无锚点/参与者可推算"}, None
    radius = int(args.get("radius_m") or 3000)
    radius = max(500, min(50000, radius))
    r = amap_search_nearby(float(center_lng), float(center_lat), keyword, radius=radius)
    if not r.get("success"):
        return {"ok": False, "error": r.get("error", "搜索失败")}, None
    pois = r.get("pois", []) or []
    # 对高德返回的全部候选计算路线和综合分；不能只让距离靠前的少数 POI 参与排名。
    participants_for_routes = [
        p for p in (st.get("participants") or [])
        if p.get("lng") is not None and p.get("lat") is not None
    ]
    enriched = pois
    if pois and participants_for_routes:
        try:
            enriched = calculate_routes(
                pois, participants_for_routes,
                st.get("city", "北京"), None,
                sort_weights=None,
            )
            enriched = _apply_feedback_ranking(_memory_device_id(sid), enriched)
        except Exception as e:
            print(f"[assistant search_pois] calculate_routes 失败：{e}")
    # 更新 session
    session_update(sid, {"last_pois": enriched, "query": keyword})
    top = [
        {
            "name":            p.get("name"),
            "address":         p.get("address"),
            "rating":          p.get("rating"),
            "cost_per_person": p.get("cost_per_person"),
        }
        for p in enriched[:6]
    ]
    summary = f"在 ({center_lng:.4f},{center_lat:.4f}) 附近 {radius}m 内找到 {len(enriched)} 家「{keyword}」"
    return {"ok": True, "summary": summary, "count": len(enriched), "top": top}, {
        "type":  "pois_replaced",
        "pois":  enriched,
        "participants": participants_for_routes,
        "anchor": st["anchor"],
        "center": {"lng": float(center_lng), "lat": float(center_lat), "radius_m": radius},
    }


_KNOWN_CITY_HINTS = (
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都", "重庆",
    "武汉", "西安", "天津", "青岛", "厦门", "宁波", "无锡", "长沙", "郑州",
    "沈阳", "大连", "哈尔滨", "长春", "济南", "合肥", "福州", "南昌", "昆明",
    "贵阳", "南宁", "海口", "三亚", "拉萨", "兰州", "银川", "西宁", "乌鲁木齐",
    "呼和浩特", "石家庄", "太原", "香港", "澳门", "台北", "高雄",
)


def _extract_city(name: str) -> str | None:
    """从地名字符串前缀里提取城市名，找不到返回 None。"""
    if not name:
        return None
    n = name.strip()
    for c in _KNOWN_CITY_HINTS:
        if n.startswith(c) or n.startswith(c + "市"):
            return c
    return None


def _tool_shift_center(sid: str, args: dict) -> tuple[dict, dict | None]:
    """草稿档：只算出建议锚点，不落盘。等用户在草稿卡按"应用"后 client 会调 apply-drafts 同步。"""
    lng = args.get("lng")
    lat = args.get("lat")
    name = (args.get("name") or "").strip()
    explicit_city = (args.get("city") or "").strip() or None
    st = _assistant_get_state(sid)
    session_city = st.get("city") or "北京"

    if lng is None or lat is None:
        if not name:
            return {"ok": False, "error": "需要 name 或 (lng,lat)"}, None

        detected_city = _extract_city(name)
        target_city = explicit_city or detected_city or session_city

        if explicit_city and not name.startswith(explicit_city):
            query = f"{explicit_city}{name}"
        elif detected_city:
            query = name
        else:
            query = f"{target_city}{name}"

        geo = amap_geocode(query)
        if not geo.get("success"):
            geo = amap_geocode(name)
        if not geo.get("success") and target_city and target_city != session_city:
            geo = amap_geocode(f"{target_city}{name.lstrip(target_city)}")
        if not geo.get("success"):
            return {"ok": False, "error": geo.get("error", f"『{name}』无法定位")}, None
        lng = geo["lng"]; lat = geo["lat"]
        located_city = geo.get("city") or target_city
    else:
        located_city = explicit_city or session_city

    old = st.get("anchor") or {}
    radius_m = int(args.get("radius_m") or (old.get("radius_m") or 5000))
    anchor = {
        "lng": float(lng), "lat": float(lat),
        "name": name or old.get("name") or "自定义锚点",
        "radius_m": radius_m, "source": "assistant",
    }
    label = f"锚点 → {anchor['name']}"
    return (
        {"ok": True, "summary": f"提议把锚点移到 {anchor['name']}（待你在草稿卡确认）"},
        {
            "type": "draft", "kind": "shift_center", "label": label,
            "detail": f"{lng:.4f}, {lat:.4f} · 半径 {radius_m}m",
            "data": {"anchor": anchor, "city": located_city},
        },
    )


def _tool_set_participant_location(sid: str, args: dict) -> tuple[dict, dict | None]:
    """草稿档：改某个参与者的位置。用 index (1-based) 或 name 定位；地名会自动地理编码。"""
    st = _assistant_get_state(sid)
    parts = st.get("participants") or []
    if not parts:
        return {"ok": False, "error": "当前没有参与者，用 add_participant 先加人"}, None

    idx = args.get("index")
    name_key = (args.get("participant_name") or "").strip()
    target = None
    if isinstance(idx, int) and 1 <= idx <= len(parts):
        target = parts[idx - 1]
    elif name_key:
        for p in parts:
            if (p.get("name") or "").strip() == name_key:
                target = p; break
    if not target:
        return {"ok": False, "error": "找不到指定的参与者，请给 index 或 participant_name"}, None

    # 硬约束：房间模式下 AI 只能改调用者自己那行
    # 房间成员的本地 id 形如 "room-<device_id>"；solo 模式无此前缀，不受限
    my_did = st.get("my_did") or ""
    tid = str(target.get("id") or "")
    if tid.startswith("room-") and my_did and tid != f"room-{my_did}":
        return {
            "ok": False,
            "error": (f"你只能通过 AI 改自己的地址。要改 {target.get('name','?')} 的地址，"
                      f"请让本人在自己端操作。"),
        }, None

    place_name = (args.get("place_name") or "").strip()
    lng = args.get("lng"); lat = args.get("lat")
    explicit_city = (args.get("city") or "").strip() or None
    session_city = st.get("city") or "北京"
    new_nickname = (args.get("new_nickname") or "").strip() or None

    # 允许"只改昵称不改位置"：place_name/lng/lat 全空但 new_nickname 有 → 只 rename
    location_specified = bool(place_name) or (lng is not None and lat is not None)
    if not location_specified and not new_nickname:
        return {"ok": False, "error": "需要 place_name / (lng,lat) 或 new_nickname 至少一个"}, None

    address = None
    if location_specified:
        if lng is None or lat is None:
            detected = _extract_city(place_name)
            target_city = explicit_city or detected or session_city
            query = place_name if detected else f"{target_city}{place_name}"
            geo = amap_geocode(query, city=target_city)
            if not geo.get("success"):
                geo = amap_geocode(place_name, city=target_city)
            if not geo.get("success"):
                return {"ok": False, "error": geo.get("error", f"『{place_name}』无法定位")}, None
            lng = geo["lng"]; lat = geo["lat"]
            address = geo.get("formatted_address") or place_name
        else:
            address = place_name or f"{lat:.4f}, {lng:.4f}"

    old_name = target.get("name", "?")
    display_name = new_nickname or old_name
    if location_specified and new_nickname:
        label = f"{old_name} → {new_nickname} @ {address}"
        detail = f"改名 + 定位到 {address}"
    elif location_specified:
        label = f"{old_name} → {address}"
        detail = f"{lng:.4f}, {lat:.4f}"
    else:
        label = f"{old_name} → 昵称改为 {new_nickname}"
        detail = ""

    data: dict = {"participant_id": target.get("id")}
    if location_specified:
        data["lng"] = float(lng); data["lat"] = float(lat); data["address"] = address
    if new_nickname:
        data["new_nickname"] = new_nickname

    summary = (
        f"提议把 {old_name} 改名为 {new_nickname}"
        + (f"、位置改为 {address}" if location_specified else "")
    ) if new_nickname else f"提议把 {old_name} 的位置改为 {address}"

    return (
        {"ok": True, "summary": summary},
        {
            "type": "draft", "kind": "set_participant_location", "label": label,
            "detail": detail,
            "data": data,
        },
    )


def _tool_add_participant(sid: str, args: dict) -> tuple[dict, dict | None]:
    """草稿档：新增一位参与者（solo 模式）。房间模式禁止 —— 让用户分享房间码让本人加入。"""
    st = _assistant_get_state(sid)
    my_did = st.get("my_did") or ""
    parts = st.get("participants") or []

    # 房间模式判定：有 my_did 且列表中至少一位 id 形如 "room-..."
    in_room = my_did and any(str(p.get("id") or "").startswith("room-") for p in parts)
    if in_room:
        return {
            "ok": False,
            "error": "房间模式下不能由 AI 加人。请把房间号分享给对方，让本人加入房间。",
        }, None

    if len(parts) >= 6:
        return {"ok": False, "error": "最多 6 位参与者"}, None

    # 空位守卫：如果列表里还有 lng/lat 为 null 的 slot，先让 AI 覆盖空位再新增。
    # 这是硬约束——防止 AI 无视 prompt 里的「空位优先覆盖」规则直接堆人。
    empty_slots = [(i + 1, p) for i, p in enumerate(parts) if p.get("lng") is None or p.get("lat") is None]
    if empty_slots:
        idx, slot = empty_slots[0]
        return {
            "ok": False,
            "error": (
                f"当前还有空位（index={idx} 『{slot.get('name', '?')}』尚未定位）。"
                f"请先用 `set_participant_location(index={idx}, place_name=..., new_nickname=...)` "
                f"覆盖这个空位，别用 add_participant 新增。"
            ),
        }, None

    # 同名守卫：如果已存在同 nickname 的参与者，说明 AI 在"修正"之前 add 定位错的那位——
    # 应该走 set_participant_location(participant_name=X) 修改，别再 add 一个同名的。
    # 之前用户 3 人场景遇到过：AI 加 Joe → 地理编码错到深圳 → AI 意识到错 → 继续 add 3 个 Joe。
    nickname_stripped = (args.get("nickname") or "").strip()
    same_name = [(i + 1, p) for i, p in enumerate(parts) if (p.get("name") or "").strip() == nickname_stripped]
    if nickname_stripped and same_name:
        idx, existing = same_name[0]
        has_loc = existing.get("lng") is not None and existing.get("lat") is not None
        loc_hint = f"当前定位『{existing.get('address', '')}』" if has_loc else "当前无定位"
        return {
            "ok": False,
    """草稿档：改搜索关键词。"""
            "error": (
                f"已有同名参与者「{nickname_stripped}」(index={idx}，{loc_hint})。"
                f"若要修改他的位置或城市，请用 `set_participant_location(index={idx}, place_name=..., city=...)`，"
                f"别再用 add_participant 加同名的。"
            ),
        }, None

    nickname = (args.get("nickname") or "").strip()
    if not nickname:
        return {"ok": False, "error": "缺少 nickname"}, None
    place_name = (args.get("place_name") or "").strip() or None
    explicit_city = (args.get("city") or "").strip() or None
    session_city = st.get("city") or "北京"
    lng = args.get("lng"); lat = args.get("lat")
    prefer = (args.get("prefer") or "auto").strip() or "auto"
    if prefer not in {"auto", "transit", "driving", "walking", "cycling"}:
        prefer = "auto"

    address = None
    if place_name and (lng is None or lat is None):
        detected = _extract_city(place_name)
        target_city = explicit_city or detected or session_city
        query = place_name if detected else f"{target_city}{place_name}"
        geo = amap_geocode(query, city=target_city)
        if not geo.get("success"):
            geo = amap_geocode(place_name, city=target_city)
        if not geo.get("success"):
            return {"ok": False, "error": geo.get("error", f"『{place_name}』无法定位")}, None
        lng = geo["lng"]; lat = geo["lat"]
        address = geo.get("formatted_address") or place_name
    elif place_name:
        address = place_name

    label = f"新增 {nickname}" + (f" @ {address}" if address else "（未填位置）")
    detail = f"{lng:.4f}, {lat:.4f}" if (lng is not None and lat is not None) else "位置待补"

    data: dict = {"nickname": nickname, "prefer": prefer}
    if lng is not None and lat is not None:
        data["lng"] = float(lng); data["lat"] = float(lat); data["address"] = address or ""

    summary = f"提议加一位参与者 {nickname}" + (f"，位置在 {address}" if address else "")
    return (
        {"ok": True, "summary": summary},
        {
            "type": "draft", "kind": "add_participant", "label": label,
            "detail": detail,
            "data": data,
        },
    )


def _tool_set_keyword(sid: str, args: dict) -> tuple[dict, dict | None]:
    kw = (args.get("keyword") or "").strip()
    if not kw:
        return {"ok": False, "error": "缺少 keyword"}, None
    st = _assistant_get_state(sid)
    old = (st.get("query") or "").strip()
    if old == kw:
        return {"ok": True, "summary": f"关键词已经是「{kw}」，不用改"}, None
    return (
        {"ok": True, "summary": f"提议关键词改为「{kw}」"},
        {
            "type": "draft", "kind": "set_keyword", "label": f"关键词 → {kw}",
            "detail": f"原：{old or '（未设置）'}",
            "data": {"keyword": kw},
        },
    )


def _tool_set_radius(sid: str, args: dict) -> tuple[dict, dict | None]:
    """草稿档：改搜索半径（米）。"""
    r = args.get("radius_m")
    try:
        r = int(r)
    except (TypeError, ValueError):
        return {"ok": False, "error": "radius_m 必须是整数（米）"}, None
    if r < 300 or r > 60000:
        return {"ok": False, "error": "半径必须在 300 ~ 60000 米之间"}, None
    st = _assistant_get_state(sid)
    old_anchor = st.get("anchor") or {}
    old_r = int(old_anchor.get("radius_m") or 5000)
    if old_r == r:
        return {"ok": True, "summary": f"半径已经是 {r} m"}, None
    def _fmt(m): return f"{m/1000:.1f} km" if m >= 1000 else f"{m} m"
    return (
        {"ok": True, "summary": f"提议半径改为 {_fmt(r)}"},
        {
            "type": "draft", "kind": "set_radius", "label": f"半径 → {_fmt(r)}",
            "detail": f"原：{_fmt(old_r)}",
            "data": {"radius_m": r},
        },
    )


def _tool_recompute_routes(sid: str, args: dict) -> tuple[dict, dict | None]:
    st = _assistant_get_state(sid)
    pois = st["pois"]
    participants = st["participants"]
    if not pois:
        return {"ok": False, "error": "当前没有 POI"}, None
    if not participants:
        return {"ok": False, "error": "当前没有参与者"}, None

    enriched: list[dict] = []
    for p in pois[:12]:
        legs = []
        for person in participants:
            r = amap_get_best_route(
                person["lng"], person["lat"],
                p["lng"], p["lat"],
                st.get("city", "北京"),
                person.get("prefer", "auto") or "auto",
                None,
            )
            legs.append(_format_route(r) | {"name": person.get("name", "?")})
        durations = [l.get("duration_minutes", 999) for l in legs]
        enriched.append({
            **p,
            "legs": legs,
            "max_time_minutes":  max(durations),
            "mean_time_minutes": round(sum(durations) / len(durations), 1),
            "time_std_minutes":  round((sum((d - sum(durations) / len(durations)) ** 2 for d in durations) / len(durations)) ** 0.5, 1),
        })
    enriched.sort(key=lambda x: (x["max_time_minutes"], -float(x.get("rating") or 0)))
    session_update(sid, {"last_pois": enriched})
    return (
        {"ok": True, "summary": f"已重算 {len(enriched)} 个 POI 的路线", "top": [
            {"name": p.get("name"), "max_time_minutes": p["max_time_minutes"]}
            for p in enriched[:5]
        ]},
        {"type": "pois_replaced", "pois": enriched, "anchor": st["anchor"]},
    )


def _tool_get_current_result(sid: str, args: dict) -> tuple[dict, dict | None]:
    st = _assistant_get_state(sid)
    top = [
        {
            "name": p.get("name"),
            "rating": p.get("rating"),
            "max_time_minutes": p.get("max_time_minutes"),
            "address": p.get("address"),
        }
        for p in (st["pois"] or [])[:6]
    ]
    return (
        {
            "ok": True,
            "anchor": st["anchor"],
            "participants": [
                {"name": p.get("name"), "lng": p.get("lng"), "lat": p.get("lat"),
                 "prefer": p.get("prefer")}
                for p in st["participants"]
            ],
            "query": st["query"],
            "pois_count": len(st["pois"] or []),
            "top": top,
        },
        None,
    )


_MEMORY_CATEGORY_LABELS = {"transport": "出行", "food": "饮食", "budget": "预算"}


def _memory_rows(device_id: str) -> list[dict]:
    conn = _db_connect()
    try:
        rows = conn.execute(
            "SELECT category, memory_key, memory_value, source, status, updated_at "
            "FROM agent_memories WHERE device_id=? ORDER BY category, updated_at DESC",
            (device_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _people_rows(device_id: str) -> list[dict]:
    now = _now(); conn = _db_connect()
    try:
        rows = conn.execute(
            "SELECT name,relation,usual_place,city,expires_at,updated_at FROM memory_people "
            "WHERE device_id=? AND (expires_at IS NULL OR expires_at>?) ORDER BY updated_at DESC LIMIT 20",
            (device_id, now),
        ).fetchall()
        return [dict(r) for r in rows]
    finally: conn.close()


def _episode_rows(device_id: str, limit: int = 8) -> list[dict]:
    conn = _db_connect()
    try:
        rows = conn.execute(
            "SELECT id,happened_at,keyword,people_json,chosen_poi_json,summary FROM memory_episodes "
            "WHERE device_id=? ORDER BY happened_at DESC LIMIT ?", (device_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally: conn.close()


def _feedback_rows(device_id: str) -> list[dict]:
    conn = _db_connect()
    try:
        rows = conn.execute(
            "SELECT poi_id,poi_name,signal,reason,updated_at FROM memory_feedback "
            "WHERE device_id=? ORDER BY updated_at DESC LIMIT 100", (device_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally: conn.close()


def _apply_feedback_ranking(device_id: str, pois: list[dict]) -> list[dict]:
    """明确不喜欢的店下沉；喜欢/收藏过的轻量加分，保留原始分便于解释。"""
    signals: dict[str, set[str]] = {}
    for row in _feedback_rows(device_id):
        signals.setdefault(row["poi_name"], set()).add(row["signal"])
    out = []
    for poi in pois:
        p = dict(poi); sig = signals.get(p.get("name") or "", set())
        base = float(p.get("_score") or 0)
        boost = (0.03 if "liked" in sig else 0) - (1.0 if "disliked" in sig else 0)
        p["_score"] = round(base + boost, 4)
        if sig: p["memory_signals"] = sorted(sig)
        out.append(p)
    out.sort(key=lambda p: p.get("_score", 0), reverse=True)
    return out


def _memory_context(device_id: str) -> str:
    rows = _memory_rows(device_id)
    people = _people_rows(device_id)
    episodes = _episode_rows(device_id, 5)
    feedback = _feedback_rows(device_id)
    items = [f"偏好:{r['category']}.{r['memory_key']}={r['memory_value']}" for r in rows]
    items += [f"人物:{p['name']}({p.get('relation') or '未注明关系'})，常用地={p.get('usual_place') or '未保存'}" for p in people]
    items += [f"经历:{e['summary']}" for e in episodes]
    items += [f"反馈:{f['poi_name']}={f['signal']}" for f in feedback[:20]]
    if not items:
        return "[长期记忆] 无。不得凭空假设用户有车、人物关系、交通、饮食或预算偏好。"
    return (
        "[已确认长期记忆] " + "; ".join(items) +
        "。人物记忆仅在用户提到同名人物时使用。这些记忆只属于当前用户，不得泄露给房间其他成员。若本轮明确表达冲突，以本轮为准；"
        "使用记忆影响规划时，在最终回复中用自然语言简短说明。"
    )


def _apply_confirmed_memory_defaults(sid: str, device_id: str) -> None:
    """仅在用户尚未选择交通方式（auto）时，把已确认的本人默认方式用于本轮。"""
    mode_map = {"公交": "transit", "骑行": "cycling", "驾车": "driving", "步行": "walking", "最快": "auto"}
    transport = next(
        (r for r in _memory_rows(device_id)
         if r["category"] == "transport" and r["memory_key"] == "default_mode"),
        None,
    )
    prefer = mode_map.get((transport or {}).get("memory_value", ""))
    if not prefer or prefer == "auto":
        return
    st = session_get(sid) or {}
    parts = [dict(p) for p in (st.get("participants") or [])]
    me_idx = _compute_me_index(parts, st.get("my_did") or "")
    if 0 < me_idx <= len(parts) and (parts[me_idx - 1].get("prefer") or "auto") == "auto":
        parts[me_idx - 1]["prefer"] = prefer
        session_update(sid, {"participants": parts})


def _memory_device_id(sid: str) -> str:
    return str((session_get(sid) or {}).get("memory_did") or (session_get(sid) or {}).get("my_did") or "").strip()


def _tool_remember_preference(sid: str, args: dict) -> tuple[dict, dict | None]:
    device_id = _memory_device_id(sid)
    category = (args.get("category") or "").strip()
    key = (args.get("key") or "").strip()[:60]
    value = (args.get("value") or "").strip()[:160]
    if category not in _MEMORY_CATEGORY_LABELS or not key or not value:
        return {"ok": False, "error": "记忆类别、键或内容无效"}, None
    # 第一版只接受显式个人偏好；阻断位置、人物等越界写入。
    blocked = ("位置", "地址", "经度", "纬度", "朋友", "同事", "家人", "住址", "公司")
    if any(x in key + value for x in blocked):
        return {"ok": False, "error": "第一版不长期保存位置或人物资料"}, None
    now = _now()
    conn = _db_connect()
    try:
        conn.execute(
            "INSERT INTO agent_memories(device_id,category,memory_key,memory_value,source,status,created_at,updated_at) "
            "VALUES(?,?,?,?, 'explicit','confirmed',?,?) "
            "ON CONFLICT(device_id,category,memory_key) DO UPDATE SET "
            "memory_value=excluded.memory_value,source='explicit',status='confirmed',updated_at=excluded.updated_at",
            (device_id, category, key, value, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "summary": f"已记住{_MEMORY_CATEGORY_LABELS[category]}偏好：{value}"}, None


def _tool_list_memories(sid: str, _args: dict) -> tuple[dict, dict | None]:
    did = _memory_device_id(sid); rows = _memory_rows(did); people = _people_rows(did); episodes = _episode_rows(did); feedback = _feedback_rows(did)
    total = len(rows)+len(people)+len(episodes)+len(feedback)
    return {"ok": True, "summary": f"共 {total} 条记忆", "preferences": rows, "people": people, "episodes": episodes, "feedback": feedback}, None


def _tool_forget_memory(sid: str, args: dict) -> tuple[dict, dict | None]:
    device_id = _memory_device_id(sid)
    category = (args.get("category") or "").strip()
    key = (args.get("key") or "").strip()
    if category not in {*_MEMORY_CATEGORY_LABELS, "all"}:
        return {"ok": False, "error": "记忆类别无效"}, None
    conn = _db_connect()
    try:
        if category == "all":
            cur = conn.execute("DELETE FROM agent_memories WHERE device_id=?", (device_id,))
            deleted = cur.rowcount
            for table in ("memory_people", "memory_episodes", "memory_feedback"):
                deleted += conn.execute(f"DELETE FROM {table} WHERE device_id=?", (device_id,)).rowcount
        elif key:
            cur = conn.execute(
                "DELETE FROM agent_memories WHERE device_id=? AND category=? AND memory_key=?",
                (device_id, category, key),
            )
        else:
            cur = conn.execute(
                "DELETE FROM agent_memories WHERE device_id=? AND category=?",
                (device_id, category),
            )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "summary": f"已忘记 {deleted if category == 'all' else cur.rowcount} 条记忆"}, None


@app.route("/api/v2/memories")
def api_v2_memories():
    did=g.device_id
    return jsonify({"preferences":_memory_rows(did),"people":_people_rows(did),"episodes":_episode_rows(did,50),"feedback":_feedback_rows(did)})


@app.route("/api/v2/memories", methods=["DELETE"])
def api_v2_memories_clear():
    conn=_db(); total=0
    for table in ("agent_memories","memory_people","memory_episodes","memory_feedback"):
        total += conn.execute(f"DELETE FROM {table} WHERE device_id=?",(g.device_id,)).rowcount
    conn.commit(); return jsonify({"ok":True,"deleted":total})


@app.route("/api/v2/memories/item", methods=["DELETE"])
def api_v2_memory_delete_item():
    data=request.json or {}; kind=(data.get("kind") or "").strip(); conn=_db()
    if kind=="preference":
        cur=conn.execute("DELETE FROM agent_memories WHERE device_id=? AND category=? AND memory_key=?",(g.device_id,data.get("category"),data.get("key")))
    elif kind=="person":
        cur=conn.execute("DELETE FROM memory_people WHERE device_id=? AND name=?",(g.device_id,data.get("name")))
    elif kind=="episode":
        cur=conn.execute("DELETE FROM memory_episodes WHERE device_id=? AND id=?",(g.device_id,data.get("id")))
    elif kind=="feedback":
        cur=conn.execute("DELETE FROM memory_feedback WHERE device_id=? AND poi_name=? AND signal=?",(g.device_id,data.get("name"),data.get("signal")))
    else: return jsonify({"error":"invalid kind"}),400
    conn.commit(); return jsonify({"ok":True,"deleted":cur.rowcount})


def _tool_remember_person(sid: str, args: dict) -> tuple[dict, dict | None]:
    did = _memory_device_id(sid)
    name = (args.get("name") or "").strip()[:60]
    if not name or name in ("我", "自己"):
        return {"ok": False, "error": "人物名字无效"}, None
    relation = (args.get("relation") or "").strip()[:60] or None
    place = (args.get("usual_place") or "").strip()[:160] or None
    city = (args.get("city") or "").strip()[:40] or None
    days = max(1, min(365, int(args.get("days") or 90)))
    now = _now(); expires = now + days * 86400 if place else None
    conn = _db_connect()
    try:
        conn.execute(
            "INSERT INTO memory_people(device_id,name,relation,usual_place,city,expires_at,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(device_id,name) DO UPDATE SET "
            "relation=COALESCE(excluded.relation,memory_people.relation),usual_place=COALESCE(excluded.usual_place,memory_people.usual_place),"
            "city=COALESCE(excluded.city,memory_people.city),expires_at=excluded.expires_at,updated_at=excluded.updated_at",
            (did,name,relation,place,city,expires,now,now),
        ); conn.commit()
    finally: conn.close()
    detail = f"，常从{place}出发（{days}天有效）" if place else ""
    return {"ok": True, "summary": f"已记住{name}是{relation or '你认识的人'}{detail}"}, None


def _tool_remember_feedback(sid: str, args: dict) -> tuple[dict, dict | None]:
    did = _memory_device_id(sid); name = (args.get("poi_name") or "").strip()[:120]
    signal = (args.get("signal") or "").strip(); reason = (args.get("reason") or "").strip()[:240] or None
    if not name or signal not in ("liked","visited","disliked"):
        return {"ok": False, "error": "店铺反馈无效"}, None
    now=_now(); conn=_db_connect()
    try:
        conn.execute(
            "INSERT INTO memory_feedback(device_id,poi_id,poi_name,signal,reason,created_at,updated_at) VALUES(?,NULL,?,?,?,?,?) "
            "ON CONFLICT(device_id,poi_name,signal) DO UPDATE SET reason=excluded.reason,updated_at=excluded.updated_at",
            (did,name,signal,reason,now,now),
        ); conn.commit()
    finally: conn.close()
    return {"ok": True, "summary": f"已记录你对{name}的反馈"}, None


def _tool_offer_choices(_sid: str, args: dict) -> tuple[dict, dict | None]:
    question=(args.get("question") or "请选择").strip()[:120]
    mode=args.get("mode") if args.get("mode") in ("single","multiple") else "single"
    # 防止模型把同一人的互斥交通方式错误标成多选。
    if mode == "multiple" and any(x in question for x in ("怎么过去", "交通方式", "出行方式")):
        mode = "single"
    options=[]
    for raw in (args.get("options") or [])[:5]:
        label=(raw.get("label") or "").strip()[:30]; value=(raw.get("value") or label).strip()[:120]
        if label and value: options.append({"label":label,"value":value})
    if len(options)<2: return {"ok":False,"error":"至少需要两个候选项"},None
    return {"ok":True,"summary":"已给出可选答案"},{"type":"choices","question":question,"mode":mode,"options":options}


TOOL_HANDLERS = {
    "offer_choices":            _tool_offer_choices,
    "remember_preference":      _tool_remember_preference,
    "list_memories":           _tool_list_memories,
    "forget_memory":           _tool_forget_memory,
    "remember_person":         _tool_remember_person,
    "remember_feedback":       _tool_remember_feedback,
    "search_pois":              _tool_search_pois,
    "shift_center":             _tool_shift_center,
    "set_participant_location": _tool_set_participant_location,
    "add_participant":          _tool_add_participant,
    "set_keyword":              _tool_set_keyword,
    "set_radius":               _tool_set_radius,
    "recompute_routes":         _tool_recompute_routes,
    "get_current_result":       _tool_get_current_result,
}


_ASSISTANT_SYSTEM = """你是「中点 Middot 会面助手」，一个**只**服务多人会面场景的专用 AI，任务是帮用户在页面上动态调整锚点、参与者、POI 与路线。

## 严格边界（第一优先级）
你**只**回答与会面直接相关的问题，具体包括：
- 换关键词 / 换餐厅类型 / 换菜系（如「换成日料」「找安静的咖啡厅」）
- 移动锚点 / 换会面中心（如「以三里屯为中心找」「往北移 2 公里」）
- 加减参与者、改出发地、改出行偏好（打车/地铁/步行）
- 重新计算路线时长、按公平度排序
- 解释当前结果为什么这么排（比如"为什么这家排第一"）

以下类型的问题**必须**礼貌拒答，并把话题引回会面：
- 通用知识问答（如 RLHF、编程、历史、翻译、写作、代码解释、数学题）
- 闲聊、玩笑、角色扮演、情感倾诉
- 让你介绍自己的模型、参数、训练方式、供应商

拒答模板（Markdown 格式）：
> 我只负责帮你调**会面地点**和**路线**，其他问题帮不上忙。
>
> 你可以试试：**"找家咖啡厅"**、**"锚点挪到国贸"**、**"加个从望京出发的人"**

## 关键约定
- **【长期记忆】**：用户明确说“记住/以后默认/以后别推荐”时，才可保存。交通、饮食和预算进入个人记忆；用户明确要求记住某个人及其常用出发地时进入关系记忆（地点默认90天有效）；明确说去过、喜欢或不喜欢某店时进入反馈记忆。“今天/这次”只用于本轮，禁止长期保存。浏览器当前位置、实时轨迹和推断出的敏感属性禁止长期保存。
- 用户问“你记得我什么”时，综合列出个人偏好、人物、近期经历和店铺反馈；说“忘掉/删除”时执行删除。人物地点只有在用户明确说“记住”时才允许保存，不能从一次规划中偷记。
- 已确认记忆只属于“我”，不得复制给朋友。本轮明确表达永远优先于旧记忆。使用长期偏好影响规划时，要在最终回复中自然说明，例如“已按你平时的公交方式规划”，但不要暴露内部字段。
- **『我 / 我自己 / 咱』= [当前会话快照] 里 `me_index` 那一位**（每轮系统会告诉你 me_index 是几）。用户说"我在北大" → `set_participant_location(index=me_index, place_name="北大")`。**永远别硬编码 index=1**——房间模式下你可能是 index=2 或更后。
- **【硬规则 · 地点先绑定语法主体】**：`X 的朋友/同事/家人` 中，地点 X 属于后面的那个人，不能因为整句话由“我想/我要/我和”开头就绑定给“我”。例如“我想和文三路的朋友吃火锅”表示**朋友在文三路、我的位置没有说**：必须把 `文三路` 用 `set_participant_location` 填到非 `me_index` 的朋友/空位，**绝不允许**写入 `me_index`。前端可能会另行征得定位许可补齐“我”，但这不改变朋友地点的归属。
- **绝不猜测用户的当前位置**：当 me_index 对应参与者的 `lng`/`lat` 为空，而用户说“当前位置”“我和某地的朋友见面”或其他隐含需要本人出发地的表达时，前端会先尝试请求浏览器定位。如果定位仍为空，明确请用户点地图上的“定位到我”或手动填写；**禁止**把用户放到北京、当前 city、IP 定位城市或任意默认坐标。
- **【硬规则 · 快照位置优先】**：本轮进入主 Agent 前，前端可能已经取得设备定位并写入 `[当前会话快照]`。只要 `me_index` 对应参与者的 `lng` 和 `lat` 非空，就代表“我”的位置**已经设置完成**，即使用户原句没有文字说明“我在哪”。此时禁止回复“你的位置还没设”、禁止再次索取位置，也不要再为“我”调用 `set_participant_location`；直接使用快照坐标继续规划。
- 用户问“我在哪 / 你能看到我在哪吗”时，读取 `me_index` 那位的 `address` 回答：地址非空就直接告诉用户页面当前显示的地址；只有 address 为空而坐标非空时，才说明目前只有坐标。快照里的 address 是地图定位和反向解析结果，复述它不属于额外猜测。
- **空位定义**：participants 里 `lng`/`lat` 为 `null` 的 slot 是**空位**（有名字，没位置）。solo 模式初始就有 2 个空位（「我 / 小伙伴」），随时可能因为用户 add 后没填位置而多出来。
- **【硬规则 · 空位优先覆盖】**：只要列表里**有任何一个空位**，用户新提到的人**必须**用 `set_participant_location` 覆盖那个空位（配合 `new_nickname` 改名），**禁止**先调 `add_participant`。
  - 例：solo 模式默认「我(1) / 小伙伴(2)」都是空位。用户说"我在 A，我朋友 Lisa 在 B" → 两次 `set_participant_location`（index=1 定 A、index=2 定 B 并 `new_nickname="Lisa"`），**不要** `add_participant`。这条几乎在 90% 场景都成立，别绕开。
  - 只有列表**位置真的不够**（如已经 3 个人都有位置，用户又提第 4 位）才 `add_participant`。
- **【硬规则 · 完事就搜】**：一轮结束时如果满足「keyword 已设 + 所有 participants 都有位置」，你**必须**再多调一次 `search_pois(keyword=当前 keyword)`，让用户立刻看到店铺结果，别让他手动点搜索。同一轮里如果已经 search 过就别再重复。
- **锚点（会面中心）默认由系统按参与者中点自动算**——只在下列情形调 `shift_center`：
  - **允许**：用户明确说"锚点挪到 X"、"定在 X"、"约在 X"、"以 X 为中心找"、"就 X 吧"（明确指定会面点）
  - **不允许**：用户只说自己/朋友在哪（"我在国贸上班"= 我的位置，用 `set_participant_location`，不要 `shift_center`）
  - **不允许**：用户说"我们在朝阳吃火锅"这种含地区+活动的模糊表达（朝阳太大，不是明确会面点，让系统按中点算）
- **『我要吃 X』/『我们想吃 X』/『找家 X』**：是搜索关键词，调 `set_keyword`。用户表达**对活动/场所类型的偏好**（吃/喝/玩/看/买/聊）都算，别只匹配"吃 X"字面。
- **房间模式下 `add_participant` 被禁用**（服务端会拒），要加人请让用户分享房间码给对方，让本人自己加入。
- **【硬规则 · 改错走 set，别再 add】**：如果发现之前 `add_participant` 或 `set_participant_location` 把某人定错了城市/位置（比如"北航"被解析到深圳而不是北京），**必须**用 `set_participant_location(participant_name="那个名字", place_name=..., city="北京")` 修改原有那位，**禁止**再 `add_participant` 加一个同名的（服务端也会拒）。同名去重是硬约束：同一个名字只能有一位。

## 工作原则
0. 当任务因缺少一个适合点击回答的信息而暂停时，优先用 `offer_choices`。**同一个人的交通方式、是否采用记忆地点、预算区间必须用 single**，绝不能让用户同时选“打车”和“开车”；忌口、氛围偏好等可并存答案用 multiple。若要分别设置多人交通，可用 multiple，但每个选项必须明确写人物与方式（如“我坐公交”“阿杰开车”）。给2～5项即可。选项的 value 必须是能直接作为用户下一句话的自然语言。调用后用一句话邀请用户选择或自行输入，本轮不要继续猜测执行。
1. 用 `get_current_result` 查看当前状态；不要盲猜。
2. 用户说「换个方向」「以 X 为中心」「再远点」「把小王的位置改到望京」「换成日料」类需求 → **优先调工具**而不是只回复文字。
3. **草稿档工具（shift_center / set_participant_location / add_participant / set_keyword / set_radius）不会直接改用户的设置**，只是把你的提议塞进一张草稿卡等用户点"应用"。所以：即使用户没明说"你去改"，只要意图明确，就大胆调；用户来把关，不会被你覆盖。
4. `search_pois` 和 `recompute_routes` 会**立即**刷新地图上的推荐列表，属于「只读式副作用」——探索场景可以自由用；如果用户改了参与者/prefer，主动 `recompute_routes`。
5. 一轮**可以并行调多个工具**——用户如果一句话里含多件事（加人 + 改多个参与者位置 + 关键词），把全部工具一起调，别一次只做一件让用户等；但同类事情别叠罗汉（不要一次改 3 遍同一个参与者的位置）。有依赖时（如先加人再改这个新人的位置），把两步合并进 `add_participant` 一次搞定，不要分两轮。
6. 跨城市地名（"杭州文三路"、"上海外滩"、"深圳南山"）→ shift_center / set_participant_location / add_participant 必须传 `city`，否则会被当作默认城市（北京）解析出错。
7. 每次最终回复用**一句到两句**中文概括完成了什么和用户接下来能做什么（如果是草稿，说“我先准备好了，你确认下”）。上方折叠区已经展示执行步骤，所以正文**禁止**复述内部过程，禁止出现函数名、工具名、参数名、索引、JSON、代码块或“我将调用/我调用了”之类实现细节。
8. 回复用 **Markdown** 格式：粗体用 `**xxx**`、列表用 `-`、代码用反引号。别用 HTML。
9. Punchy，别啰嗦。中文优先。你的名字叫「小 Mid」。
10. **不要使用 Emoji**。界面已有统一线性图标，回复只用文字和 Markdown。

## 例子

**用户输**：我在北大，我闺蜜 Lisa 在对外经贸，我俩想去吃火锅
（假设列表默认是「我(1) / 小伙伴(2)」两位，都是空位；快照里 me_index=1）
**你的解析**：
- 列表有 2 个空位 → **禁止** add_participant，两次都用 set_participant_location 覆盖
- "我在北大" → `set_participant_location(index=1, place_name="北京大学")`（me_index=1）
- "我闺蜜 Lisa 在对外经贸" → `set_participant_location(index=2, place_name="对外经济贸易大学", new_nickname="Lisa")`（覆盖空位 + 改名）
- "想去吃火锅" → `set_keyword(keyword="火锅")`
- 完事条件满足（keyword 有 + 2 人都定位置）→ **必须再多调一次** `search_pois(keyword="火锅")`
- 没说见面点 → **别调 shift_center**，中点系统自动算
**你的回复**：位置都填上了，关键词已换成火锅，也帮你搜过了。

**用户输**：我想和文三路的朋友吃火锅
**你的解析**：
- “文三路的朋友”是定中结构 → 文三路属于朋友，不属于“我”
- 用 `set_participant_location` 把非 `me_index` 的朋友/空位设为文三路
- “我”的位置没有在文字里给出，不得把文三路写给 `me_index`；设备定位由前端 Agent 征求授权后补齐
- 调 `set_keyword(keyword="火锅")`
**你的回复**：朋友的位置先设在文三路，火锅也选好了；再用你的当前位置一起找合适的店。

**用户输**：加个从望京出发的同事王小明（solo 模式，列表已有「我(在国贸) / 小伙伴(在西直门) / 张三(在望京)」都有位置，无空位）
**你的解析**：列表 3 人**都有位置**，无空位 → 允许 `add_participant(nickname="王小明", place_name="望京")`
**你的回复**：加上啦，王小明从望京出发 ✓

**用户输**：定在国贸吃火锅
**你的解析**：
- "定在国贸" = **明确会面点** → `shift_center(name="国贸")` 允许
- "吃火锅" → `set_keyword(keyword="火锅")`
**你的回复**：锚点定国贸、找火锅 ✓

**用户输**：我在国贸上班，晚上想找附近吃点
**你的解析**：
- "我在国贸上班" = **陈述我的位置**（不是会面点） → `set_participant_location(index=me_index, place_name="国贸")`
- "吃点" → `set_keyword(keyword="餐厅")` 或按上下文更具体
- **不要**调 shift_center —— 用户没说定在国贸
**你的回复**：你的位置已填为国贸，中点系统会按参与者位置自动算锚点。

**用户输**（房间模式）：帮我加个朋友张三
**你的解析**：房间模式下 `add_participant` 被禁 → 直接回复用户
**你的回复**：房间里加人得让本人自己进来。把顶栏的房间码发给张三，他打开链接输入即可。

**用户输**：再加个 Joe，从北航出发
**你的解析**（solo，列表已有「我(北大) / Lisa(人大)」都定位置，无空位、无同名）：
- `add_participant(nickname="Joe", place_name="北航", city="北京")`（跨识别地名，必须带 city）
- 假设服务端返回 Joe 定位到"深圳北航科技园"（结果错了，我读到 city="深圳"）
- **【硬规则 · 改错走 set】**：**禁止**再 `add_participant("Joe", ...)`。改用 `set_participant_location(participant_name="Joe", place_name="北京航空航天大学", city="北京")` 覆盖原有那位
**你的回复**：刚才加错到深圳了，我改成北航（北京）✓
"""


def _assistant_history(sid: str) -> list[dict]:
    s = session_get(sid)
    if not s:
        return []
    return s.setdefault("chat_history", [])


def _assistant_append_history(sid: str, msg: dict) -> None:
    s = session_get(sid)
    if not s:
        return
    hist = s.setdefault("chat_history", [])
    hist.append(msg)
    # 限制历史长度，保留最近 20 轮
    if len(hist) > 40:
        del hist[: len(hist) - 40]


@app.route("/api/v2/session/apply-drafts", methods=["POST"])
def api_v2_apply_drafts():
    """把前端草稿卡里勾选的一批草稿应用到会话。

    请求：{ session_id: str, drafts: [{ kind, data }, ...] }
    响应：{ ok: true, applied: [kind1, kind2, ...], anchor?, participants?, query? }
    """
    data = request.json or {}
    sid = data.get("session_id") or ""
    drafts = data.get("drafts") or []
    if not sid or not session_get(sid):
        return jsonify({"ok": False, "error": "会话不存在"}), 404
    if not isinstance(drafts, list) or not drafts:
        return jsonify({"ok": False, "error": "drafts 空"}), 400

    updates: dict = {}
    applied: list[str] = []
    s = session_get(sid) or {}
    parts = list(s.get("participants") or [])
    parts_dirty = False

    for d in drafts:
        if not isinstance(d, dict):
            continue
        kind = d.get("kind")
        body = d.get("data") or {}
        if kind == "shift_center":
            anchor = body.get("anchor")
            if isinstance(anchor, dict) and anchor.get("lng") is not None and anchor.get("lat") is not None:
                updates["anchor"] = anchor
                city = body.get("city")
                if city:
                    updates["city"] = city
                applied.append(kind)
        elif kind == "set_participant_location":
            pid = body.get("participant_id")
            lng = body.get("lng"); lat = body.get("lat")
            addr = body.get("address") or ""
            new_nickname = (body.get("new_nickname") or "").strip()
            if pid is not None and (
                (lng is not None and lat is not None) or new_nickname
            ):
                for p in parts:
                    if p.get("id") == pid:
                        if lng is not None and lat is not None:
                            p["lng"] = float(lng); p["lat"] = float(lat)
                            if addr: p["address"] = addr
                        if new_nickname:
                            p["name"] = new_nickname
                        parts_dirty = True
                        break
                applied.append(kind)
        elif kind == "add_participant":
            nickname = (body.get("nickname") or "").strip()
            lng = body.get("lng"); lat = body.get("lat")
            addr = body.get("address") or ""
            prefer = (body.get("prefer") or "auto").strip() or "auto"
            if nickname and len(parts) < 6:
                # 分配一个新的本地 id：solo 模式无 room- 前缀
                new_id = body.get("participant_id") or f"p_{int(time.time()*1000)}_{len(parts)}"
                new_p = {
                    "id": new_id,
                    "name": nickname,
                    "prefer": prefer,
                    "lng": float(lng) if lng is not None else None,
                    "lat": float(lat) if lat is not None else None,
                    "address": addr,
                }
                parts.append(new_p)
                parts_dirty = True
                applied.append(kind)
        elif kind == "set_keyword":
            kw = (body.get("keyword") or "").strip()
            if kw:
                updates["query"] = kw
                applied.append(kind)
        elif kind == "set_radius":
            r = body.get("radius_m")
            try:
                r = int(r)
            except (TypeError, ValueError):
                r = None
            if r and 300 <= r <= 60000:
                cur_anchor = updates.get("anchor") or s.get("anchor") or {}
                if cur_anchor:
                    new_anchor = dict(cur_anchor); new_anchor["radius_m"] = r
                    updates["anchor"] = new_anchor
                applied.append(kind)

    if parts_dirty:
        updates["participants"] = parts

    if updates:
        session_update(sid, updates)

    snap = session_get(sid) or {}
    return jsonify({
        "ok": True,
        "applied": applied,
        "anchor": snap.get("anchor"),
        "participants": snap.get("participants"),
        "query": snap.get("query"),
    })


@app.route("/api/v2/assistant/location-intent", methods=["POST"])
def api_v2_assistant_location_intent():
    """让语言模型判断本轮是否需要设备当前位置；不再由前端关键词抢跑。"""
    data = request.json or {}
    message = (data.get("message") or "").strip()[:1200]
    participants = data.get("participants") or []
    if not message:
        return jsonify({"needs_current_location": False})
    participant_hint = [
        {
            "name": str(p.get("name") or "")[:40],
            "is_me": bool(p.get("is_me")),
            "has_location": bool(p.get("has_location")),
        }
        for p in participants[:8]
    ]
    system = """你是会面应用的位置语义判定 Agent。只判断是否应请求当前设备的位置。
输出 JSON：{"needs_current_location": boolean, "reason": string}。

规则：
1. 先判断地点属于谁，中文定语不可错绑：『文三路的朋友』= 朋友在文三路，不是用户在文三路。
2. 用户明确给出自己的出发地（如『我在北大』『我从国贸出发』）时，不请求设备定位。
3. 用户明确说『用我的当前位置』『定位我』时，请求设备定位。
4. 会面规划需要用户出发地、用户本人位置仍为空、且句中只给了朋友位置时，请求设备定位。
5. 只是闲聊、修改关键词、修改朋友位置或不需要本人出发地时，不请求。
6. 不要根据『我想』『我和』就把后面的朋友地点归给用户。

示例：
- 『我想和文三路的朋友吃火锅』→ true；朋友在文三路，用户位置缺失。
- 『我在文三路，想和朋友吃火锅』→ false；用户已明确在文三路。
- 『把朋友的位置改成文三路』→ false。
- 『用我的当前位置和黄龙的朋友找中点』→ true。
只输出 JSON。"""
    try:
        completion = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({
                    "message": message,
                    "participants": participant_hint,
                }, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            stream=False,
        )
        raw = completion.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return jsonify({
            "needs_current_location": parsed.get("needs_current_location") is True,
            "reason": str(parsed.get("reason") or "")[:240],
        })
    except Exception as exc:
        app.logger.warning("[location-intent] agent failed: %s", exc)
        # 判断失败时不打断用户，也不退回关键词规则。
        return jsonify({"needs_current_location": False, "reason": "agent_unavailable"})


@app.route("/api/v2/assistant/stream", methods=["POST"])
def api_v2_assistant_stream():
    """AI 助手：SSE 流式输出 DeepSeek 回复 + 工具调用可视化。

    请求：{ session_id: str, message: str, bootstrap?: {anchor, participants, pois, query} }
    响应：SSE，event 类型见 _sse。
    """
    data = request.json or {}
    sid = data.get("session_id") or ""
    user_msg = (data.get("message") or "").strip()
    bootstrap = data.get("bootstrap") or {}

    if not user_msg:
        return jsonify({"success": False, "error": "缺少 message"}), 400
    if not DEEPSEEK_API_KEY:
        return jsonify({"success": False, "error": "DeepSeek 未配置"}), 500

    # 没有 session 时用 bootstrap 建一个（第一次开对话就发起搜索的场景）
    if not sid or not session_get(sid):
        sid = session_create({
            "anchor":       bootstrap.get("anchor"),
            "participants": bootstrap.get("participants") or [],
            "last_pois":    bootstrap.get("pois") or [],
            "query":        bootstrap.get("query", ""),
            "city":         bootstrap.get("city", "北京"),
            "chat_history": [],
        })
    else:
        # 用最新的 bootstrap 覆盖动态字段（前端能保证是最新的）
        updates = {"my_did": g.device_id}
        if "anchor" in bootstrap:       updates["anchor"] = bootstrap["anchor"]
        if "participants" in bootstrap: updates["participants"] = bootstrap["participants"]
        if "pois" in bootstrap:         updates["last_pois"] = bootstrap["pois"]
        if "query" in bootstrap:        updates["query"] = bootstrap["query"]
        session_update(sid, updates)

    caller_did = g.device_id
    # 长期记忆身份与房间里的 my_did 语义分离，并在进入 SSE worker 前固定下来。
    session_update(sid, {"memory_did": caller_did})
    _apply_confirmed_memory_defaults(sid, caller_did)

    def generate():
        yield _sse({"type": "session", "session_id": sid})
        # 一 stream = 一次原子操作：headline 用例『我在北大，Lisa在对外经贸，吃火锅』
        # 一句话要一次调 3 个工具。10s 限流只对跨 stream 生效，同 stream 内所有 write tool
        # 沿用第一次的 gate 判定（要么全放，要么全拒；不能拆成半成品）。
        stream_gate_decided = False
        stream_allow = True
        stream_gate_err = ""

        history = list(_assistant_history(sid))
        # 系统消息 + 历史 + 本次
        state = _assistant_get_state(sid)
        me_idx = _compute_me_index(state["participants"], state.get("my_did") or "")
        me_p = state["participants"][me_idx - 1] if 0 < me_idx <= len(state["participants"]) else None
        me_name = (me_p or {}).get("name") or "（未命名）"
        me_has_location = bool(
            me_p and me_p.get("lng") is not None and me_p.get("lat") is not None
        )
        in_room = bool(state.get("my_did")) and any(
            str(p.get("id") or "").startswith("room-") for p in state["participants"]
        )
        state_hint = (
            f"[当前会话快照] mode={'room' if in_room else 'solo'}  "
            f"me_index={me_idx}  me_name={me_name!r}  "
            f"me_has_location={me_has_location}  "
            f"anchor={state['anchor']}  "
            f"participants={[{'idx':i+1,'name':p.get('name'),'lng':p.get('lng'),'lat':p.get('lat'),'address':p.get('address') or ''} for i,p in enumerate(state['participants'])]}  "
            f"query={state['query']!r}  pois_count={len(state['pois'])}"
        )
        messages: list[dict] = [
            {"role": "system", "content": _ASSISTANT_SYSTEM},
            {"role": "system", "content": state_hint},
            {"role": "system", "content": _memory_context(caller_did)},
            {"role": "system", "content": (
                "运行时位置约束：设备已为‘我’提供有效经纬度。无论用户原句是否写出本人地点，"
                "都必须视为‘我’已定位；禁止声称其位置未设置、禁止要求再次定位。"
                if me_has_location else
                "运行时位置约束：快照中‘我’没有有效经纬度，不得猜测其位置。"
            )},
            *history,
            {"role": "user", "content": user_msg},
        ]
        _assistant_append_history(sid, {"role": "user", "content": user_msg})

        # 7 = 单轮最多 7 次 tool_call 循环。原来 5 太紧：
        # "add 3 人 + set keyword + auto search" 就 5 次，一旦某个 add 定位错要 set 修正就爆表。
        MAX_ITERS = 7
        try:
            for it in range(MAX_ITERS):
                stream = llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    tools=ASSISTANT_TOOLS,
                    stream=True,
                    temperature=0.4,
                )
                content_buf = ""
                # tool_call 累积（stream 里 args 是 delta，需要拼接）
                tc_buf: dict[int, dict] = {}
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_buf += delta.content
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            slot = tc_buf.setdefault(tc.index, {
                                "id": None, "name": None, "arguments": "",
                            })
                            if tc.id:                              slot["id"] = tc.id
                            if tc.function and tc.function.name:   slot["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                slot["arguments"] += tc.function.arguments

                # 无工具调用 → 结束
                if not tc_buf:
                    content_buf = _guard_assistant_location_claim(content_buf, me_has_location)
                    _assistant_append_history(sid, {"role": "assistant", "content": content_buf})
                    # 只有确定本轮没有工具调用时才把正文交给前端。
                    # 带工具调用轮次中的文字通常是模型的执行计划/函数说明，折叠步骤区已展示，正文不应重复。
                    if content_buf:
                        yield _sse({"type": "token", "delta": content_buf})
                    yield _sse({"type": "done"})
                    return

                # 有工具调用：记录 assistant 消息（带 tool_calls 结构）
                tool_calls_serialized = [
                    {
                        "id": slot["id"] or f"call_{it}_{idx}",
                        "type": "function",
                        "function": {
                            "name": slot["name"] or "",
                            "arguments": slot["arguments"] or "{}",
                        },
                    }
                    for idx, slot in tc_buf.items()
                ]
                assistant_msg = {
                    "role": "assistant",
                    "content": content_buf or None,
                    "tool_calls": tool_calls_serialized,
                }
                messages.append(assistant_msg)
                _assistant_append_history(sid, assistant_msg)

                # 逐个执行工具
                for tc in tool_calls_serialized:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    yield _sse({
                        "type": "tool_call",
                        "id": tc["id"], "name": name, "args": args,
                    })
                    handler = TOOL_HANDLERS.get(name)
                    if not handler:
                        tool_result = {"ok": False, "error": f"未知工具: {name}"}
                        state_patch = None
                    else:
                        if name in _AI_WRITE_TOOLS:
                            if not stream_gate_decided:
                                stream_allow, stream_gate_err = _ai_write_gate(caller_did, name)
                                stream_gate_decided = True
                            allowed, gate_err = stream_allow, stream_gate_err
                        else:
                            allowed, gate_err = True, ""
                        if not allowed:
                            tool_result = {"ok": False, "error": gate_err}
                            state_patch = None
                        else:
                            try:
                                tool_result, state_patch = handler(sid, args)
                            except Exception as e:
                                tool_result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                                state_patch = None
                            if tool_result.get("ok"):
                                _ai_write_touch(caller_did, name)

                    yield _sse({
                        "type": "tool_result",
                        "id": tc["id"], "name": name,
                        "ok": bool(tool_result.get("ok")),
                        "summary": tool_result.get("summary") or tool_result.get("error") or "",
                        "data": tool_result,
                    })
                    if state_patch:
                        yield _sse({"type": "state_patch", "patch": state_patch})

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                    messages.append(tool_msg)
                    _assistant_append_history(sid, tool_msg)

                # 工具轮次后模型容易只关注刚执行的工具而遗忘初始设备定位，
                # 因此在生成最终总结前重新注入不可被工具结果覆盖的位置事实。
                if me_has_location:
                    messages.append({
                        "role": "system",
                        "content": "最终回复校验：‘我’在本轮开始时已有设备定位。不得说‘你的位置没填/没设’，也不得让用户再次定位。",
                    })

                # 继续下一轮，让 LLM 基于工具结果生成回复
                continue

            # 达到最大轮数
            yield _sse({"type": "error", "msg": f"达到工具调用上限 ({MAX_ITERS})"})
            yield _sse({"type": "done"})
        except Exception as e:
            yield _sse({"type": "error", "msg": f"{type(e).__name__}: {e}"})
            yield _sse({"type": "done"})

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  智能中间点推荐系统 v2（多 Agent 架构）")
    print("=" * 60)
    print(f"  DeepSeek API Key: {'已配置' if DEEPSEEK_API_KEY else '未配置 ⚠️'}")
    print(f"  高德地图 API Key:  {'已配置' if AMAP_KEY else '未配置 ⚠️'}")
    print(f"  Session 缓存: 内存（TTL {SESSION_TTL // 3600} 小时）")
    print("=" * 60)
    print("  访问地址: http://localhost:5000")
    print("=" * 60)
    # threaded=True：每个请求在独立线程中处理，允许多用户同时访问/打开多个网页
    # 不加此参数（或设为 False）时，Flask 单线程串行处理，一个请求卡住会阻塞所有人
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
