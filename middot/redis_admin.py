"""Read-only, privacy-aware Redis inspection for the Middot admin console."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


REDIS_CATEGORIES = {
    "all": ("全部 Middot 数据", "middot:*"),
    "sessions": ("Agent 会话", "middot:session:*"),
    "room_leases": ("房间租约", "middot:room:*:lease"),
    "room_presence": ("房间在线状态", "middot:room:*:presence:*"),
    "room_expiries": ("房间过期索引", "middot:rooms:expiries"),
    "rate_limits": ("访问限流", "middot:ratelimit:*"),
}

_SECRET_FIELD = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|authorization|cookie|access[_-]?token|"
    r"refresh[_-]?token|choice[_-]?token|resume[_-]?token|private[_-]?key)$",
    re.IGNORECASE,
)
_IDENTIFIER_FIELD = re.compile(
    r"(?:^id$|_id$)|^(?:my_did|memory_did|did|sid)$",
    re.IGNORECASE,
)
_TEXT_FIELD = re.compile(
    r"(?:^|_)(?:content|message|prompt|visible_content|current_user_message)$",
    re.IGNORECASE,
)


def classify_key(key: str) -> str:
    if key == "middot:rooms:expiries":
        return "room_expiries"
    if key.startswith("middot:session:"):
        return "sessions"
    if re.fullmatch(r"middot:room:[^:]+:presence:[^:]+", key):
        return "room_presence"
    if re.fullmatch(r"middot:room:[^:]+:lease", key):
        return "room_leases"
    if key.startswith("middot:ratelimit:"):
        return "rate_limits"
    return "other"


def _short_identifier(value: Any, keep: int = 4) -> str:
    raw = str(value or "")
    if not raw:
        return raw
    if len(raw) <= keep:
        return "•" * len(raw)
    return raw[:keep] + "…" + raw[-2:]


def mask_redis_key(key: str) -> str:
    parts = key.split(":")
    if len(parts) >= 3 and parts[1] == "session":
        parts[2] = _short_identifier(parts[2])
    elif len(parts) >= 3 and parts[1] == "room" and parts[2] != "expiries":
        parts[2] = _short_identifier(parts[2], keep=2)
        if "presence" in parts and len(parts) >= 5:
            parts[4] = _short_identifier(parts[4])
    elif len(parts) >= 4 and parts[1] == "ratelimit":
        identity = parts[-1]
        if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", identity):
            octets = identity.split(".")
            parts[-1] = ".".join(octets[:2] + ["×", "×"])
        else:
            parts[-1] = _short_identifier(identity)
    return ":".join(parts)


def _redact_scalar(field: str, value: Any) -> Any:
    if _SECRET_FIELD.search(field):
        return "[已隐藏]"
    if _IDENTIFIER_FIELD.search(field):
        return _short_identifier(value)
    if field.lower() in {"lat", "latitude", "lng", "lon", "longitude"}:
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return value
    if field.lower() in {"location", "coordinate", "coordinates", "coord"} and isinstance(value, str):
        if re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*[,，]\s*-?\d+(?:\.\d+)?\s*", value):
            return "[精确坐标已隐藏]"
    if field.lower() in {"address", "formatted_address", "exact_address"} and value:
        return "[精确地址已隐藏]"
    if field.lower() in {"name", "nickname", "owner_name"} and isinstance(value, str):
        return value[:1] + "**" if value else value
    if _TEXT_FIELD.search(field) and isinstance(value, str):
        return f"[文本 {len(value)} 字]"
    if isinstance(value, str) and len(value) > 240:
        return value[:240] + f"…（共 {len(value)} 字）"
    return value


def redact_value(value: Any, field: str = "", depth: int = 0) -> Any:
    """Bound previews and hide secrets, identifiers, exact locations and chat text."""
    if depth >= 6:
        return "[结构过深，已省略]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        items = list(value.items())
        for key, child in items[:60]:
            key_text = str(key)
            result[key_text] = redact_value(child, key_text, depth + 1)
        if len(items) > 60:
            result["__more__"] = f"另有 {len(items) - 60} 个字段"
        return result
    if isinstance(value, list):
        result = [redact_value(child, field, depth + 1) for child in value[:24]]
        if len(value) > 24:
            result.append(f"[另有 {len(value) - 24} 项]")
        return result
    return _redact_scalar(field, value)


def _decode_json_preview(raw: str, truncated: bool) -> Any:
    if not truncated:
        try:
            return redact_value(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass
    if len(raw) > 500:
        raw = raw[:500] + "…"
    return {"text_preview": raw, "truncated": truncated}


def _preview_for_key(client: Any, key: str, redis_type: str) -> Any:
    try:
        if redis_type == "string":
            length = int(client.strlen(key))
            raw = client.getrange(key, 0, 16383)
            return _decode_json_preview(str(raw or ""), length > 16384)
        if redis_type == "zset":
            values = client.zrange(key, 0, 29, withscores=True)
            return [
                {"member": _short_identifier(member, keep=2), "score": score}
                for member, score in values
            ]
        if redis_type == "hash":
            values = client.hscan(key, cursor=0, count=30)[1]
            return redact_value(values)
        if redis_type == "list":
            return redact_value(client.lrange(key, 0, 29))
        if redis_type == "set":
            return redact_value(list(client.sscan(key, cursor=0, count=30)[1]))
        if redis_type == "stream":
            return redact_value(client.xrange(key, count=20))
    except Exception:
        return {"notice": "读取预览时该 Key 已变化或暂不可读"}
    return {"notice": f"暂不预览 {redis_type} 类型"}


def _count_middot_keys(client: Any, maximum: int = 50000) -> tuple[Counter, bool]:
    counts: Counter = Counter()
    cursor = 0
    seen = 0
    complete = True
    while True:
        cursor, keys = client.scan(cursor=cursor, match="middot:*", count=500)
        for key in keys:
            counts[classify_key(str(key))] += 1
            seen += 1
            if seen >= maximum:
                complete = False
                break
        if not complete or int(cursor) == 0:
            break
    counts["all"] = seen
    return counts, complete


def redis_overview(client: Any) -> dict[str, Any]:
    memory = client.info("memory")
    clients = client.info("clients")
    stats = client.info("stats")
    server = client.info("server")
    keyspace = client.info("keyspace")
    counts, complete = _count_middot_keys(client)
    hits = int(stats.get("keyspace_hits") or 0)
    misses = int(stats.get("keyspace_misses") or 0)
    total_lookups = hits + misses
    return {
        "available": True,
        "redis_version": str(server.get("redis_version") or ""),
        "uptime_seconds": int(server.get("uptime_in_seconds") or 0),
        "used_memory_bytes": int(memory.get("used_memory") or 0),
        "used_memory_human": str(memory.get("used_memory_human") or ""),
        "connected_clients": int(clients.get("connected_clients") or 0),
        "blocked_clients": int(clients.get("blocked_clients") or 0),
        "total_keys": int(client.dbsize()),
        "middot_keys": int(counts.get("all", 0)),
        "hit_rate": round(hits / total_lookups, 4) if total_lookups else None,
        "operations_per_second": int(stats.get("instantaneous_ops_per_sec") or 0),
        "expired_keys": int(stats.get("expired_keys") or 0),
        "evicted_keys": int(stats.get("evicted_keys") or 0),
        "keyspace": keyspace,
        "category_counts": {
            key: int(counts.get(key, 0))
            for key in (*REDIS_CATEGORIES.keys(), "other")
        },
        "count_complete": complete,
        "categories": [
            {"id": key, "label": label} for key, (label, _pattern) in REDIS_CATEGORIES.items()
        ],
    }


def redis_key_page(
    client: Any, category: str = "all", cursor: int = 0, limit: int = 40
) -> dict[str, Any]:
    category = category if category in REDIS_CATEGORIES else "all"
    limit = min(80, max(1, int(limit)))
    pattern = REDIS_CATEGORIES[category][1]
    next_cursor, keys = client.scan(cursor=max(0, int(cursor)), match=pattern, count=limit)
    keys = [str(key) for key in keys[:80]]

    pipe = client.pipeline(transaction=False)
    for key in keys:
        pipe.type(key)
        pipe.pttl(key)
        pipe.memory_usage(key)
    metadata = pipe.execute() if keys else []

    items = []
    for index, key in enumerate(keys):
        redis_type = str(metadata[index * 3] or "none")
        if redis_type == "none":
            continue
        ttl_ms = int(metadata[index * 3 + 1])
        memory_bytes = metadata[index * 3 + 2]
        items.append({
            "key": mask_redis_key(key),
            "category": classify_key(key),
            "type": redis_type,
            "ttl_ms": ttl_ms,
            "memory_bytes": int(memory_bytes or 0),
            "preview": _preview_for_key(client, key, redis_type),
        })
    return {
        "category": category,
        "cursor": str(next_cursor),
        "items": items,
        "limit": limit,
    }
