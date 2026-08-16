"""Shared ephemeral state for production workers.

Redis is optional in development.  In production it is required before more
than one Gunicorn worker is enabled, otherwise a session created by one worker
would be invisible to another worker.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any


class StateStoreUnavailable(RuntimeError):
    pass


class _MemorySessionStore:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create(self, data: dict[str, Any]) -> str:
        sid = str(uuid.uuid4())[:8]
        with self._lock:
            self._items[sid] = {
                "data": dict(data),
                "expires_at": time.time() + self.ttl_seconds,
            }
            self.cleanup()
        return sid

    def get(self, sid: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(sid)
            if not item or float(item["expires_at"]) <= time.time():
                self._items.pop(sid, None)
                return None
            return dict(item["data"])

    def update(self, sid: str, patch: dict[str, Any]) -> bool:
        with self._lock:
            item = self._items.get(sid)
            if not item or float(item["expires_at"]) <= time.time():
                self._items.pop(sid, None)
                return False
            item["data"].update(patch)
            item["expires_at"] = time.time() + self.ttl_seconds
            return True

    def cleanup(self) -> None:
        now = time.time()
        for sid in [key for key, value in self._items.items() if value["expires_at"] <= now]:
            self._items.pop(sid, None)

    def ping(self) -> bool:
        return True


_UPDATE_SESSION_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local current = cjson.decode(raw)
local patch = cjson.decode(ARGV[1])
for key, value in pairs(patch) do current[key] = value end
redis.call('SET', KEYS[1], cjson.encode(current), 'EX', tonumber(ARGV[2]))
return 1
"""

_RATE_LIMIT_FAILURE_LUA = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1])) end
return count
"""


class RedisSessionStore:
    def __init__(self, client: Any, ttl_seconds: int, prefix: str = "middot:session:"):
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        self._update_script = client.register_script(_UPDATE_SESSION_LUA)

    def _key(self, sid: str) -> str:
        return self.prefix + sid

    @staticmethod
    def _dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    def create(self, data: dict[str, Any]) -> str:
        for _ in range(8):
            sid = str(uuid.uuid4())[:8]
            if self.client.set(self._key(sid), self._dumps(data), ex=self.ttl_seconds, nx=True):
                return sid
        raise StateStoreUnavailable("unable to allocate a unique session id")

    def get(self, sid: str) -> dict[str, Any] | None:
        raw = self.client.get(self._key(sid))
        if raw is None:
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None

    def update(self, sid: str, patch: dict[str, Any]) -> bool:
        result = self._update_script(
            keys=[self._key(sid)], args=[self._dumps(patch), self.ttl_seconds]
        )
        return bool(result)

    def cleanup(self) -> None:
        # Redis removes expired keys itself.
        return None

    def ping(self) -> bool:
        return bool(self.client.ping())


_store: _MemorySessionStore | RedisSessionStore | None = None
_store_lock = threading.Lock()
_redis: Any | None = None
_redis_lock = threading.Lock()


def redis_client() -> Any | None:
    global _redis
    url = str(os.getenv("MIDDOT_REDIS_URL") or "").strip()
    if not url:
        return None
    if _redis is not None:
        return _redis
    with _redis_lock:
        if _redis is None:
            import redis

            _redis = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=float(os.getenv("MIDDOT_REDIS_CONNECT_TIMEOUT", "1.5")),
                socket_timeout=float(os.getenv("MIDDOT_REDIS_SOCKET_TIMEOUT", "2.0")),
                health_check_interval=30,
            )
    return _redis


def session_store(ttl_seconds: int) -> _MemorySessionStore | RedisSessionStore:
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        client = redis_client()
        if client is None:
            _store = _MemorySessionStore(ttl_seconds)
        else:
            client.ping()
            _store = RedisSessionStore(client, ttl_seconds)
        return _store


def require_shared_state_for_workers(worker_count: int) -> None:
    if worker_count > 1 and not str(os.getenv("MIDDOT_REDIS_URL") or "").strip():
        raise StateStoreUnavailable(
            "MIDDOT_REDIS_URL is required when MIDDOT_WEB_WORKERS is greater than 1"
        )


def rate_limit_failures(scope: str, identity: str) -> int | None:
    client = redis_client()
    if client is None:
        return None
    raw = client.get(f"middot:ratelimit:{scope}:{identity}")
    return int(raw or 0)


def record_rate_limit_failure(scope: str, identity: str, window_seconds: int) -> int | None:
    client = redis_client()
    if client is None:
        return None
    key = f"middot:ratelimit:{scope}:{identity}"
    return int(client.eval(_RATE_LIMIT_FAILURE_LUA, 1, key, window_seconds))


def clear_rate_limit(scope: str, identity: str) -> None:
    client = redis_client()
    if client is not None:
        client.delete(f"middot:ratelimit:{scope}:{identity}")
