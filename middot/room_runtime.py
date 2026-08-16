"""Redis-derived room leases and online presence.

PostgreSQL remains the source of truth.  Redis makes expiry and presence cheap
and shared across web workers; losing Redis cannot delete a locked room.
"""

from __future__ import annotations

import time
from typing import Iterable

from .state_store import redis_client


def schedule_room_expiry(code: str, expires_at: int) -> None:
    client = redis_client()
    if client is None:
        return
    ttl = max(1, int(expires_at) - int(time.time()))
    pipe = client.pipeline(transaction=False)
    pipe.set(f"middot:room:{code}:lease", str(expires_at), ex=ttl)
    pipe.zadd("middot:rooms:expiries", {code: int(expires_at)})
    pipe.execute()


def remove_room_runtime(code: str) -> None:
    client = redis_client()
    if client is None:
        return
    keys = list(client.scan_iter(match=f"middot:room:{code}:presence:*", count=100))
    pipe = client.pipeline(transaction=False)
    pipe.delete(f"middot:room:{code}:lease")
    pipe.zrem("middot:rooms:expiries", code)
    if keys:
        pipe.delete(*keys)
    pipe.execute()


def heartbeat(code: str, device_id: str, ttl_seconds: int = 15) -> None:
    client = redis_client()
    if client is not None:
        client.set(f"middot:room:{code}:presence:{device_id}", "1", ex=ttl_seconds)


def leave_presence(code: str, device_id: str) -> None:
    client = redis_client()
    if client is not None:
        client.delete(f"middot:room:{code}:presence:{device_id}")


def online_members(code: str, device_ids: Iterable[str]) -> set[str]:
    ids = list(device_ids)
    client = redis_client()
    if client is None or not ids:
        return set()
    pipe = client.pipeline(transaction=False)
    for device_id in ids:
        pipe.exists(f"middot:room:{code}:presence:{device_id}")
    return {device_id for device_id, present in zip(ids, pipe.execute()) if present}
