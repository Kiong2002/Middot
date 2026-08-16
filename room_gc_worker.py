"""Close expired rooms and reconcile Redis's derived room leases."""

from __future__ import annotations

import os
import time

from app_v2 import _begin_immediate, _db_connect, _sweep_stale_rooms
from middot.room_runtime import schedule_room_expiry


POLL_SECONDS = max(5.0, float(os.getenv("MIDDOT_ROOM_GC_POLL_S", "30")))


def sweep_once() -> int:
    conn = _db_connect()
    try:
        _begin_immediate(conn)
        _sweep_stale_rooms(conn)
        active = conn.execute(
            "SELECT code,expires_at FROM rooms WHERE status='active' AND expires_at IS NOT NULL"
        ).fetchall()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    for room in active:
        try:
            schedule_room_expiry(str(room["code"]), int(room["expires_at"]))
        except Exception:
            # PostgreSQL remains authoritative; Redis will be reconciled later.
            pass
    return len(active)


def run() -> None:
    while True:
        sweep_once()
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run()
