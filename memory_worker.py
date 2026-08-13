"""Middot 记忆整理 Worker。

与 Web 进程共享 middot.db；任务领取和结果提交只持有短事务，LLM 调用期间
不占用 SQLite 写锁。建议由 systemd/supervisor 作为独立进程常驻。
"""

import os
import socket
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from app_v2 import _memory_enqueue_nightly_catchup, memory_worker_heartbeat, memory_worker_once


POLL_SECONDS = max(1.0, float(os.getenv("MIDDOT_MEMORY_WORKER_POLL_S", "2")))
NIGHTLY_HOUR = min(23, max(0, int(os.getenv("MIDDOT_MEMORY_NIGHTLY_HOUR", "3"))))
MEMORY_TIMEZONE = ZoneInfo(os.getenv("MIDDOT_MEMORY_TIMEZONE", "Asia/Shanghai"))


def run() -> None:
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    started_at = int(time.time())
    last_heartbeat = 0.0
    last_nightly_day = ""
    while True:
        now = time.time()
        if now - last_heartbeat >= 10:
            memory_worker_heartbeat(worker_id, os.getpid(), started_at)
            last_heartbeat = now
        local = datetime.now(MEMORY_TIMEZONE)
        day_key = local.strftime("%Y-%m-%d")
        if local.hour == NIGHTLY_HOUR and day_key != last_nightly_day:
            _memory_enqueue_nightly_catchup()
            last_nightly_day = day_key
        result = memory_worker_once(worker_id)
        if result is not None:
            memory_worker_heartbeat(worker_id, os.getpid(), started_at, result)
            last_heartbeat = time.time()
        else:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run()
