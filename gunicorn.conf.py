import os

from middot.state_store import require_shared_state_for_workers


workers = int(os.getenv("MIDDOT_WEB_WORKERS", "2"))
threads = int(os.getenv("MIDDOT_WEB_THREADS", "8"))
require_shared_state_for_workers(workers)

bind = os.getenv("MIDDOT_BIND", "127.0.0.1:5000")
worker_class = "gthread"
timeout = int(os.getenv("MIDDOT_REQUEST_TIMEOUT", "180"))
graceful_timeout = 30
keepalive = 5
max_requests = 1200
max_requests_jitter = 120
accesslog = "-"
errorlog = "-"
capture_output = True


def child_exit(server, worker):
    del server
    from prometheus_client import multiprocess

    multiprocess.mark_process_dead(worker.pid)
