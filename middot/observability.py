"""Health checks and Prometheus request metrics."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from flask import Flask, Response, g, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client import multiprocess


REQUESTS = Counter(
    "middot_http_requests_total",
    "HTTP requests handled by Middot",
    ("method", "endpoint", "status"),
)
LATENCY = Histogram(
    "middot_http_request_duration_seconds",
    "Middot request latency",
    ("method", "endpoint"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)


def _endpoint_name() -> str:
    return str(request.endpoint or "unmatched")[:120]


def install_observability(
    app: Flask,
    *,
    database_ping: Callable[[], bool],
    redis_ping: Callable[[], bool | None],
) -> None:
    @app.before_request
    def _metrics_start() -> None:
        g.middot_request_started = time.perf_counter()

    @app.after_request
    def _metrics_finish(response: Any) -> Any:
        endpoint = _endpoint_name()
        method = request.method
        REQUESTS.labels(method, endpoint, str(response.status_code)).inc()
        started = getattr(g, "middot_request_started", None)
        if started is not None:
            LATENCY.labels(method, endpoint).observe(time.perf_counter() - started)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.get("/health/live")
    def health_live():
        return jsonify({"ok": True, "service": "middot"})

    @app.get("/health/ready")
    def health_ready():
        checks: dict[str, bool | None] = {"database": False, "redis": None}
        try:
            checks["database"] = bool(database_ping())
        except Exception:
            checks["database"] = False
        try:
            checks["redis"] = redis_ping()
        except Exception:
            checks["redis"] = False
        required_redis = bool(str(os.getenv("MIDDOT_REDIS_URL") or "").strip())
        ready = bool(checks["database"]) and (checks["redis"] is True or not required_redis)
        return jsonify({"ok": ready, "checks": checks}), (200 if ready else 503)

    @app.get("/metrics")
    def metrics():
        if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            payload = generate_latest(registry)
        else:
            payload = generate_latest()
        return Response(payload, mimetype=CONTENT_TYPE_LATEST)
