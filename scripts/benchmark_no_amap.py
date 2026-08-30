"""Small, side-effect-limited throughput benchmark that never calls AMap.

The allowlist intentionally contains only web/process health and read endpoints.
Authenticated API requests still refresh one benchmark device's last_seen timestamp,
so they exercise the real PostgreSQL middleware without creating one device per call.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


ALLOWED_PATHS = {
    "/health/live",
    "/health/ready",
    "/api/me",
    "/api/v2/conversations?limit=10",
}
PROFILES = (
    ("/health/live", 1000, 16, False),
    ("/health/ready", 400, 16, False),
    ("/api/me", 300, 8, True),
    ("/api/v2/conversations?limit=10", 300, 8, True),
)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class RequestResult:
    status: int
    duration_ms: float
    error: str = ""


class Benchmark:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.local = threading.local()
        self.cookies: dict[str, str] = {}

    def prepare_identity(self) -> None:
        response = requests.get(urljoin(self.base_url, "api/me"), timeout=self.timeout)
        response.raise_for_status()
        self.cookies = response.cookies.get_dict()
        if not self.cookies:
            raise RuntimeError("/api/me did not return the benchmark device cookie")

    def _session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.cookies.update(self.cookies)
            self.local.session = session
        return session

    def request(self, path: str) -> RequestResult:
        started = time.perf_counter()
        try:
            response = self._session().get(urljoin(self.base_url, path.lstrip("/")), timeout=self.timeout)
            return RequestResult(response.status_code, (time.perf_counter() - started) * 1000)
        except Exception as exc:  # benchmark reports failures instead of hiding them
            return RequestResult(0, (time.perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}")

    def run(self, path: str, requests_count: int, concurrency: int) -> dict[str, Any]:
        if path not in ALLOWED_PATHS:
            raise ValueError(f"path is not in the no-AMap allowlist: {path}")
        for _ in range(min(10, requests_count)):
            self.request(path)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda _index: self.request(path), range(requests_count)))
        wall_seconds = time.perf_counter() - started
        latencies = [item.duration_ms for item in results]
        status_counts: dict[str, int] = {}
        for item in results:
            key = str(item.status)
            status_counts[key] = status_counts.get(key, 0) + 1
        successful = sum(200 <= item.status < 400 for item in results)
        return {
            "path": path,
            "requests": requests_count,
            "concurrency": concurrency,
            "successful": successful,
            "errors": requests_count - successful,
            "status_counts": status_counts,
            "throughput_rps": round(requests_count / wall_seconds, 2),
            "wall_seconds": round(wall_seconds, 3),
            "latency_ms": {
                "p50": round(_percentile(latencies, 0.50), 2),
                "p95": round(_percentile(latencies, 0.95), 2),
                "p99": round(_percentile(latencies, 0.99), 2),
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    host = (urlparse(args.base_url).hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("This benchmark is intentionally limited to the local server")
    benchmark = Benchmark(args.base_url, args.timeout)
    benchmark.prepare_identity()
    reports = [benchmark.run(path, count, concurrency) for path, count, concurrency, _auth in PROFILES]
    output = {"amap_called": False, "base_url": args.base_url, "reports": reports}
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        for item in reports:
            print(
                f"{item['path']}: {item['throughput_rps']} req/s, "
                f"p50={item['latency_ms']['p50']}ms, p95={item['latency_ms']['p95']}ms, "
                f"errors={item['errors']}"
            )
    return 0 if all(item["errors"] == 0 for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())

