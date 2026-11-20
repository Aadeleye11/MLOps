"""Process-local measurements of requests to the prediction endpoint."""

from collections import deque
from datetime import datetime, timezone
from math import ceil
from threading import Lock
from time import perf_counter


class PredictionMetrics:
    def __init__(self):
        self.lock = Lock()
        self.requests = 0
        self.failures = 0
        self.latencies = deque(maxlen=100)
        self.recent_failures = deque(maxlen=10)

    def record(self, status, elapsed_ms):
        with self.lock:
            self.requests += 1
            if status >= 400:
                self.failures += 1
                self.recent_failures.appendleft({"time": datetime.now(timezone.utc), "status": status})
            elif 200 <= status < 300:
                self.latencies.append(elapsed_ms)

    def snapshot(self):
        with self.lock:
            samples = sorted(self.latencies)
            return {
                "requests": self.requests,
                "failures": self.failures,
                "latency_samples": len(samples),
                "latency_window": self.latencies.maxlen,
                "mean_ms": sum(samples) / len(samples) if samples else None,
                "p95_ms": samples[ceil(0.95 * len(samples)) - 1] if samples else None,
                "recent_failures": list(self.recent_failures),
            }


class PredictionTimingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/predict":
            return await self.app(scope, receive, send)

        metrics = getattr(scope["app"].state, "prediction_metrics", None)
        if metrics is None:
            return await self.app(scope, receive, send)
        start = perf_counter()
        status = None

        async def observe_response(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, observe_response)
        except Exception:
            # Starlette's outer error handler sends a 500 for an unhandled exception.
            metrics.record(status or 500, (perf_counter() - start) * 1000)
            raise
        else:
            if status is not None:
                metrics.record(status, (perf_counter() - start) * 1000)
