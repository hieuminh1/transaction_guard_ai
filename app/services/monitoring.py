from collections import deque
from threading import Lock

import numpy as np


class Monitor:
    """Bounded, per-process operational statistics; no ground-truth claims."""

    def __init__(self):
        self.lock = Lock()
        self.latencies = deque(maxlen=10000)
        self.requests = self.errors = self.scored = self.flagged = 0
        self.score_bands = {"low": 0, "medium": 0, "high": 0}
        self.database_log_failures = 0

    def request(self, latency_ms: float, status: int) -> None:
        with self.lock:
            self.requests += 1
            self.errors += int(status >= 400)
            self.latencies.append(latency_ms)

    def predictions(self, predictions: list) -> None:
        with self.lock:
            self.scored += len(predictions)
            self.flagged += sum(item.is_flagged for item in predictions)
            for item in predictions:
                self.score_bands[item.risk_level] += 1

    def logging_failure(self) -> None:
        with self.lock:
            self.database_log_failures += 1

    def snapshot(self) -> dict:
        with self.lock:
            values = list(self.latencies)
            return {
                "scope": "per-process; resets on restart; latency window = last 10000 requests",
                "request_count": self.requests,
                "error_count": self.errors,
                "latency_ms_p50": float(np.percentile(values, 50)) if values else None,
                "latency_ms_p95": float(np.percentile(values, 95)) if values else None,
                "total_scored": self.scored,
                "flagged_count": self.flagged,
                "flagged_rate": self.flagged / self.scored if self.scored else 0,
                "risk_distribution": dict(self.score_bands),
                "database_log_failures": self.database_log_failures,
            }
