import time
import statistics
import logging
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)


class LatencyBenchmarker:
    """
    Tracks and reports latency, throughput, and quality metrics over time.
    
    All times stored in milliseconds.
    Reports P50, P95, P99 percentiles for production SLA monitoring.
    
    P50 (median): typical performance
    P95: performance experienced by 95th percentile of users
    P99: near-worst-case performance, important for SLA guarantees
    """

    def __init__(self):
        self.retrieval_times_ms: List[float] = []
        self.generation_times_ms: List[float] = []
        self.e2e_times_ms: List[float] = []
        self.token_counts: List[int] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.total_queries: int = 0

    def record_retrieval(self, latency_ms: float):
        self.retrieval_times_ms.append(latency_ms)

    def record_generation(self, latency_ms: float, token_count: int = 0):
        self.generation_times_ms.append(latency_ms)
        if token_count > 0:
            self.token_counts.append(token_count)

    def record_e2e(self, latency_ms: float, cache_hit: bool = False):
        self.e2e_times_ms.append(latency_ms)
        self.total_queries += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    @staticmethod
    def _percentiles(data: List[float]) -> Dict[str, float]:
        if not data:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        s = sorted(data)
        n = len(s)
        return {
            "p50": round(s[int(n * 0.50)], 2),
            "p95": round(s[int(n * 0.95)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
            "mean": round(statistics.mean(data), 2),
            "min": round(min(data), 2),
            "max": round(max(data), 2),
            "count": n
        }

    def report(self) -> Dict[str, Any]:
        """Generate full benchmark report."""
        avg_tokens = statistics.mean(self.token_counts) if self.token_counts else 0
        avg_gen_ms = statistics.mean(self.generation_times_ms) if self.generation_times_ms else 1
        tokens_per_second = (avg_tokens / (avg_gen_ms / 1000)) if avg_gen_ms > 0 else 0

        cache_hit_rate = (
            self.cache_hits / self.total_queries if self.total_queries > 0 else 0
        )

        return {
            "retrieval_latency_ms": self._percentiles(self.retrieval_times_ms),
            "generation_latency_ms": self._percentiles(self.generation_times_ms),
            "e2e_latency_ms": self._percentiles(self.e2e_times_ms),
            "throughput": {
                "tokens_per_second": round(tokens_per_second, 2),
                "avg_tokens_per_response": round(avg_tokens, 1)
            },
            "cache": {
                "hit_rate": round(cache_hit_rate, 4),
                "total_hits": self.cache_hits,
                "total_misses": self.cache_misses,
                "total_queries": self.total_queries
            }
        }