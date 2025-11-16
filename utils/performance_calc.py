from collections import deque
import threading
import numpy as np

class DequeMetric:
    """Thread-safe fixed-window percentile metric store."""

    def __init__(self, max_samples=5000):
        self.values = deque(maxlen=max_samples)
        self.lock = threading.Lock()

    def record(self, ms: float):
        with self.lock:
            self.values.append(ms)

    def stats(self):
        with self.lock:
            if len(self.values) == 0:
                return {}

            arr = np.array(self.values)
            return {
                "count": len(arr),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }

    def reset(self):
        with self.lock:
            self.values.clear()


class Metrics:
    """Global singleton metrics for the entire RAG pipeline."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)

                    # Create metric buckets
                    cls._instance.embedding = DequeMetric()
                    cls._instance.retrieval = DequeMetric()
                    cls._instance.rerank = DequeMetric()
                    cls._instance.llm = DequeMetric()
                    cls._instance.total = DequeMetric()

        return cls._instance

    def stats(self):
        return {
            "embedding": self.embedding.stats(),
            "retrieval": self.retrieval.stats(),
            "rerank": self.rerank.stats(),
            "llm": self.llm.stats(),
            "total": self.total.stats(),
        }

    def reset_all(self):
        self.embedding.reset()
        self.retrieval.reset()
        self.rerank.reset()
        self.llm.reset()
        self.total.reset()
