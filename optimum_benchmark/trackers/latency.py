import time
from contextlib import contextmanager
from logging import getLogger
from typing import List

LOGGER = getLogger("latency")


class LatencyTracker:
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend
        self.latencies: List[float] = []

    @contextmanager
    def track(self):
        if self.device == "cuda" and self.backend == "pytorch":
            yield from self._cuda_latency()
        else:
            yield from self._cpu_latency()

    def get_latencies(self):
        return self.latencies

    def _cpu_latency(self):
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()
        latency_ns = end - start
        latency = latency_ns / 1e9

        LOGGER.debug(f"Tracked CPU latency: {latency:.2e}s")
        self.latencies.append(latency)

    def _cuda_latency(self):
        import torch.cuda

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        torch.cuda.synchronize()
        yield
        torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        latency = latency_ms / 1e3

        LOGGER.debug(f"Tracked CUDA latency: {latency:.2e}s")
        self.latencies.append(latency)
