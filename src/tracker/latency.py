from contextlib import contextmanager
from logging import getLogger
from typing import List

import torch
import time


LOGGER = getLogger("latency_tracker")


class LatencyTracker:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.latencies: List[float] = []

    @contextmanager
    def track(self):
        if self.device == "cuda":
            yield from self._cuda_latency()
        else:
            yield from self._cpu_latency()

    def get_latencies(self):
        return self.latencies

    def _cuda_latency(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device=torch.device(self.device))
        start_event.record(stream=torch.cuda.Stream(device=self.device))
        yield
        end_event.record(stream=torch.cuda.Stream(device=self.device))
        torch.cuda.synchronize(device=torch.device(self.device))
        latency_ms = start_event.elapsed_time(end_event)
        latency = latency_ms / 1e3

        LOGGER.debug(f"Tracked CUDA latency: {latency:.2e}s")
        self.latencies.append(latency)

    def _cpu_latency(self):
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()
        latency_ns = end - start
        latency = latency_ns / 1e9

        LOGGER.debug(f"Tracked CPU latency: {latency:.2e}s")
        self.latencies.append(latency)
