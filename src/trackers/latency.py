from contextlib import contextmanager
from logging import getLogger
from typing import List

import torch
import time


LOGGER = getLogger("latency")


class LatencyTracker:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tracked_latencies: List[float] = []

    def track(self, duration: int = 5):
        while sum(self.tracked_latencies) < duration:
            if self.device == "cuda":
                yield from self._cuda_inference_latency()
            else:
                yield from self._cpu_inference_latency()

    def get_tracked_latencies(self):
        return self.tracked_latencies

    def _cuda_inference_latency(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record(stream=torch.cuda.current_stream())
        yield
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        latency = latency_ms / 1e3

        LOGGER.debug(f"Tracked CUDA latency: {latency:.2e}s")
        self.tracked_latencies.append(latency)

    def _cpu_inference_latency(self):
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()
        latency_ns = end - start
        latency = latency_ns / 1e9

        LOGGER.debug(f"Tracked CPU latency: {latency:.2e}s")
        self.tracked_latencies.append(latency)
