import os
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
        if os.environ.get("WORLD_SIZE", "0") != "0":
            yield from self._distributed_latency()
        elif self.device == "cuda" and self.backend == "pytorch":
            yield from self._cuda_pytorch_latency()
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

    def _distributed_latency(self):
        import torch.distributed as dist

        dist.barrier()
        start = time.perf_counter_ns()
        yield
        dist.barrier()
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9

        LOGGER.debug(f"Tracked distributed latency: {latency:.2e}s")

        self.latencies.append(latency)

    def _cuda_pytorch_latency(self):
        import torch

        torch.cuda.synchronize()  # synchronize before starting the timer
        start = time.perf_counter_ns()
        yield
        torch.cuda.synchronize()  # synchronize before stopping the timer
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9

        LOGGER.debug(f"Tracked CUDA latency: {latency:.2e}s")

        self.latencies.append(latency)
