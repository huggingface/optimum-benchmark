import time
from contextlib import contextmanager
from logging import getLogger
from typing import List

from ..import_utils import (
    is_torch_available,
    is_torch_distributed_available,
)

if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed

LOGGER = getLogger("latency")


class LatencyTracker:
    def __init__(self, backend: str, device: str):
        self.device = device
        self.backend = backend
        self.latencies: List[float] = []

        if is_torch_distributed_available() and torch.distributed.is_initialized():
            LOGGER.debug("Tracking distributed latency")
            self.tracker = "distributed"
        elif is_torch_available() and self.backend == "pytorch" and self.device == "cuda":
            LOGGER.debug("Tracking PyTorch CUDA latency")
            self.tracker = "cuda_pytorch"
        else:
            LOGGER.debug("Tracking latency")
            self.tracker = "default"

    @contextmanager
    def track(self):
        if self.tracker == "distributed":
            yield from self._distributed_latency()
        elif self.tracker == "cuda_pytorch":
            yield from self._cuda_pytorch_latency()
        else:
            yield from self._latency()

    def get_latencies(self):
        return self.latencies

    def get_total_latency(self):
        return sum(self.latencies)

    def _latency(self):
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9

        LOGGER.debug(f"Tracked CPU latency: {latency:.2e}s")
        self.latencies.append(latency)

    def _distributed_latency(self):
        torch.distributed.barrier()  # synchronize before starting the timer
        start = time.perf_counter_ns()
        yield
        torch.distributed.barrier()  # synchronize before starting the timer
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9

        LOGGER.debug(f"Tracked distributed latency: {latency:.2e}s")

        self.latencies.append(latency)

    def _cuda_pytorch_latency(self):
        torch.cuda.synchronize()  # synchronize before starting the timer
        start = time.perf_counter_ns()
        yield
        torch.cuda.synchronize()  # synchronize before stopping the timer
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9

        LOGGER.debug(f"Tracked CUDA latency: {latency:.2e}s")

        self.latencies.append(latency)
