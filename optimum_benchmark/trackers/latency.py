import time
from contextlib import contextmanager
from logging import getLogger
from typing import List

import torch

from ..import_utils import is_torch_distributed_available

if is_torch_distributed_available():
    import torch.distributed

LOGGER = getLogger("latency")


class LatencyTracker:
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend

        self.latencies: List[float] = []

        if is_torch_distributed_available() and torch.distributed.is_initialized():
            LOGGER.info("Tracking Pytorch Distributed latency")
        elif self.device == "cuda" and self.backend == "pytorch":
            LOGGER.info("Tracking Pytorch CUDA latency")
        else:
            LOGGER.info("Tracking CPU latency")

    @contextmanager
    def track(self):
        if is_torch_distributed_available() and torch.distributed.is_initialized():
            yield from self._pytorch_distributed_tracker()
        elif self.backend == "pytorch" and self.device == "cuda":
            yield from self._pytorch_cuda_tracker()
        else:
            yield from self._cpu_tracker()

    def _pytorch_distributed_tracker(self):
        torch.distributed.barrier()  # synchronize before workload
        start = time.perf_counter_ns()
        yield
        torch.distributed.barrier()  # synchronize after workload
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9
        self.latencies.append(latency)

        LOGGER.debug(f"Tracked Pytorch Distributed latency: {latency:.2e}s")

    def _pytorch_cuda_tracker(self):
        torch.cuda.synchronize()  # synchronize before workload
        start = time.perf_counter_ns()
        yield
        torch.cuda.synchronize()  # synchronize after workload
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9
        self.latencies.append(latency)

        LOGGER.debug(f"Tracked Pytorch CUDA latency: {latency:.2e}s")

    def _cpu_tracker(self):
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9
        self.latencies.append(latency)

        LOGGER.debug(f"Tracked CPU latency: {latency:.2e}s")

    def get_latencies(self):
        return self.latencies

    def get_total_latency(self):
        return sum(self.latencies)
