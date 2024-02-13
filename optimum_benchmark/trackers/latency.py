from contextlib import contextmanager
from logging import getLogger
from typing import List
import time

from ..import_utils import is_torch_distributed_available, is_torch_available

if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed

LOGGER = getLogger("latency")


class LatencyTracker:
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend

        self.latencies: List[float] = []

        # this is not in track, because this tracker is used repeatedly
        if is_torch_distributed_available() and torch.distributed.is_initialized():
            LOGGER.info("\t+ Tracking Pytorch Distributed latency")
        elif self.device == "cuda" and self.backend == "pytorch":
            LOGGER.info("\t+ Tracking Pytorch CUDA latency")
        else:
            LOGGER.info("\t+ Tracking CPU latency")

    def reset(self):
        self.latencies = []

    @contextmanager
    def track(self):
        if is_torch_distributed_available() and torch.distributed.is_initialized():
            yield from self._pytorch_distributed_latency()
        elif self.backend == "pytorch" and self.device == "cuda":
            yield from self._pytorch_cuda_latency()
        else:
            yield from self._cpu_latency()

    def _pytorch_distributed_latency(self):
        torch.distributed.barrier()  # synchronize before workload
        start = time.perf_counter_ns()
        yield
        torch.distributed.barrier()  # synchronize after workload
        end = time.perf_counter_ns()

        latency = (end - start) / 1e9
        self.latencies.append(latency)

        LOGGER.debug(f"\t+ Tracked Pytorch distributed latency: {latency:.2e}s")

    def _pytorch_cuda_latency(self):
        # Note: torch.cuda.Event is not used here,
        # there's actually no specific need to use cuda events if you're synchronizing
        # it's rather a feature that can be used to measure kernel latency without synchronizing,
        # allowing us to measure the time it takes to perform an operation without necessarily stalling the GPU.
        # An interesting use case is with cuda graphs where synchronization makes us shoot the optimization in the foot.
        # details: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
        torch.cuda.synchronize()  # synchronize before workload
        start = time.perf_counter_ns()
        yield
        torch.cuda.synchronize()  # synchronize after workload
        end = time.perf_counter_ns()

        latency = (end - start) / 1e9
        self.latencies.append(latency)

        LOGGER.debug(f"\t+ Tracked Pytorch CUDA latency: {latency:.2e}s")

    def _cpu_latency(self):
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()

        latency = (end - start) / 1e9
        self.latencies.append(latency)

        LOGGER.debug(f"\t+ Tracked CPU latency: {latency:.2e}s")

    def get_total_count(self):
        return len(self.latencies)

    def get_total_latency(self):
        return sum(self.latencies)

    def get_latencies_list(self) -> List[float]:
        return self.latencies
