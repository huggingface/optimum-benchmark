import time
from contextlib import contextmanager
from logging import getLogger
from typing import List

import torch

LOGGER = getLogger("latency_tracker")


class LatencyTracker:
    def __init__(self, device: torch.device, backend: str):
        self.device = device
        self.backend = backend
        self.latencies: List[float] = []

        if self.device.type == "cuda" and self.backend == "pytorch":
            # because pytorch will always see devices as 0, 1, 2, ... CUDA_VISIBLE_DEVICES doesn't matter
            self.device_ids = list(range(torch.cuda.device_count()))
            LOGGER.info(f"Tracking Pytorch CUDA devices: {self.device_ids}")

    @contextmanager
    def track(self):
        if self.device.type == "cuda" and self.backend == "pytorch":
            yield from self._cuda_latency()
        else:
            yield from self._cpu_latency()

    def get_latencies(self):
        return self.latencies

    def _cuda_latency(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for device_index in self.device_ids:
            torch.cuda.synchronize(device=device_index)
        # here must record the start event after the synchronization of all devices
        start_event.record(stream=torch.cuda.current_stream(device=self.device_ids[-1]))
        yield
        for device_index in self.device_ids:
            if device_index == self.device_ids[-1]:
                # here we must record the end event before the synchronization of the last device
                end_event.record(stream=torch.cuda.current_stream(device=self.device_ids[-1]))
            torch.cuda.synchronize(device=device_index)

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
