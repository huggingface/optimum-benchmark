from contextlib import contextmanager
from logging import getLogger
from typing import List
import torch
import time


LOGGER = getLogger("latency_tracker")


class LatencyTracker:
    def __init__(self, backend):
        self.device = backend.device
        self.latencies: List[float] = []

    @contextmanager
    def track(self):
        if self.device.type == "cuda":
            yield from self._cuda_latency()
        else:
            yield from self._cpu_latency()

    def get_latencies(self):
        return self.latencies

    def _cuda_latency(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device=self.device)
        start_event.record(stream=torch.cuda.Stream(device=self.device))
        yield
        end_event.record(stream=torch.cuda.Stream(device=self.device))
        torch.cuda.synchronize(device=self.device)
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

class PyTorchLatencyTracker(LatencyTracker):
    def __init__(self, backend):
        super().__init__(backend)
        if backend.config.device_map:
            self.hf_device_map = backend.pretrained_model.hf_device_map
            # This logic will break if anything else than device_map="auto" is used.
            self.start_device = min(self.hf_device_map.values())
            self.end_device = max(self.hf_device_map.values())
        else:
            self.hf_device_map = None
            self.start_device = self.device
            self.end_device = self.device

    def _cuda_latency(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(device=self.device)
        start_event.record(stream=torch.cuda.Stream(device=self.start_device))
        yield
        end_event.record(stream=torch.cuda.Stream(device=self.end_device))
        torch.cuda.synchronize(device=self.device)
        latency_ms = start_event.elapsed_time(end_event)
        latency = latency_ms / 1e3

        LOGGER.debug(f"Tracked CUDA latency: {latency:.2e}s")
        self.latencies.append(latency)


latency_tracker_class_for_backend = {
    "neural_compressor": LatencyTracker,
    "onnxruntime": LatencyTracker,
    "openvino": LatencyTracker,
    "pytorch": PyTorchLatencyTracker,
}