from multiprocessing.connection import Connection
from multiprocessing import Pipe, Process
from contextlib import contextmanager
from logging import getLogger
import psutil
import torch
import os

from optimum_benchmark.utils import bytes_to_mega_bytes


LOGGER = getLogger("memory_tracker")


class MemoryTracker:
    def __init__(self, backend):
        self.device = backend.device
        self.peak_memory: int = 0

    @contextmanager
    def track(self, interval: float = 0.01):
        if self.device.type == "cuda":
            yield from self._track_cuda_peak_memory()
        else:
            yield from self._track_cpu_peak_memory(interval)

    def get_peak_memory(self):
        return bytes_to_mega_bytes(self.peak_memory)

    def _track_cuda_peak_memory(self):
        import py3nvml.py3nvml as nvml

        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        yield
        meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
        nvml.nvmlShutdown()

        # At least for PyTorch, relying on meminfo.used is fine 
        # here as PyTorch does not deallocate its cache after running forward.
        self.peak_memory = max(self.peak_memory, meminfo.used)
        LOGGER.debug(f"Peak memory usage: {self.get_peak_memory()} MB")

    def _track_cpu_peak_memory(self, interval: float):
        child_connection, parent_connection = Pipe()
        # instantiate process
        mem_process: Process = PeakMemoryMeasureProcess(
            os.getpid(), child_connection, interval
        )
        mem_process.start()
        # wait until we get memory
        parent_connection.recv()
        yield
        # start parent connection
        parent_connection.send(0)
        # receive peak memory
        self.peak_memory = parent_connection.recv()
        LOGGER.debug(f"Peak memory usage: {self.get_peak_memory()} MB")


class PeakMemoryMeasureProcess(Process):
    def __init__(self, process_id: int, child_connection: Connection, interval: float):
        super().__init__()
        self.process_id = process_id
        self.interval = interval
        self.connection = child_connection
        self.mem_usage = 0

    def run(self):
        self.connection.send(0)
        stop = False

        while True:
            process = psutil.Process(self.process_id)
            meminfo_attr = (
                "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
            )
            memory = getattr(process, meminfo_attr)()[0]
            self.mem_usage = max(self.mem_usage, memory)

            if stop:
                break
            stop = self.connection.poll(self.interval)

        # send results to parent pipe
        self.connection.send(self.mem_usage)
        self.connection.close()


class PyTorchMemoryTracker(MemoryTracker):
    def __init__(self, backend):
        super().__init__(backend)

        if backend.config.device_map:
            self.hf_device_map = backend.pretrained_model.hf_device_map
            self.device_indexes = set(self.hf_device_map.values())
        else:
            self.device_indexes = {
                self.device.index if self.device.index is not None else 0
            }

        # This variable is used only when CUDA device is used.
        self.peak_per_device = [0 for _ in range(len(self.device_indexes))]

    def _track_cuda_peak_memory(self):
        import py3nvml.py3nvml as nvml

        nvml.nvmlInit()
        handles = []

        for device_index in self.device_indexes:
            handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
            handles.append(handle)
        yield
        for i, handle in enumerate(handles):
            meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)

            self.peak_per_device[i] = max(self.peak_per_device[i], meminfo.used)

        for i, peak_device in enumerate(self.peak_per_device):
            LOGGER.debug(f"Peak memory {i} usage: {peak_device * 1e-6} MB")

        self.peak_memory = sum(self.peak_per_device)

        nvml.nvmlShutdown()
        LOGGER.info(f"Peak memory usage: {self.get_peak_memory()} MB")


memory_tracker_class_for_backend = {
    "neural_compressor": MemoryTracker,
    "onnxruntime": MemoryTracker,
    "openvino": MemoryTracker,
    "pytorch": PyTorchMemoryTracker,
}
