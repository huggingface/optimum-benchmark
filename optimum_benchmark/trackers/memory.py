import os
from logging import getLogger
from typing import List, Optional
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

import psutil
import torch

from ..env_utils import bytes_to_mega_bytes, is_nvidia_system, is_rocm_system
from ..import_utils import (
    is_py3nvml_available,
    is_pyrsmi_available,
)

if is_nvidia_system():
    if is_py3nvml_available():
        import py3nvml.py3nvml as nvml
    else:
        raise ValueError(
            "The library py3nvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
            "Please install it through `pip install py3nvml`."
        )

if is_rocm_system():
    if is_pyrsmi_available():
        # TODO: use amdsmi instead of pyrsmi
        from pyrsmi import rocml
    else:
        raise ValueError(
            "The library pyrsmi is required to run memory benchmark on ROCm-powered GPUs, but is not installed. "
            "Please install it through `pip install pyrsmi@git+https://github.com/RadeonOpenCompute/pyrsmi.git."
        )


LOGGER = getLogger("memory")


class MemoryTracker:
    def __init__(self, device: str, backend: str, device_ids: Optional[List[int]] = None):
        self.device = device
        self.backend = backend
        self.device_ids = device_ids

        self.max_memory_used: int = 0
        self.max_memory_reserved: int = 0
        self.max_memory_allocated: int = 0

        if self.device == "cuda":
            if self.device_ids is None:
                self.device_ids = infer_cuda_device_ids()

            LOGGER.info(f"Tracking CUDA devices: {self.device_ids}")

    @contextmanager
    def track(self):
        if self.device == "cuda" and self.backend == "pytorch":
            yield from self._cuda_pytorch_memory()
        elif self.device == "cuda":
            yield from self._cuda_memory()
        else:
            yield from self._cpu_memory()

    def get_max_memory_used(self):
        return bytes_to_mega_bytes(self.max_memory_used)

    def get_max_memory_reserved(self):
        return bytes_to_mega_bytes(self.max_memory_reserved)

    def get_max_memory_allocated(self):
        return bytes_to_mega_bytes(self.max_memory_allocated)

    def _cuda_pytorch_memory(self):
        torch.cuda.empty_cache()

        for device_index in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(device=device_index)
            except Exception as e:
                LOGGER.warning(f"Could not reset peak memory stats for device {device_index}: {e}")

        yield from self._cuda_memory()

        for device_index in range(torch.cuda.device_count()):
            self.max_memory_allocated += torch.cuda.max_memory_allocated(device=device_index)
            self.max_memory_reserved += torch.cuda.max_memory_reserved(device=device_index)

        LOGGER.debug(f"Pytorch max memory allocated: {self.get_max_memory_allocated()} MB")
        LOGGER.debug(f"Pytorch max memory reserved: {self.get_max_memory_reserved()} MB")

    def _cuda_memory(self):
        if is_nvidia_system() and is_py3nvml_available():
            handles = []
            nvml.nvmlInit()
            for device_index in self.device_ids:
                handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
                handles.append(handle)

            yield

            for handle in handles:
                meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                self.max_memory_used += meminfo.used
            nvml.nvmlShutdown()
            LOGGER.debug(f"PyNVML max memory used: {self.get_max_memory_used()} MB")

        elif is_rocm_system() and is_pyrsmi_available():
            rocml.smi_initialize()

            yield

            for device_index in self.device_ids:
                meminfo_used = rocml.smi_get_device_memory_used(device_index)
                self.max_memory_used += meminfo_used
            rocml.smi_shutdown()
            LOGGER.debug(f"PyRSMI max memory used: {self.get_max_memory_used()} MB")
        else:
            raise ValueError("Only NVIDIA and AMD RoCm GPUs are supported for CUDA memory tracking.")

    def _cpu_memory(self, interval: float = 0.0001):
        child_connection, parent_connection = Pipe()
        # instantiate process
        memory_process = Process(
            target=monitor_process_peak_memory,
            args=(os.getpid(), child_connection, interval),
            daemon=True,
        )
        memory_process.start()
        parent_connection.recv()

        yield

        parent_connection.send(0)
        self.max_memory_used = parent_connection.recv()
        LOGGER.debug(f"Peak memory usage: {self.get_max_memory_used()} MB")


def monitor_process_peak_memory(process_id: int, connection: Connection, interval: float):
    process = psutil.Process(process_id)
    peak_memory_usage = 0
    connection.send(0)
    stop = False

    while not stop:
        meminfo_attr = "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
        current_memory_usage = getattr(process, meminfo_attr)()[0]
        peak_memory_usage = max(peak_memory_usage, current_memory_usage)
        stop = connection.poll(interval)

    connection.send(peak_memory_usage)
    connection.close()


def infer_cuda_device_ids() -> List[int]:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None:
        cuda_device_ids = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    else:
        if is_nvidia_system() and is_py3nvml_available():
            nvml.nvmlInit()
            cuda_device_ids = list(range(nvml.nvmlDeviceGetCount()))
            nvml.nvmlShutdown()
        elif is_rocm_system() and is_pyrsmi_available():
            rocml.smi_initialize()
            cuda_device_ids = list(range(rocml.smi_get_device_count()))
            rocml.smi_shutdown()
        else:
            raise ValueError("Only NVIDIA and AMD ROCm GPUs are supported for CUDA memory tracking.")

    return cuda_device_ids
