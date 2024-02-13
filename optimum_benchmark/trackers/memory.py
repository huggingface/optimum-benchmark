import os
from logging import getLogger
from contextlib import contextmanager
from typing import List, Optional, Dict
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from ..env_utils import bytes_to_mega_bytes, get_cuda_device_ids, is_nvidia_system, is_rocm_system
from ..import_utils import is_py3nvml_available, is_pyrsmi_available, is_torch_available

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
        from pyrsmi import rocml
    else:
        raise ValueError(
            "The library pyrsmi is required to run memory benchmark on AMD GPUs, but is not installed. "
            "Please install it through `pip install pyrsmi@git+https://github.com/RadeonOpenCompute/pyrsmi.git."
        )

if is_torch_available():
    import torch

import psutil


LOGGER = getLogger("memory")


class MemoryTracker:
    """
    Memory tracker to measure max memory usage of CPU or GPU devices.

    Args:
        device (str): Device to track memory usage. Can be either "cuda" or any other device.
        backend (str): Backend to track memory usage. Can be either "pytorch" or any other backend.
        device_ids (List[int], optional): List of device IDs to track memory usage. Defaults to None.
    """

    def __init__(self, device: str, backend: str, device_ids: Optional[str] = None):
        self.device = device
        self.backend = backend

        self.max_memory_used = 0
        self.max_memory_reserved = 0
        self.max_memory_allocated = 0

        if self.device == "cuda":
            if device_ids is None:
                LOGGER.warning("\t+ `device=cuda` but `device_ids` not provided. Using all available CUDA devices.")
                self.device_ids = list(map(int, get_cuda_device_ids().split(",")))
            else:
                self.device_ids = list(map(int, device_ids.split(",")))

            LOGGER.info(f"\t+ Tracking VRAM memory of CUDA devices: {self.device_ids}")

            if self.backend == "pytorch":
                self.pytorch_device_ids = list(range(torch.cuda.device_count()))
                LOGGER.info(f"\t+ Tracking Pytorch memory of Pytorch CUDA devices: {self.pytorch_device_ids}")

                if len(self.device_ids) != len(self.pytorch_device_ids):
                    raise ValueError(
                        "The number of CUDA devices and Pytorch CUDA devices must be the same. "
                        f"Got {len(self.device_ids)} and {len(self.pytorch_device_ids)} respectively."
                    )
        else:
            LOGGER.info("\t+ Tracking RAM memory")

    def reset(self):
        self.max_memory_used = 0
        self.max_memory_reserved = 0
        self.max_memory_allocated = 0

    @contextmanager
    def track(self):
        if self.device == "cuda" and self.backend == "pytorch":
            yield from self._cuda_pytorch_memory()
        elif self.device == "cuda":
            yield from self._cuda_memory()
        else:
            yield from self._cpu_memory()

    def _cuda_pytorch_memory(self):
        torch.cuda.empty_cache()
        for pytorch_device_index in self.pytorch_device_ids:
            try:
                torch.cuda.reset_peak_memory_stats(device=pytorch_device_index)
            except Exception as e:
                LOGGER.warning(f"\t+ Could not reset max memory stats for device {pytorch_device_index}: {e}")

        yield from self._cuda_memory()

        for pytorch_device_index in self.pytorch_device_ids:
            self.max_memory_reserved += torch.cuda.max_memory_reserved(device=pytorch_device_index)
            self.max_memory_allocated += torch.cuda.max_memory_allocated(device=pytorch_device_index)

        LOGGER.debug(f"\t+ Pytorch max memory reserved: {self.get_max_memory_reserved_mb()} MB")
        LOGGER.debug(f"\t+ Pytorch max memory allocated: {self.get_max_memory_allocated_mb()} MB")

    def _cuda_memory(self, interval: float = 0.001):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_gpu_max_vram_memory,
            args=(self.device_ids, child_connection, interval),
            daemon=True,
        )
        memory_process.start()
        parent_connection.recv()  # wait for memory process to be ready

        yield

        parent_connection.send(True)
        self.max_memory_used = parent_connection.recv()
        LOGGER.debug(f"\t+ Max memory (VRAM) used: {self.get_max_memory_used_mb()} MB")

    def _cpu_memory(self, interval: float = 0.001):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_cpu_max_ram_memory,
            args=(os.getpid(), child_connection, interval),
            daemon=True,
        )
        memory_process.start()
        parent_connection.recv()  # wait for memory process to be ready

        yield

        parent_connection.send(True)
        self.max_memory_used = parent_connection.recv()
        LOGGER.debug(f"\t+ Max memory (RAM) used: {self.get_max_memory_used_mb()} MB")

    def get_max_memory_used_mb(self) -> int:
        return bytes_to_mega_bytes(self.max_memory_used)

    def get_max_memory_allocated_mb(self) -> int:
        return bytes_to_mega_bytes(self.max_memory_allocated)

    def get_max_memory_reserved_mb(self) -> int:
        return bytes_to_mega_bytes(self.max_memory_reserved)

    def get_memories_dict(self) -> Dict[str, int]:
        if self.device == "cuda" and self.backend == "pytorch":
            return {
                "max_vram_used(MB)": self.get_max_memory_used_mb(),
                "max_memory_reserved(MB)": self.get_max_memory_reserved_mb(),
                "max_memory_allocated(MB)": self.get_max_memory_allocated_mb(),
            }
        elif self.device == "cuda":
            return {"max_vram_used(MB)": self.get_max_memory_used_mb()}
        else:
            return {"max_ram_used(MB)": self.get_max_memory_used_mb()}


def monitor_cpu_max_ram_memory(process_id: int, connection: Connection, interval: float):
    process = psutil.Process(process_id)
    max_memory_usage = 0
    connection.send(0)
    stop = False

    while not stop:
        meminfo_attr = "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
        current_memory_usage = getattr(process, meminfo_attr)()[0]
        max_memory_usage = max(max_memory_usage, current_memory_usage)
        stop = connection.poll(interval)

    connection.send(max_memory_usage)
    connection.close()


def monitor_gpu_max_vram_memory(device_ids: List[int], connection: Connection, interval: float):
    if is_nvidia_system() and is_py3nvml_available():
        nvml.nvmlInit()
        handles = [nvml.nvmlDeviceGetHandleByIndex(device_id) for device_id in device_ids]
        max_memory_usage = 0
        connection.send(0)
        stop = False

        while not stop:
            current_memory_usage = sum(nvml.nvmlDeviceGetMemoryInfo(handle).used for handle in handles)
            max_memory_usage = max(max_memory_usage, current_memory_usage)
            stop = connection.poll(interval)

        connection.send(max_memory_usage)
        nvml.nvmlShutdown()
        connection.close()
    elif is_rocm_system() and is_pyrsmi_available():
        rocml.smi_initialize()
        max_memory_usage = 0
        connection.send(0)
        stop = False

        while not stop:
            current_memory_usage = sum(rocml.smi_get_device_memory_used(device_id) for device_id in device_ids)
            max_memory_usage = max(max_memory_usage, current_memory_usage)
            stop = connection.poll(interval)

        connection.send(max_memory_usage)
        rocml.smi_shutdown()
        connection.close()
    else:
        raise ValueError("Only NVIDIA and AMD ROCm GPUs are supported for CUDA memory tracking.")
