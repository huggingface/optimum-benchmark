import os
from functools import reduce
from logging import getLogger
from dataclasses import dataclass
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from typing import List, Optional, Literal
from multiprocessing.connection import Connection

from ..env_utils import get_cuda_device_ids, is_nvidia_system, is_rocm_system
from ..import_utils import is_pynvml_available, is_amdsmi_available, is_torch_available, torch_version

if is_nvidia_system() and is_pynvml_available():
    import pynvml

if is_rocm_system() and is_amdsmi_available():
    import amdsmi  # type: ignore

if is_torch_available():
    import torch

import psutil

LOGGER = getLogger("memory")

MEMORY_UNIT = "MB"
Memory_Unit_Literal = Literal["MB"]


@dataclass
class Memory:
    unit: Memory_Unit_Literal

    max_ram: float
    max_vram: Optional[float] = None
    max_reserved: Optional[float] = None
    max_allocated: Optional[float] = None

    def __add__(self, other: "Memory") -> "Memory":
        if self.unit != other.unit:
            raise ValueError(f"Cannot add memory with different units: {self.unit} and {other.unit}")

        max_ram = self.max_ram + other.max_ram
        max_vram = self.max_vram + other.max_vram if self.max_vram is not None else None
        max_reserved = self.max_reserved + other.max_reserved if self.max_reserved is not None else None
        max_allocated = self.max_allocated + other.max_allocated if self.max_allocated is not None else None

        return Memory(
            unit=self.unit, max_ram=max_ram, max_vram=max_vram, max_reserved=max_reserved, max_allocated=max_allocated
        )

    @staticmethod
    def aggregate(max_memories: List["Memory"]) -> "Memory":
        if len(max_memories) == 0 or all(memory is None for memory in max_memories):
            return None
        elif any(memory is None for memory in max_memories):
            raise ValueError("Some memory measurements are missing")

        return reduce(lambda x, y: x + y, max_memories)

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} max RAM memory: {self.max_ram:f} ({self.unit})")
        if self.max_vram is not None:
            LOGGER.info(f"\t\t+ {prefix} max VRAM memory: {self.max_vram:f} ({self.unit})")
        if self.max_reserved is not None:
            LOGGER.info(f"\t\t+ {prefix} max reserved memory: {self.max_reserved:f} ({self.unit})")
        if self.max_allocated is not None:
            LOGGER.info(f"\t\t+ {prefix} max allocated memory: {self.max_allocated:f} ({self.unit})")


class MemoryTracker:
    def __init__(self, device: str, backend: str, device_ids: Optional[str] = None):
        self.device = device
        self.backend = backend
        self.device_ids = device_ids

        self.max_ram_memory: float = 0
        self.max_vram_memory: float = 0
        self.max_reserved_memory: float = 0
        self.max_allocated_memory: float = 0

        LOGGER.info("\t+ Tracking RAM memory")

        if self.device == "cuda":
            if self.device_ids is None:
                LOGGER.warning("\t+ `device=cuda` but `device_ids` not provided. Using all available CUDA devices.")
                self.device_ids = get_cuda_device_ids()

            self.device_ids = list(map(int, self.device_ids.split(",")))
            LOGGER.info(f"\t+ Tracking VRAM memory of CUDA devices: {self.device_ids}")

            if self.backend == "pytorch":
                num_pytorch_devices = torch.cuda.device_count()
                if len(self.device_ids) != num_pytorch_devices:
                    raise ValueError(
                        "The number of CUDA devices and Pytorch CUDA devices must be the same. "
                        f"Got {len(self.device_ids)} and {num_pytorch_devices} respectively."
                    )
                LOGGER.info(f"\t+ Tracking Allocated/Reserved memory of {num_pytorch_devices} Pytorch CUDA devices")

    def reset(self):
        self.max_ram_memory = 0
        self.max_vram_memory = 0
        self.max_reserved_memory = 0
        self.max_allocated_memory = 0

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

        for device in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(device=device)
            except Exception as e:
                LOGGER.warning(f"\t+ Could not reset max memory stats for device {device}: {e}")

        yield from self._cuda_memory()

        self.max_allocated_memory = (
            sum(torch.cuda.max_memory_allocated(device=device) for device in range(torch.cuda.device_count())) / 1e6
        )
        self.max_reserved_memory = (
            sum(torch.cuda.max_memory_reserved(device=device) for device in range(torch.cuda.device_count())) / 1e6
        )

    def _cuda_memory(self):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_gpu_vram_memory,
            args=(os.getpid(), self.device_ids, child_connection),
            daemon=True,
        )
        memory_process.start()
        parent_connection.recv()  # wait for memory process to be ready

        yield from self._cpu_memory()

        parent_connection.send(True)
        self.max_vram_memory = parent_connection.recv()

    def _cpu_memory(self):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_cpu_ram_memory,
            args=(os.getpid(), child_connection),
            daemon=True,
        )
        memory_process.start()
        parent_connection.recv()  # wait for memory process to be ready

        yield

        parent_connection.send(True)
        self.max_ram_memory = parent_connection.recv()

    def get_max_memory(self):
        if self.device == "cuda" and self.backend == "pytorch":
            return Memory(
                unit=MEMORY_UNIT,
                max_ram=self.max_ram_memory,
                max_vram=self.max_vram_memory,
                max_reserved=self.max_reserved_memory,
                max_allocated=self.max_allocated_memory,
            )
        elif self.device == "cuda":
            return Memory(
                unit=MEMORY_UNIT,
                max_ram=self.max_ram_memory,
                max_vram=self.max_vram_memory,
            )
        else:
            return Memory(
                unit=MEMORY_UNIT,
                max_ram=self.max_ram_memory,
            )


def monitor_cpu_ram_memory(process_id: int, connection: Connection, interval: float = 0.001):
    stop = False
    max_memory = 0
    process = psutil.Process(process_id)
    connection.send(0)

    while not stop:
        meminfo_attr = "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
        current_used_memory = getattr(process, meminfo_attr)()[0]
        max_memory = max(max_memory, current_used_memory)
        stop = connection.poll(interval)

    connection.send(max_memory / 1e6)  # convert to MB
    connection.close()


def monitor_gpu_vram_memory(
    process_id: int,
    device_ids: List[int],
    connection: Connection,
    interval: float = 0.01,
):
    stop = False
    max_memory = 0
    connection.send(0)

    if is_nvidia_system():
        if not is_pynvml_available():
            raise ValueError(
                "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
            )
        pynvml.nvmlInit()
        devices_handles = [pynvml.nvmlDeviceGetHandleByIndex(device_id) for device_id in device_ids]

        while not stop:
            current_used_memory = 0
            for device_id, device_handle in zip(device_ids, devices_handles):
                try:
                    device_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
                except Exception as e:
                    LOGGER.warning(f"Could not get process list for device {device_id}: {e}")
                    continue
                for device_process in device_processes:
                    if device_process.pid == process_id or (
                        psutil.pid_exists(device_process.pid)
                        and psutil.Process(device_process.pid).parent().pid == process_id
                    ):
                        # only memory usage of the process and its children is tracked
                        current_used_memory += device_process.usedGpuMemory

            max_memory = max(max_memory, current_used_memory)
            stop = connection.poll(interval)

        pynvml.nvmlShutdown()

    elif is_rocm_system():
        if not is_amdsmi_available():
            raise ValueError(
                "The library amdsmi is required to run memory benchmark on AMD GPUs, but is not installed. "
                "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
            )
        amdsmi.amdsmi_init()
        rocm_version = torch_version().split("rocm")[-1]

        if rocm_version >= "5.7":
            devices_handles = amdsmi.amdsmi_get_processor_handles()
            while not stop:
                current_used_memory = 0
                for device_id in device_ids:
                    device_handle = devices_handles[device_id]
                    try:
                        device_process = amdsmi.amdsmi_get_gpu_process_list(device_handle)
                    except Exception as e:
                        LOGGER.warning(f"Could not get process list for device {device_id}: {e}")
                        continue

                    for process_handle in device_process:
                        try:
                            process_info = amdsmi.amdsmi_get_gpu_process_info(device_handle, process_handle)
                        except Exception as e:
                            LOGGER.warning(f"Could not get process info for process {process_handle}: {e}")
                            continue

                        if process_info["pid"] == process_id or (
                            psutil.pid_exists(process_info["pid"])
                            and psutil.Process(process_info["pid"]).parent().pid == process_id
                        ):
                            # only memory usage of the monitored process and its children is tracked
                            current_used_memory += process_info["memory_usage"]["vram_mem"]

                max_memory = max(max_memory, current_used_memory)
                stop = connection.poll(interval)
        else:
            devices_handles = amdsmi.amdsmi_get_device_handles()
            while not stop:
                current_used_memory = 0
                for device_id in device_ids:
                    device_handle = devices_handles[device_id]
                    try:
                        device_process = amdsmi.amdsmi_get_process_list(device_handle)
                    except Exception as e:
                        LOGGER.warning(f"Could not get process list for device {device_id}: {e}")
                        continue

                    for process_handle in device_process:
                        try:
                            process_info = amdsmi.amdsmi_get_process_info(device_handle, process_handle)
                        except Exception as e:
                            LOGGER.warning(f"Could not get process info for process {process_handle}: {e}")
                            continue

                        if process_info["pid"] == process_id or (
                            psutil.pid_exists(process_info["pid"])
                            and psutil.Process(process_info["pid"]).parent().pid == process_id
                        ):
                            # only memory usage of the monitored process and its children is tracked
                            current_used_memory += process_info["memory_usage"]["vram_mem"]

                max_memory = max(max_memory, current_used_memory)
                stop = connection.poll(interval)

        amdsmi.amdsmi_shut_down()
    else:
        raise ValueError("Only NVIDIA and AMD ROCm GPUs are supported for CUDA memory tracking.")

    connection.send(max_memory / 1e6)  # convert to MB
    connection.close()
