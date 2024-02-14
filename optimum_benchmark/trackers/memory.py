import os
from functools import reduce
from logging import getLogger
from dataclasses import dataclass
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from typing import List, Optional, Dict, Literal
from multiprocessing.connection import Connection

from ..env_utils import bytes_to_mega_bytes, get_cuda_device_ids, is_nvidia_system, is_rocm_system
from ..import_utils import is_pynvml_available, is_amdsmi_available, is_torch_available
from .utils import compute_max

if is_nvidia_system() and is_pynvml_available():
    import pynvml

if is_rocm_system() and is_amdsmi_available():
    import amdsmi  # type: ignore

if is_torch_available():
    import torch

import psutil

Memory_Unit_Literal = Literal["MB"]

LOGGER = getLogger("memory")


@dataclass
class MaxMemory:
    unit: Memory_Unit_Literal

    ram: float
    vram: Optional[float] = None
    reserved: Optional[float] = None
    allocated: Optional[float] = None

    def __add__(self, other: "MaxMemory") -> "MaxMemory":
        if self.unit != other.unit:
            raise ValueError(f"Cannot add memory with different units: {self.unit} and {other.unit}")

        ram = self.ram + other.ram
        vram = self.vram + other.vram if self.vram is not None and other.vram is not None else None
        reserved = self.reserved + other.reserved if self.reserved is not None and other.reserved is not None else None
        allocated = (
            self.allocated + other.allocated if self.allocated is not None and other.allocated is not None else None
        )

        return MaxMemory(unit=self.unit, ram=ram, vram=vram, reserved=reserved, allocated=allocated)

    @staticmethod
    def aggregate(max_memories: List["MaxMemory"]) -> "MaxMemory":
        if len(max_memories) == 0 or all(memory is None for memory in max_memories):
            return None
        elif any(memory is None for memory in max_memories):
            raise ValueError("Some memory measurements are missing")

        return reduce(lambda x, y: x + y, max_memories)

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} max RAM memory: {self.ram:f} ({self.unit})")
        if self.vram is not None:
            LOGGER.info(f"\t\t+ {prefix} max VRAM memory: {self.vram:f} ({self.unit})")
        if self.reserved is not None:
            LOGGER.info(f"\t\t+ {prefix} max reserved memory: {self.reserved:f} ({self.unit})")
        if self.allocated is not None:
            LOGGER.info(f"\t\t+ {prefix} max allocated memory: {self.allocated:f} ({self.unit})")


class MemoryTracker:
    def __init__(self, device: str, backend: str, device_ids: Optional[str] = None):
        self.device = device
        self.backend = backend
        self.device_ids = device_ids

        self.ram_memory = []
        self.vram_memory = []
        self.reserved_memory = {}
        self.allocated_memory = {}

        LOGGER.info("\t+ Tracking RAM memory")

        if self.device == "cuda":
            if self.device_ids is None:
                LOGGER.warning("\t+ `device=cuda` but `device_ids` not provided. Using all available CUDA devices.")
                self.device_ids = get_cuda_device_ids()

            self.device_ids = list(map(int, self.device_ids.split(",")))
            LOGGER.info(f"\t+ Tracking VRAM memory of CUDA devices: {self.device_ids}")

            if self.backend == "pytorch":
                num_pytorch_cuda_devices = torch.cuda.device_count()
                if len(self.device_ids) != num_pytorch_cuda_devices:
                    raise ValueError(
                        "The number of CUDA devices and Pytorch CUDA devices must be the same. "
                        f"Got {len(self.device_ids)} and {num_pytorch_cuda_devices} respectively."
                    )
                LOGGER.info(
                    f"\t+ Tracking Allocated/Reserved memory of {num_pytorch_cuda_devices} Pytorch CUDA devices"
                )

    def reset(self):
        self.ram_memory = []
        self.vram_memory = []
        self.reserved_memory = {}
        self.allocated_memory = {}

    @contextmanager
    def track(self):
        if self.device == "cuda" and self.backend == "pytorch":
            yield from self._cuda_pytorch_memory()
        elif self.device == "cuda":
            yield from self._cuda_memory()
        else:
            yield from self._cpu_memory()

    def _cuda_pytorch_memory(self, interval: float = 0.001):
        torch.cuda.empty_cache()

        for device in range(torch.cuda.device_count()):
            self.reserved_memory[device] = []
            self.allocated_memory[device] = []
            try:
                torch.cuda.reset_peak_memory_stats(device=device)
            except Exception as e:
                LOGGER.warning(f"\t+ Could not reset max memory stats for device {device}: {e}")

            # initial memory usage
            self.reserved_memory[device].append(torch.cuda.memory_reserved(device=device))
            self.allocated_memory[device].append(torch.cuda.memory_allocated(device=device))

        # start recording memory allocations
        torch.cuda.memory._record_memory_history()

        yield from self._cuda_memory(interval)

        # get snapshots and stop recording memory allocations
        memory_device_traces = torch.cuda.memory._snapshot()["device_traces"]
        torch.cuda.memory._record_memory_history(enabled=None)

        for device in range(torch.cuda.device_count()):
            device_trace = memory_device_traces[device]
            for entry in device_trace:
                if entry["action"] == "alloc":
                    self.allocated_memory[device].append(self.allocated_memory[device][-1] + entry["size"])
                elif entry["action"] == "free_completed":
                    self.allocated_memory[device].append(self.allocated_memory[device][-1] - entry["size"])
                elif entry["action"] == "segment_alloc":
                    self.reserved_memory[device].append(self.reserved_memory[device][-1] + entry["size"])
                elif entry["action"] == "segment_free":
                    self.reserved_memory[device].append(self.reserved_memory[device][-1] - entry["size"])

        LOGGER.debug(f"\t+ Max allocated memory: {self.get_max_allocated_memory_mb()} MB")
        LOGGER.debug(f"\t+ Max reserved memory: {self.get_max_reserved_memory_mb()} MB")

    def _cuda_memory(self, interval: float = 0.0001):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_gpu_vram_memory,
            args=(os.getpid(), self.device_ids, child_connection, interval),
            daemon=True,
        )
        memory_process.start()
        parent_connection.recv()  # wait for memory process to be ready

        yield from self._cpu_memory(interval)

        parent_connection.send(True)
        self.vram_memory = parent_connection.recv()
        LOGGER.debug(f"\t+ Max memory (VRAM) used: {self.get_max_vram_memory_mb()} MB")

    def _cpu_memory(self, interval: float = 0.0001):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_cpu_ram_memory,
            args=(os.getpid(), child_connection, interval),
            daemon=True,
        )
        memory_process.start()
        parent_connection.recv()  # wait for memory process to be ready

        yield

        parent_connection.send(True)
        self.ram_memory = parent_connection.recv()
        LOGGER.debug(f"\t+ Max memory (RAM) used: {self.get_max_ram_memory_mb()} MB")

    def get_ram_memory_mb(self) -> List[int]:
        return [bytes_to_mega_bytes(memory) for memory in self.ram_memory]

    def get_vram_memory_mb(self) -> List[int]:
        return [bytes_to_mega_bytes(memory) for memory in self.vram_memory]

    def get_reserved_memory_mb(self) -> Dict[str, List[int]]:
        return {
            device: [bytes_to_mega_bytes(memory) for memory in self.reserved_memory[device]]
            for device in self.reserved_memory
        }

    def get_allocated_memory_mb(self) -> Dict[str, List[int]]:
        return {
            device: [bytes_to_mega_bytes(memory) for memory in self.allocated_memory[device]]
            for device in self.allocated_memory
        }

    def get_max_ram_memory_mb(self) -> int:
        return compute_max(self.get_ram_memory_mb())

    def get_max_vram_memory_mb(self) -> int:
        return compute_max(self.get_vram_memory_mb())

    def get_max_reserved_memory_mb(self) -> Dict[str, int]:
        reserved_memory_mb = self.get_reserved_memory_mb()
        return sum(compute_max(reserved_memory_mb[device]) for device in reserved_memory_mb)

    def get_max_allocated_memory_mb(self) -> Dict[str, int]:
        allocated_memory_mb = self.get_allocated_memory_mb()
        return sum(compute_max(allocated_memory_mb[device]) for device in allocated_memory_mb)

    def get_max_memory(self):
        if self.device == "cuda" and self.backend == "pytorch":
            return MaxMemory(
                unit="MB",
                ram=self.get_max_ram_memory_mb(),
                vram=self.get_max_vram_memory_mb(),
                reserved=self.get_max_reserved_memory_mb(),
                allocated=self.get_max_allocated_memory_mb(),
            )
        elif self.device == "cuda":
            return MaxMemory(
                unit="MB",
                ram=self.get_max_ram_memory_mb(),
                vram=self.get_max_vram_memory_mb(),
            )
        else:
            return MaxMemory(
                unit="MB",
                ram=self.get_max_ram_memory_mb(),
            )


def monitor_cpu_ram_memory(process_id: int, connection: Connection, interval: float):
    process = psutil.Process(process_id)
    connection.send(0)
    used_memory = []
    stop = False

    while not stop:
        meminfo_attr = "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
        current_used_memory = getattr(process, meminfo_attr)()[0]
        used_memory.append(current_used_memory)
        stop = connection.poll(interval)

    connection.send(used_memory)
    connection.close()


def monitor_gpu_vram_memory(process_id: int, device_ids: List[int], connection: Connection, interval: float):
    if is_nvidia_system():
        if not is_pynvml_available():
            raise ValueError(
                "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
            )
        pynvml.nvmlInit()
        devices_handles = [pynvml.nvmlDeviceGetHandleByIndex(device_id) for device_id in device_ids]
        connection.send(0)
        used_memory = []
        stop = False

        while not stop:
            current_used_memory = 0
            for device_id, device_handle in zip(device_ids, devices_handles):
                device_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
                for device_process in device_processes:
                    if device_process.pid == process_id or (
                        psutil.pid_exists(device_process.pid)
                        and psutil.Process(device_process.pid).parent().pid == process_id
                    ):
                        # only memory usage of the process and its children is tracked
                        current_used_memory += device_process.usedGpuMemory

            used_memory.append(current_used_memory)
            stop = connection.poll(interval)

        connection.send(used_memory)
        pynvml.nvmlShutdown()
        connection.close()
    elif is_rocm_system():
        if not is_amdsmi_available():
            raise ValueError(
                "The library amdsmi is required to run memory benchmark on AMD GPUs, but is not installed. "
                "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
            )
        amdsmi.amdsmi_init()
        # we can only get all handles at once
        devices_handles = amdsmi.amdsmi_get_processor_handles()
        connection.send(0)
        used_memory = []
        stop = False

        while not stop:
            current_used_memory = 0
            for device_id in device_ids:
                device_handle = devices_handles[device_id]
                device_process = amdsmi.amdsmi_get_gpu_process_list(device_handle)
                for process_handle in device_process:
                    process_info = amdsmi.amdsmi_get_gpu_process_info(device_handle, process_handle)
                    if process_info["pid"] == process_id or (
                        psutil.pid_exists(process_info["pid"])
                        and psutil.Process(process_info["pid"]).parent().pid == process_id
                    ):
                        # only memory usage of the process and its children is tracked
                        current_used_memory += process_info["memory_usage"]["vram_mem"]

            used_memory.append(current_used_memory)
            stop = connection.poll(interval)

        connection.send(used_memory)
        amdsmi.amdsmi_shut_down()
        connection.close()
    else:
        raise ValueError("Only NVIDIA and AMD ROCm GPUs are supported for CUDA memory tracking.")
