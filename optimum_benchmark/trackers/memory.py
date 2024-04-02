import os
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import List, Literal, Optional

from ..import_utils import (
    is_amdsmi_available,
    is_pynvml_available,
    is_pyrsmi_available,
    is_torch_available,
    is_torch_distributed_available,
)
from ..system_utils import get_gpu_device_ids, is_nvidia_system, is_rocm_system

if is_rocm_system() and is_pyrsmi_available():
    from pyrsmi import rocml

if is_torch_distributed_available():
    import torch.distributed

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


PROCESS_SPECIFIC_VRAM = os.environ.get("PROCESS_SPECIFIC_VRAM", "1") == "1"


@dataclass
class Memory:
    unit: Memory_Unit_Literal

    max_ram: float
    max_vram: Optional[float] = None
    max_reserved: Optional[float] = None
    max_allocated: Optional[float] = None

    @staticmethod
    def aggregate(memories: List["Memory"]) -> "Memory":
        if len(memories) == 0:
            raise ValueError("No memory measurements to aggregate")
        elif any(memory is None for memory in memories):
            raise ValueError("Some memory measurements are missing")

        unit = memories[0].unit
        max_ram = sum(memory.max_ram for memory in memories)

        if PROCESS_SPECIFIC_VRAM:
            max_vram = sum(memory.max_vram for memory in memories) if memories[0].max_vram is not None else None
        else:
            max_vram = max(memory.max_vram for memory in memories) if memories[0].max_vram is not None else None

        max_reserved = sum(memory.max_reserved for memory in memories) if memories[0].max_reserved is not None else None
        max_allocated = (
            sum(memory.max_allocated for memory in memories) if memories[0].max_allocated is not None else None
        )

        return Memory(
            unit=unit, max_ram=max_ram, max_vram=max_vram, max_reserved=max_reserved, max_allocated=max_allocated
        )

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
        self.distributed = is_torch_distributed_available() and torch.distributed.is_initialized()

        LOGGER.info("\t+ Tracking RAM memory")

        if self.device == "cuda":
            if self.device_ids is None:
                LOGGER.warning("\t+ `device=cuda` but `device_ids` not provided. Using all available CUDA devices.")
                self.device_ids = get_gpu_device_ids()

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

        self.reset()

    def reset(self):
        self.max_ram_memory = 0
        self.max_vram_memory = 0
        self.max_reserved_memory = 0
        self.max_allocated_memory = 0

    @contextmanager
    def track(self):
        if self.distributed:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()] if self.device == "cuda" else None)

        if self.device == "cuda" and self.backend == "pytorch":
            yield from self._cuda_pytorch_memory()
        elif self.device == "cuda":
            yield from self._cuda_memory()
        else:
            yield from self._cpu_memory()

        if self.distributed:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()] if self.device == "cuda" else None)

    def _cuda_pytorch_memory(self):
        torch.cuda.empty_cache()

        for device in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(device=device)
            except Exception as e:
                LOGGER.warning(f"\t\t+ Could not reset max memory stats for device {device}: {e}")

        yield from self._cuda_memory()

        self.max_allocated_memory = sum(
            torch.cuda.max_memory_allocated(device=device) / 1e6 for device in range(torch.cuda.device_count())
        )
        self.max_reserved_memory = sum(
            torch.cuda.max_memory_reserved(device=device) / 1e6 for device in range(torch.cuda.device_count())
        )

        torch.cuda.empty_cache()

    def _cuda_memory(self):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_gpu_vram_memory, args=(os.getpid(), self.device_ids, child_connection), daemon=True
        )
        memory_process.start()
        parent_connection.recv()  # wait for memory process to be ready

        yield from self._cpu_memory()

        parent_connection.send(True)
        self.max_vram_memory = parent_connection.recv()

    def _cpu_memory(self):
        child_connection, parent_connection = Pipe()
        memory_process = Process(target=monitor_cpu_ram_memory, args=(os.getpid(), child_connection), daemon=True)
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
            return Memory(unit=MEMORY_UNIT, max_ram=self.max_ram_memory, max_vram=self.max_vram_memory)
        else:
            return Memory(unit=MEMORY_UNIT, max_ram=self.max_ram_memory)


def monitor_cpu_ram_memory(monitored_pid: int, connection: Connection, interval: float = 0.001):
    stop = False
    max_used_memory = 0
    process = psutil.Process(monitored_pid)
    connection.send(0)

    while not stop:
        meminfo_attr = "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
        used_memory = getattr(process, meminfo_attr)()[0]
        max_used_memory = max(max_used_memory, used_memory)
        stop = connection.poll(interval)

    connection.send(max_used_memory / 1e6)  # convert to MB
    connection.close()


def monitor_gpu_vram_memory(monitored_pid: int, device_ids: List[int], connection: Connection, interval: float = 0.01):
    stop = False
    max_used_memory = 0
    monitored_process = psutil.Process(monitored_pid)
    connection.send(0)

    if PROCESS_SPECIFIC_VRAM:
        LOGGER.warning(
            "Tracking process-specific VRAM usage. This will track the memory usage of the monitored process and its children only."
        )
    else:
        LOGGER.warning(
            "Tracking global-device VRAM usage. This will track the memory usage of monitored device(s). "
            "Which may include memory used by other processes that are not relevant to the monitored process."
        )

    if is_nvidia_system():
        if not is_pynvml_available():
            raise ValueError(
                "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
            )

        pynvml.nvmlInit()
        devices_handles = [pynvml.nvmlDeviceGetHandleByIndex(device_id) for device_id in device_ids]

        if PROCESS_SPECIFIC_VRAM:
            while not stop:
                used_memory = 0
                monitored_pids = [monitored_pid] + [child.pid for child in monitored_process.children(recursive=True)]

                for device_id, device_handle in zip(device_ids, devices_handles):
                    try:
                        device_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
                    except Exception as e:
                        LOGGER.warning(f"Could not get process list for device {device_id}: {e}.")
                        continue

                    for device_process in device_processes:
                        if device_process.pid in monitored_pids:
                            used_memory += device_process.usedGpuMemory

                max_used_memory = max(max_used_memory, used_memory)
                stop = connection.poll(interval)

        else:
            while not stop:
                used_memory = 0

                for device_id, device_handle in zip(device_ids, devices_handles):
                    try:
                        device_memory = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
                    except Exception as e:
                        LOGGER.warning(f"Could not get memory info for device {device_id}: {e}")
                        continue

                    used_memory += device_memory.used

                max_used_memory = max(max_used_memory, used_memory)
                stop = connection.poll(interval)

        pynvml.nvmlShutdown()

    elif is_rocm_system():
        if not is_amdsmi_available() and not is_pyrsmi_available():
            raise ValueError(
                "Either the library AMD SMI or PyRSMI is required to run memory benchmark on AMD GPUs, but neither is installed. "
                "Please install the official and AMD maintained AMD SMI library from https://github.com/ROCm/amdsmi "
                "or PyRSMI library from https://github.com/ROCm/pyrsmi."
            )

        if PROCESS_SPECIFIC_VRAM:
            if not is_amdsmi_available():
                raise ValueError(
                    "The library AMD SMI is required to run process-specific memory benchmark on AMD GPUs, but is not installed. "
                    "Please install the official and AMD maintained AMD SMI library from https://github.com/ROCm/amdsmi."
                )

            amdsmi.amdsmi_init()
            devices_handles = amdsmi.amdsmi_get_processor_handles()
            while not stop:
                used_memory = 0
                monitored_pids = [monitored_pid] + [child.pid for child in monitored_process.children(recursive=True)]

                for device_id in device_ids:
                    device_handle = devices_handles[device_id]
                    try:
                        processes_handles = amdsmi.amdsmi_get_gpu_process_list(device_handle)
                    except Exception as e:
                        LOGGER.warning(f"Could not get process list for device {device_id}: {e}")
                        continue

                    for process_handle in processes_handles:
                        try:
                            gpu_process_info = amdsmi.amdsmi_get_gpu_process_info(device_handle, process_handle)
                        except Exception as e:
                            LOGGER.warning(f"Could not get process info for process {process_handle}: {e}")
                            continue

                        if gpu_process_info["pid"] in monitored_pids:
                            used_memory += gpu_process_info["memory_usage"]["vram_mem"]

                max_used_memory = max(max_used_memory, used_memory)
                stop = connection.poll(interval)

            amdsmi.amdsmi_shut_down()

        else:
            if not is_pyrsmi_available():
                raise ValueError(
                    "The library PyRSMI is required to run global-device memory benchmark on AMD GPUs, but is not installed. "
                    "Please install the official and AMD maintained PyRSMI library from https://github.com/ROCm/pyrsmi."
                )

            rocml.smi_initialize()
            while not stop:
                used_memory = 0
                for device_id in device_ids:
                    try:
                        used_memory += rocml.smi_get_device_memory_used(device_id)
                    except Exception as e:
                        LOGGER.warning(f"Could not get memory usage for device {device_id}: {e}")

                max_used_memory = max(max_used_memory, used_memory)
                stop = connection.poll(interval)

            rocml.smi_shutdown()

    else:
        raise ValueError("Only NVIDIA and AMD ROCm GPUs are supported for CUDA memory tracking.")

    connection.send(max_used_memory / 1e6)  # convert to MB
    connection.close()
