import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from logging import getLogger
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import List, Literal, Optional, Union

from rich.console import Console
from rich.markdown import Markdown

from ..import_utils import is_amdsmi_available, is_pynvml_available, is_torch_available
from ..system_utils import is_nvidia_system, is_rocm_system

if is_nvidia_system() and is_pynvml_available():
    import pynvml  # type: ignore

if is_rocm_system() and is_amdsmi_available():
    import amdsmi  # type: ignore

if is_torch_available():
    import torch

import psutil

CONSOLE = Console()
LOGGER = getLogger("memory")

MEMORY_UNIT = "MB"
MEMORY_CONSUMPTION_SAMPLING_RATE = 0.01  # in seconds

Memory_Unit_Literal = Literal["MB"]


@dataclass
class Memory:
    unit: Memory_Unit_Literal

    max_ram: float
    max_global_vram: Optional[float] = None
    max_process_vram: Optional[float] = None
    max_reserved: Optional[float] = None
    max_allocated: Optional[float] = None

    @staticmethod
    def aggregate_across_processes(memories: List["Memory"]) -> "Memory":
        if len(memories) == 0:
            raise ValueError("No memory measurements to aggregate")
        elif any(memory is None for memory in memories):
            raise ValueError("Some memory measurements are missing")

        # ram, reserved, allocated, and process_vram measurements are process-specific so they are summed
        max_ram = sum(memory.max_ram for memory in memories) if memories[0].max_ram is not None else None
        max_reserved = sum(memory.max_reserved for memory in memories) if memories[0].max_reserved is not None else None
        max_allocated = (
            sum(memory.max_allocated for memory in memories) if memories[0].max_allocated is not None else None
        )
        max_process_vram = (
            sum(memory.max_process_vram for memory in memories) if memories[0].max_process_vram is not None else None
        )
        # global_vram is not process-specific so we take the average
        max_global_vram = (
            sum(memory.max_global_vram for memory in memories) / len(memories)
            if memories[0].max_global_vram is not None
            else None
        )
        unit = memories[0].unit

        return Memory(
            unit=unit,
            max_ram=max_ram,
            max_global_vram=max_global_vram,
            max_process_vram=max_process_vram,
            max_reserved=max_reserved,
            max_allocated=max_allocated,
        )

    def to_plain_text(self) -> str:
        plain_text = ""
        if self.max_ram is not None:
            plain_text += "\t\t+ max_ram: {max_ram:.2f} ({unit})\n"
        if self.max_global_vram is not None:
            plain_text += "\t\t+ max_global_vram: {max_global_vram:.2f} ({unit})\n"
        if self.max_process_vram is not None:
            plain_text += "\t\t+ max_process_vram: {max_process_vram:.2f} ({unit})\n"
        if self.max_reserved is not None:
            plain_text += "\t\t+ max_reserved: {max_reserved:.2f} ({unit})\n"
        if self.max_allocated is not None:
            plain_text += "\t\t+ max_allocated: {max_allocated:.2f} ({unit})\n"
        return plain_text.format(**asdict(self))

    def log(self):
        for line in self.to_plain_text().split("\n"):
            if line:
                LOGGER.info(line)

    def to_markdown_text(self) -> str:
        markdown_text = ""
        markdown_text += "| metric | value | unit |\n"
        markdown_text += "| ------ | ----: | ---: |\n"
        if self.max_ram is not None:
            markdown_text += "| max_ram          |          {max_ram:.2f} | {unit} |\n"
        if self.max_global_vram is not None:
            markdown_text += "| max_global_vram  |  {max_global_vram:.2f} | {unit} |\n"
        if self.max_process_vram is not None:
            markdown_text += "| max_process_vram | {max_process_vram:.2f} | {unit} |\n"
        if self.max_reserved is not None:
            markdown_text += "| max_reserved     |     {max_reserved:.2f} | {unit} |\n"
        if self.max_allocated is not None:
            markdown_text += "| max_allocated    |    {max_allocated:.2f} | {unit} |\n"
        return markdown_text.format(**asdict(self))

    def print(self):
        CONSOLE.print(Markdown(self.to_markdown_text()))


class MemoryTracker:
    def __init__(self, device: str, backend: str, device_ids: Optional[Union[str, int, List[int]]] = None):
        self.device = device
        self.backend = backend
        self.device_ids = device_ids
        self.monitored_pid = os.getpid()

        self.is_gpu = device == "cuda"
        self.is_pytorch_cuda = (self.backend, self.device) == ("pytorch", "cuda")

        LOGGER.info(f"\t\t+ Tracking RAM memory of process {self.monitored_pid}")

        if self.is_gpu:
            if isinstance(self.device_ids, str):
                self.device_ids = list(map(int, self.device_ids.split(",")))
            elif isinstance(self.device_ids, int):
                self.device_ids = [self.device_ids]
            elif isinstance(self.device_ids, list):
                self.device_ids = self.device_ids
            elif self.device_ids is None:
                raise ValueError("GPU device IDs must be provided for energy tracking on GPUs")
            else:
                raise ValueError("GPU device IDs must be a string, an integer, or a list of integers")

            LOGGER.info(f"\t\t+ Tracking GPU memory of devices {self.device_ids}")

        if self.is_pytorch_cuda:
            self.num_pytorch_devices = torch.cuda.device_count()
            if len(self.device_ids) != self.num_pytorch_devices:
                raise ValueError(
                    "The number of target GPU devices and Pytorch's GPU device count do not match. "
                    f"Got {len(self.device_ids)} and {self.num_pytorch_devices} respectively."
                )

            LOGGER.info(f"\t\t+ Tracking Allocated/Reserved memory of {self.num_pytorch_devices} Pytorch CUDA devices")

        self.max_ram_memory = None
        self.max_global_vram_memory = None
        self.max_process_vram_memory = None
        self.max_reserved_memory = None
        self.max_allocated_memory = None

    def reset(self):
        self.max_ram_memory = None
        self.max_global_vram_memory = None
        self.max_process_vram_memory = None
        self.max_reserved_memory = None
        self.max_allocated_memory = None

    @contextmanager
    def track(self):
        if self.is_pytorch_cuda:
            yield from self._cuda_pytorch_memory()
        elif self.is_gpu:
            yield from self._gpu_memory()
        else:
            yield from self._cpu_memory()

    def _cuda_pytorch_memory(self):
        self.max_allocated_memory = 0
        self.max_reserved_memory = 0

        for device in range(self.num_pytorch_devices):
            try:
                torch.cuda.reset_peak_memory_stats(device=device)
            except Exception as e:
                LOGGER.warning(f"\t\t+ Could not reset max memory stats for device {device}: {e}")

        torch.cuda.synchronize()

        yield from self._gpu_memory()

        torch.cuda.synchronize()

        for device in range(self.num_pytorch_devices):
            try:
                self.max_allocated_memory += torch.cuda.max_memory_allocated(device=device) / 1e6
                self.max_reserved_memory += torch.cuda.max_memory_reserved(device=device) / 1e6
            except Exception as e:
                LOGGER.warning(f"\t\t+ Could not get max memory stats for device {device}: {e}")

    def _gpu_memory(self):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_gpu_vram_memory, args=(self.monitored_pid, self.device_ids, child_connection), daemon=True
        )
        memory_process.start()

        if memory_process.is_alive():
            _ = parent_connection.recv()
        else:
            raise ValueError("Could not start memory tracking process for GPU devices.")

        yield from self._cpu_memory()

        if memory_process.is_alive():
            parent_connection.send(0)
        else:
            raise ValueError("Could not stop memory tracking process for GPU devices.")

        if memory_process.is_alive():
            self.max_global_vram_memory = parent_connection.recv()
            self.max_process_vram_memory = parent_connection.recv()
        else:
            raise ValueError("Could not get memory tracking results for GPU devices.")

        parent_connection.close()

    def _cpu_memory(self):
        child_connection, parent_connection = Pipe()
        memory_process = Process(
            target=monitor_cpu_ram_memory, args=(self.monitored_pid, child_connection), daemon=True
        )
        memory_process.start()

        if memory_process.is_alive():
            _ = parent_connection.recv()
        else:
            raise ValueError("Could not start memory tracking process for CPU.")

        yield

        if memory_process.is_alive():
            parent_connection.send(0)
        else:
            raise ValueError("Could not stop memory tracking process for CPU.")

        if memory_process.is_alive():
            self.max_ram_memory = parent_connection.recv()
        else:
            raise ValueError("Could not get memory tracking results for CPU.")

        parent_connection.close()

    def get_max_memory(self):
        assert self.max_ram_memory is not None, "Memory tracker must be run before getting the maximum memory"

        return Memory(
            unit=MEMORY_UNIT,
            max_ram=self.max_ram_memory,
            max_global_vram=self.max_global_vram_memory,
            max_process_vram=self.max_process_vram_memory,
            max_reserved=self.max_reserved_memory,
            max_allocated=self.max_allocated_memory,
        )


def monitor_cpu_ram_memory(monitored_pid: int, connection: Connection):
    stop = False
    max_used_memory = 0
    monitored_process = psutil.Process(monitored_pid)

    if monitored_process.is_running():
        try:
            connection.send(0)
        except Exception:
            exit(0)

    while monitored_process.is_running() and not stop:
        meminfo_attr = "memory_info" if hasattr(monitored_process, "memory_info") else "get_memory_info"
        used_memory = getattr(monitored_process, meminfo_attr)()[0]
        max_used_memory = max(max_used_memory, used_memory)
        stop = connection.poll(MEMORY_CONSUMPTION_SAMPLING_RATE)

    if monitored_process.is_running():
        try:
            connection.send(max_used_memory / 1e6)  # convert to MB
        except Exception:
            exit(0)

    connection.close()


def monitor_gpu_vram_memory(monitored_pid: int, device_ids: List[int], connection: Connection):
    stop = False
    max_used_global_memory = 0
    max_used_process_memory = 0
    monitored_process = psutil.Process(monitored_pid)

    if monitored_process.is_running():
        try:
            connection.send(0)
        except Exception:
            exit(0)

    if is_nvidia_system():
        if not is_pynvml_available():
            raise ValueError(
                "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
            )

        pynvml.nvmlInit()
        devices_handles = [pynvml.nvmlDeviceGetHandleByIndex(device_id) for device_id in device_ids]

        while monitored_process.is_running() and not stop:
            used_global_memory = 0
            used_process_memory = 0

            monitored_pids = [monitored_pid] + [child.pid for child in monitored_process.children(recursive=True)]

            for device_id, device_handle in zip(device_ids, devices_handles):
                try:
                    device_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
                except Exception as e:
                    LOGGER.warning(f"Could not get process list for device {device_id}: {e}.")
                    continue

                for device_process in device_processes:
                    if device_process.pid in monitored_pids:
                        used_process_memory += device_process.usedGpuMemory

                try:
                    device_memory = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
                except Exception as e:
                    LOGGER.warning(f"Could not get memory info for device {device_id}: {e}.")
                    continue

                used_global_memory += device_memory.used

            max_used_global_memory = max(max_used_global_memory, used_global_memory)
            max_used_process_memory = max(max_used_process_memory, used_process_memory)
            stop = connection.poll(MEMORY_CONSUMPTION_SAMPLING_RATE)

        pynvml.nvmlShutdown()

    elif is_rocm_system():
        if not is_amdsmi_available():
            raise ValueError(
                "The library AMD SMI is required to track process-specific memory benchmark on AMD GPUs, but is not installed. "
                "Please install the official and AMD maintained AMD SMI library from https://github.com/ROCm/amdsmi."
            )

        amdsmi.amdsmi_init()
        permission_denied = False
        devices_handles = amdsmi.amdsmi_get_processor_handles()

        while monitored_process.is_running() and not stop:
            used_global_memory = 0
            used_process_memory = 0

            monitored_pids = [monitored_pid] + [child.pid for child in monitored_process.children(recursive=True)]

            for device_id in device_ids:
                device_handle = devices_handles[device_id]

                try:
                    used_global_memory += amdsmi.amdsmi_get_gpu_memory_total(
                        device_handle, mem_type=amdsmi.AmdSmiMemoryType.VRAM
                    )
                except Exception as e:
                    LOGGER.warning(f"Could not get memory usage for device {device_id}: {e}")

                if permission_denied:
                    continue

                try:
                    processes_handles = amdsmi.amdsmi_get_gpu_process_list(device_handle)
                except Exception as e:
                    LOGGER.warning(f"Could not get process list for device {device_id}: {e}")
                    permission_denied = "Permission Denied" in str(e)
                    continue

                for process_handle in processes_handles:
                    try:
                        gpu_process_info = amdsmi.amdsmi_get_gpu_process_info(device_handle, process_handle)
                    except Exception as e:
                        LOGGER.warning(f"Could not get process info for process {process_handle}: {e}")
                        permission_denied = "Permission Denied" in str(e)
                        continue

                    if gpu_process_info["pid"] in monitored_pids:
                        max_used_process_memory += gpu_process_info["memory_usage"]["vram_mem"]

            max_used_global_memory = max(max_used_global_memory, used_global_memory)
            max_used_process_memory = max(max_used_process_memory, used_process_memory)
            stop = connection.poll(MEMORY_CONSUMPTION_SAMPLING_RATE)

        amdsmi.amdsmi_shut_down()

    else:
        raise ValueError("Only NVIDIA and AMD ROCm GPUs are supported for VRAM tracking.")

    if monitored_process.is_running():
        try:
            connection.send(max_used_global_memory / 1e6)  # convert to MB
            connection.send(max_used_process_memory / 1e6)  # convert to MB
        except Exception:
            exit(0)

    connection.close()
