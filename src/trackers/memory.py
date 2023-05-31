from multiprocessing.connection import Connection
from multiprocessing import Pipe, Process
from contextlib import contextmanager
from logging import getLogger
from typing import List

# import platform
# if platform.system() == "Windows":
#     from signal import CTRL_C_EVENT as SIGKILL  # type: ignore
# else:
#     from signal import SIGKILL

import py3nvml.py3nvml as nvml
import psutil
import torch
import os

from src.utils import bytes_to_mega_bytes

LOGGER = getLogger("memory")


class PeakMemoryTracker:
    def __init__(self, device: str):
        self.device = device
        self.tracked_peak_memory: int = 0

    @contextmanager
    def track(self, interval: float = 0.01):
        if self.device == "cuda":
            yield from self._track_cuda_peak_memory()
        else:
            yield from self._track_cpu_peak_memory(interval)

    def get_tracked_peak_memory(self):
        return self.tracked_peak_memory

    def _track_cuda_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()
        nvml.nvmlInit()
        yield
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
        nvml.nvmlShutdown()

        self.tracked_peak_memory = meminfo.used  # type: ignore
        LOGGER.debug(f"Peak memory usage: {bytes_to_mega_bytes(self.tracked_peak_memory)} MB")  # type: ignore

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
        # receive memory and num measurements
        max_memory = parent_connection.recv()
        num_measurements = parent_connection.recv()

        self.tracked_peak_memory = max_memory
        LOGGER.debug(f"Peak memory usage: {bytes_to_mega_bytes(max_memory)} MB")
        LOGGER.debug(f"Peak memory in {num_measurements} measurements")


class PeakMemoryMeasureProcess(Process):
    """
    `MemoryMeasureProcess` inherits from `Process` and overwrites its `run()` method. Used to measure the
    memory usage of a process
    """

    def __init__(self, process_id: int, child_connection: Connection, interval: float):
        super().__init__()
        self.process_id = process_id
        self.interval = interval
        self.connection = child_connection
        self.num_measurements = 1
        self.mem_usage = 0

    def run(self):
        self.connection.send(0)
        stop = False

        while True:
            process = psutil.Process(self.process_id)
            try:
                meminfo_attr = (
                    "memory_info"
                    if hasattr(process, "memory_info")
                    else "get_memory_info"
                )
                memory = getattr(process, meminfo_attr)()[0]
            except psutil.AccessDenied:
                raise ValueError("Error with Psutil.")

            self.mem_usage = max(self.mem_usage, memory)
            self.num_measurements += 1

            if stop:
                break

            stop = self.connection.poll(self.interval)

        # send results to parent pipe
        self.connection.send(self.mem_usage)
        self.connection.send(self.num_measurements)
        self.connection.close()
