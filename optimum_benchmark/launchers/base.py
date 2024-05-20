import os
import shutil
import sys
import tempfile
from abc import ABC
from contextlib import contextmanager
from logging import getLogger
from multiprocessing import Process, set_executable
from typing import Any, Callable, ClassVar, Generic, List, Optional

from ..benchmark.report import BenchmarkReport
from ..system_utils import is_nvidia_system, is_rocm_system
from .config import LauncherConfigT
from .device_isolation_utils import assert_device_isolation

NUMA_EXECUTABLE_CONTENT = """#!/bin/bash
echo "Running with numactl wrapper"
echo "numactl path: {numactl_path}"
echo "numactl args: {numactl_args}"
echo "python path: {python_path}"
echo "python args: $@"
{numactl_path} {numactl_args} {python_path} "$@"
"""


class Launcher(Generic[LauncherConfigT], ABC):
    NAME: ClassVar[str]

    config: LauncherConfigT

    def __init__(self, config: LauncherConfigT):
        self.config = config
        self.logger = getLogger(self.NAME)
        self.logger.info(f"Allocated {self.NAME} launcher")

    def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
        raise NotImplementedError("Launcher must implement launch method")

    @contextmanager
    def device_isolation(self, pid: int, device_ids: Optional[str] = None):
        if device_ids is None:
            if is_nvidia_system():
                device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            elif is_rocm_system():
                device_ids = os.environ.get("ROCR_VISIBLE_DEVICES", None)

        self.device_isolation_process = Process(
            target=assert_device_isolation,
            kwargs={"action": self.config.device_isolation_action, "device_ids": device_ids, "pid": pid},
            daemon=True,
        )
        self.device_isolation_process.start()
        self.logger.info(f"\t+ Isolating device(s) [{device_ids}] for process [{pid}] and its children")
        self.logger.info(f"\t+ Executing action [{self.config.device_isolation_action}] in case of violation")

        yield

        self.logger.info("\t+ Stopping device isolation process")
        self.device_isolation_process.terminate()
        self.device_isolation_process.join()
        self.device_isolation_process.close()

    @contextmanager
    def numactl_executable(self):
        self.logger.info("\t+ Creating numactl wrapper executable for multiprocessing")
        python_path = sys.executable
        numactl_path = shutil.which("numactl")
        if numactl_path is None:
            raise RuntimeError("ِCould not find numactl executable. Please install numactl and try again.")
        numactl_args = " ".join([f"--{key}={value}" for key, value in self.config.numactl_kwargs.items()])
        numa_executable = tempfile.NamedTemporaryFile(delete=False, prefix="numa_executable_", suffix=".sh")
        numa_executable_content = NUMA_EXECUTABLE_CONTENT.format(
            numactl_path=numactl_path, numactl_args=numactl_args, python_path=python_path
        )
        numa_executable.write(numa_executable_content.encode())
        os.chmod(numa_executable.name, 0o777)
        numa_executable.close()

        self.logger.info("\t+ Setting multiprocessing executable to numactl wrapper")
        set_executable(numa_executable.name)

        yield

        self.logger.info("\t+ Resetting default multiprocessing executable")
        os.unlink(numa_executable.name)
        set_executable(sys.executable)
