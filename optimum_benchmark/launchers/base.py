import os
from abc import ABC
from logging import getLogger
from multiprocessing import Process
from typing import Any, Callable, ClassVar, Generic, List, Optional

from ..report import BenchmarkReport
from ..system_utils import is_nvidia_system, is_rocm_system
from .config import LauncherConfigT
from .device_isolation_utils import assert_device_isolation


class Launcher(Generic[LauncherConfigT], ABC):
    NAME: ClassVar[str]

    config: LauncherConfigT

    def __init__(self, config: LauncherConfigT):
        self.config = config
        self.logger = getLogger(self.NAME)
        self.logger.info(f"Allocated {self.NAME} launcher")

    def start_device_isolation_process(self, pid: int, device_ids: Optional[str] = None):
        if device_ids is None:
            if is_nvidia_system():
                device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            elif is_rocm_system():
                device_ids = os.environ.get("ROCR_VISIBLE_DEVICES", None)

            if device_ids is None:
                raise ValueError(
                    "Device isolation requires either device_ids to be specified or the environment variable "
                    "`CUDA_VISIBLE_DEVICES` (for NVIDIA GPUs) or `ROCR_VISIBLE_DEVICES` (for AMD GPUs) to be set."
                )

        self.device_isolation_process = Process(
            target=assert_device_isolation,
            kwargs={
                "action": self.config.device_isolation_action,
                "device_ids": device_ids,
                "pid": pid,
            },
            daemon=True,
        )
        self.device_isolation_process.start()
        self.logger.info(f"\t+ Isolating device(s) [{device_ids}] for process [{pid}] and its children")
        self.logger.info(f"\t+ Executing action [{self.config.device_isolation_action}] in case of violation")

    def stop_device_isolation_process(self):
        self.logger.info("\t+ Stopping device isolation process")
        self.device_isolation_process.terminate()
        self.device_isolation_process.join()
        self.device_isolation_process.close()

    def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
        raise NotImplementedError("Launcher must implement launch method")
