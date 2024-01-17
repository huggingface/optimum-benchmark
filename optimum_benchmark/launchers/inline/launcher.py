import os
from logging import getLogger
from typing import Callable

from ..base import Launcher
from ..isolation_utils import device_isolation
from .config import InlineConfig

LOGGER = getLogger("inline")


class InlineLauncher(Launcher[InlineConfig]):
    NAME = "inline"

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: InlineConfig) -> None:
        super().configure(config)

    def launch(self, worker: Callable, *worker_args):
        with device_isolation(enabled=self.config.device_isolation, benchmark_pid=os.getpid()):
            LOGGER.info("\t+ Launching inline experiment (no process isolation)")
            worker(*worker_args)
