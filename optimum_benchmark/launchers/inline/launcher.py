import os
from logging import getLogger
from typing import Callable, Dict, Any

from ..base import Launcher
from .config import InlineConfig
from ..isolation_utils import device_isolation

LOGGER = getLogger("inline")


class InlineLauncher(Launcher[InlineConfig]):
    NAME = "inline"

    def __init__(self, config: InlineConfig):
        super().__init__(config)

    def launch(self, worker: Callable, *worker_args) -> Dict[str, Any]:
        with device_isolation(
            benchmark_pid=os.getpid(),
            enabled=self.config.device_isolation,
        ):
            LOGGER.info("\t+ Launching inline experiment (no process isolation)")
            report: Dict[str, Any] = worker(*worker_args)

        return report
