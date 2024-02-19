from typing import Callable
from logging import getLogger

from ..base import Launcher
from .config import InlineConfig
from ..isolation_utils import device_isolation
from ...benchmarks.report import BenchmarkReport

LOGGER = getLogger("inline")


class InlineLauncher(Launcher[InlineConfig]):
    NAME = "inline"

    def __init__(self, config: InlineConfig):
        super().__init__(config)

    def launch(self, worker: Callable, *worker_args) -> BenchmarkReport:
        with device_isolation(enabled=self.config.device_isolation):
            LOGGER.info("\t+ Launching inline worker (no process isolation)")
            report = worker(*worker_args)

        return report
