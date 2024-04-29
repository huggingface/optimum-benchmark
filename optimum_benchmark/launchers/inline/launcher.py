from logging import getLogger
from typing import Callable

from ...report import BenchmarkReport
from ..base import Launcher
from .config import InlineConfig

LOGGER = getLogger("inline")


class InlineLauncher(Launcher[InlineConfig]):
    NAME = "inline"

    def __init__(self, config: InlineConfig):
        super().__init__(config)

    def launch(self, worker: Callable, *worker_args) -> BenchmarkReport:
        LOGGER.warn(
            "\t+ Running benchmark in the main process. "
            "This is only recommended for debugging purposes and not for benchmarking."
        )
        report = worker(*worker_args)

        return report
