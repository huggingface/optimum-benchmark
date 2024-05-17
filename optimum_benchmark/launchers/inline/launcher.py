from typing import Any, Callable, List

from ...benchmark.report import BenchmarkReport
from ..base import Launcher
from .config import InlineConfig


class InlineLauncher(Launcher[InlineConfig]):
    NAME = "inline"

    def __init__(self, config: InlineConfig):
        super().__init__(config)

    def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
        self.logger.warn("The inline launcher is only recommended for debugging purposes and not for benchmarking")
        report = worker(*worker_args)
        return report
