from abc import ABC
from logging import getLogger
from typing import Callable, ClassVar, Generic

from ..benchmarks.report import BenchmarkReport
from .config import LauncherConfigT

LOGGER = getLogger("launcher")


class Launcher(Generic[LauncherConfigT], ABC):
    NAME: ClassVar[str]

    config: LauncherConfigT

    def __init__(self, config: LauncherConfigT):
        LOGGER.info(f"ََAllocating {self.NAME} launcher")
        self.config = config

    def launch(self, worker: Callable, *worker_args) -> BenchmarkReport:
        raise NotImplementedError("Launcher must implement launch method")
