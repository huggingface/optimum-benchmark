from abc import ABC
from logging import getLogger
from typing import ClassVar, Generic

from ..backends.base import Backend
from .config import BenchmarkConfigT
from .report import BenchmarkReport

LOGGER = getLogger("benchmark")


class Benchmark(Generic[BenchmarkConfigT], ABC):
    NAME: ClassVar[str]

    def __init__(self, config: BenchmarkConfigT) -> None:
        LOGGER.info(f"Allocating {self.NAME} benchmark")
        self.config = config

    def run(self, backend: Backend) -> None:
        raise NotImplementedError("Benchmark must implement run method")

    def get_report(self) -> BenchmarkReport:
        raise NotImplementedError("Benchmark must implement report method")
