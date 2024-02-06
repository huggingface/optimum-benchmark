from abc import ABC
from logging import getLogger
from typing import ClassVar, Generic, Dict, Any

from ..backends.base import Backend
from .config import BenchmarkConfigT


LOGGER = getLogger("benchmark")


class Benchmark(Generic[BenchmarkConfigT], ABC):
    NAME: ClassVar[str]

    config: BenchmarkConfigT

    def __init__(self, config: BenchmarkConfigT) -> None:
        LOGGER.info(f"Allocating {self.NAME} benchmark")
        self.config = config

    def run(self, backend: Backend) -> None:
        raise NotImplementedError("Benchmark must implement run method")

    def report(self) -> Dict[str, Any]:
        raise NotImplementedError("Benchmark must implement save method")
