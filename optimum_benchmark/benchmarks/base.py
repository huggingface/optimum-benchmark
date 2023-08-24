from dataclasses import dataclass
from logging import getLogger
from abc import ABC

from optimum_benchmark.backends.base import Backend


LOGGER = getLogger("benchmark")


@dataclass
class BenchmarkConfig(ABC):
    name: str
    _target_: str


class Benchmark(ABC):
    name: str
    config: BenchmarkConfig

    def __init__(self) -> None:
        pass

    def configure(self, config: BenchmarkConfig) -> None:
        LOGGER.info(f"Configuring {self.name} benchmark")
        self.config = config

    def run(self, backend: Backend) -> None:
        raise NotImplementedError("Benchmark must implement run method")

    def save(self) -> None:
        raise NotImplementedError("Benchmark must implement save method")
