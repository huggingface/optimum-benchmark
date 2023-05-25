from dataclasses import dataclass, MISSING
from abc import ABC, abstractmethod
from logging import getLogger
from pandas import DataFrame

from src.backend.base import Backend

LOGGER = getLogger("benchmark")


@dataclass
class BenchmarkConfig(ABC):
    name: str = MISSING  # type: ignore


class Benchmark(ABC):
    def __init__(self, model: str, task: str, device: str) -> None:
        self.model = model
        self.task = task
        self.device = device

    @abstractmethod
    def configure(self, config: BenchmarkConfig) -> None:
        raise NotImplementedError("Benchmark must implement configure method")

    @abstractmethod
    def run(self, backend: Backend) -> None:
        raise NotImplementedError("Benchmark must implement run_benchmark method")

    @property
    @abstractmethod
    def results(self, path: str = "") -> DataFrame:
        raise NotImplementedError("Benchmark must implement save_results method")

    @property
    @abstractmethod
    def objective(self) -> float:
        raise NotImplementedError("Benchmark must implement save_results method")

    @abstractmethod
    def save(self, path: str = "") -> None:
        raise NotImplementedError("Benchmark must implement save_results method")
