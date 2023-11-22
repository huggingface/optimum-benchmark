from abc import ABC
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from ..backends.base import Backend

LOGGER = getLogger("benchmark")


@dataclass
class BenchmarkConfig(ABC):
    name: str
    _target_: str


BenchmarkConfigT = TypeVar("BenchmarkConfigT", bound=BenchmarkConfig)


class Benchmark(Generic[BenchmarkConfigT], ABC):
    NAME: ClassVar[str]

    config: BenchmarkConfigT

    def __init__(self) -> None:
        pass

    def configure(self, config: BenchmarkConfigT) -> None:
        LOGGER.info(f"Configuring {self.NAME} benchmark")
        self.config = config

    def run(self, backend: "Backend") -> None:
        raise NotImplementedError("Benchmark must implement run method")

    def get_results_df(self) -> None:
        raise NotImplementedError("Benchmark must implement get_results_df method")

    def save(self) -> None:
        raise NotImplementedError("Benchmark must implement save method")
