from dataclasses import dataclass, MISSING
from abc import ABC, abstractmethod
from logging import getLogger


from optimum_benchmark.backends.base import Backend
from optimum_benchmark.utils import set_seed


LOGGER = getLogger("benchmark")


@dataclass
class BenchmarkConfig(ABC):
    name: str = MISSING  # type: ignore
    _target_: str = MISSING  # type: ignore

    # seed for reproducibility
    seed: int = 42


class Benchmark(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def configure(self, config: BenchmarkConfig) -> None:
        LOGGER.info(f"Configuring {config.name} benchmark")
        LOGGER.info(f"\t+ Setting seed({config.seed})")
        set_seed(config.seed)

    @abstractmethod
    def run(self, backend: Backend) -> None:
        raise NotImplementedError("Benchmark must implement run method")

    @abstractmethod
    def save(self, path: str = "") -> None:
        raise NotImplementedError("Benchmark must implement save method")
