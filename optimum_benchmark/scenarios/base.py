from abc import ABC
from logging import getLogger
from typing import ClassVar, Generic

from ..backends.base import Backend
from ..benchmark.report import BenchmarkReport
from .config import ScenarioConfigT


class Scenario(Generic[ScenarioConfigT], ABC):
    NAME: ClassVar[str]

    def __init__(self, config: ScenarioConfigT) -> None:
        self.config = config
        self.logger = getLogger(self.NAME)
        self.logger.info(f"Allocating {self.NAME} scenario")

    def run(self, backend: Backend) -> BenchmarkReport:
        raise NotImplementedError("Scenario must implement run method")
