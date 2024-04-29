from abc import ABC
from logging import getLogger
from typing import ClassVar, Generic

from ..backends.base import Backend
from ..report import BenchmarkReport
from .config import ScenarioConfigT

LOGGER = getLogger("benchmark")


class Scenario(Generic[ScenarioConfigT], ABC):
    NAME: ClassVar[str]

    def __init__(self, config: ScenarioConfigT) -> None:
        LOGGER.info(f"Allocating {self.NAME} benchmark")
        self.config = config

    def run(self, backend: Backend) -> None:
        raise NotImplementedError("Scenario must implement run method")

    def get_report(self) -> BenchmarkReport:
        raise NotImplementedError("Scenario must implement report method")
