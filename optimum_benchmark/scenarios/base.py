from abc import ABC
from logging import getLogger
from typing import ClassVar, Generic

from ..backends.base import Backend
from ..report import BenchmarkReport
from .config import ScenarioConfigT

LOGGER = getLogger("scenario")


class Scenario(Generic[ScenarioConfigT], ABC):
    NAME: ClassVar[str]

    def __init__(self, config: ScenarioConfigT) -> None:
        LOGGER.info(f"Allocating {self.NAME} scenario")
        self.config = config

    def run(self, backend: Backend) -> BenchmarkReport:
        raise NotImplementedError("Scenario must implement run method")
