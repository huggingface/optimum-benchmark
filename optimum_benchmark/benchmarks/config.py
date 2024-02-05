from abc import ABC
from typing import TypeVar
from logging import getLogger
from dataclasses import dataclass


LOGGER = getLogger("benchmark")


@dataclass
class BenchmarkConfig(ABC):
    name: str
    _target_: str

    def __post_init__(self):
        pass


BenchmarkConfigT = TypeVar("BenchmarkConfigT", bound=BenchmarkConfig)
