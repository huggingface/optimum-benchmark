from logging import getLogger
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, ClassVar, Tuple

import torch

from src.benchmark.base import Benchmark
from src.backends.config import BackendConfig

LOGGER = getLogger('backend')
BackendConfigT = TypeVar('BackendConfigT', bound=BackendConfig)


class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    def __init__(self, model: str):
        self.model = model

    @classmethod
    @abstractmethod
    def allocate(cls, config: 'BenchmarkConfig') -> 'Backend':
        raise NotImplementedError()

    @abstractmethod
    def configure(self, config: BackendConfigT) -> None:
        raise NotImplementedError()

    @abstractmethod
    def execute(self, config: 'BenchmarkConfig') -> Tuple[Benchmark, torch.Tensor]:
        raise NotImplementedError()