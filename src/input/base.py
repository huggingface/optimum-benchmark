from dataclasses import MISSING, dataclass
from abc import ABC, abstractmethod
from typing import ClassVar, Dict
from logging import getLogger
from torch import Tensor

from optimum.utils import NormalizedConfigManager
from transformers import AutoConfig

LOGGER = getLogger('input')


@dataclass
class InputConfig(ABC):
    name: str = MISSING


class InputGenerator(ABC):
    NAME: ClassVar[str]

    def __init__(self, model: str, task: str, device: str) -> None:
        self.model = model
        self.task = task
        self.device = device

    @abstractmethod
    def configure(self, config: InputConfig) -> None:
        raise NotImplementedError(
            'InputGenerator must implement configure method')

    @abstractmethod
    def generate(self, config: InputConfig) -> Dict[str, Tensor]:
        raise NotImplementedError(
            'InputGenerator must implement generate method')
