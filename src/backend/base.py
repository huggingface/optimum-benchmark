from dataclasses import dataclass, MISSING
from typing import Dict, List, Optional
from abc import abstractmethod, ABC
from logging import getLogger

from torch import Tensor
from psutil import cpu_count

LOGGER = getLogger("backend")


@dataclass
class BackendConfig(ABC):
    name: str = MISSING  # type: ignore
    version: str = MISSING  # type: ignore
    _target_: str = MISSING  # type: ignore

    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None


class Backend(ABC):
    def __init__(self, model: str, task: str, device: str, model_kwargs: dict):
        self.model = model
        self.task = task
        self.device = device
        self.model_kwargs = model_kwargs

    @abstractmethod
    def configure(self, config: BackendConfig) -> None:
        LOGGER.info(f"Configuring {config.name} backend")
        if config.inter_op_num_threads is not None:
            if config.inter_op_num_threads == -1:
                config.inter_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.inter_op_num_threads to cpu_count({config.inter_op_num_threads})"
                )

        if config.intra_op_num_threads is not None:
            if config.intra_op_num_threads == -1:
                config.intra_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.intra_op_num_threads to cpu_count({config.intra_op_num_threads})"
                )

    @abstractmethod
    def forward(self, input: Dict[str, Tensor]):
        raise NotImplementedError("Backend must implement forward method")

    @abstractmethod
    def generate(self, input: Dict[str, Tensor], prefix_length: int) -> str:
        raise NotImplementedError("Backend must implement generate method")

    @abstractmethod
    def prepare_for_profiling(self, input_names: List[str]) -> None:
        raise NotImplementedError("Backend must implement prepare_for_profiling method")

    @abstractmethod
    def clean(self) -> None:
        raise NotImplementedError("Backend must implement clean method")
