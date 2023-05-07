from typing import ClassVar, Dict, Optional
from dataclasses import dataclass, MISSING
from abc import abstractmethod
from logging import getLogger
from psutil import cpu_count
from abc import ABC

from torch import Tensor
from transformers import PreTrainedModel

LOGGER = getLogger('backend')


@dataclass
class BackendConfig(ABC):
    name: str = MISSING
    version: str = MISSING

    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None


class Backend(ABC):
    NAME: ClassVar[str]

    # every backend will have a pretrained model
    pretrained_model: Optional[PreTrainedModel] = None

    def __init__(self, model: str, task: str, device: str) -> None:
        self.model = model
        self.task = task
        self.device = device

    def configure(self, config: BackendConfig) -> None:
        if config.inter_op_num_threads is not None:
            if config.inter_op_num_threads == -1:
                config.inter_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.inter_op_num_threads to {config.inter_op_num_threads}")

        if config.intra_op_num_threads is not None:
            if config.intra_op_num_threads == -1:
                config.intra_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.intra_op_num_threads to {config.intra_op_num_threads}")

    @abstractmethod
    def run_inference(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Run inference on the backend with the given inputs
        """
        raise NotImplementedError()

    @abstractmethod
    def clean(self) -> None:
        """
        Clean the backend after execution
        """
        raise NotImplementedError()
