from typing import ClassVar, Dict, Optional
from abc import abstractmethod
from logging import getLogger
from psutil import cpu_count
from abc import ABC

from torch import Tensor
from transformers import PreTrainedModel

from src.backend.config import BackendConfig

LOGGER = getLogger('backend')


class Backend(ABC):
    NAME: ClassVar[str]

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
