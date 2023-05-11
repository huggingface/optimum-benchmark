from torch import Tensor, __version__ as torch_version
from optimum.bettertransformer import BetterTransformer
from optimum.exporters import TasksManager
from dataclasses import dataclass
from logging import getLogger
from typing import Dict
import torch

from src.backend.base import Backend, BackendConfig

BACKEND_NAME = "pytorch"

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchOptimizationConfig:
    bettertransformer: bool = False
    torch_compile: bool = False

@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = torch_version

    # base options
    disable_grad: bool = False
    eval_mode: bool = False

    # graph optimization options
    optimization: PyTorchOptimizationConfig = PyTorchOptimizationConfig()


class PyTorchBackend(Backend):
    NAME = BACKEND_NAME

    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

    def configure(self, config: PyTorchConfig) -> None:
        LOGGER.info("Configuring pytorch Backend:")
        super().configure(config)

        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting pytorch inter_op_num_threads({config.inter_op_num_threads}))"
            )
            torch.set_num_threads(config.inter_op_num_threads)

        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting pytorch intra_op_num_threads({config.intra_op_num_threads}))"
            )
            torch.set_num_interop_threads(config.intra_op_num_threads)

        # Load model
        automodel_class = TasksManager.get_model_class_for_task(self.task)
        LOGGER.info(f"\t+ Loading model {self.model} for task {self.task}")
        self.pretrained_model = automodel_class.from_pretrained(self.model)

        # Move model to device
        LOGGER.info(f"\t+ Moving Module to device {self.device}")
        self.pretrained_model.to(self.device)

        # Disable gradients
        if not config.disable_grad or config.eval_mode:
            LOGGER.info("\t+ Disabling gradients")
            torch.set_grad_enabled(False)

        # Turn on eval mode
        if config.eval_mode:
            LOGGER.info("\t+ Turning eval mode on Module")
            self.pretrained_model.eval()

        # Turn on better transformer inference
        if config.optimization.bettertransformer:
            LOGGER.info("\t+ Using BetterTransformer Fastpath")
            self.pretrained_model = BetterTransformer.transform(
                self.pretrained_model, keep_original_model=False)

        # Compile model
        if config.optimization.torch_compile:
            LOGGER.info("\t+ Using torch.compile")
            self.pretrained_model = torch.compile(self.pretrained_model)

    def run_inference(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.pretrained_model(**inputs)

    def clean(self) -> None:
        pass
