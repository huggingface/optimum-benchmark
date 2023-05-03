from dataclasses import dataclass
from logging import getLogger
from typing import Set

import torch
from optimum.exporters import TasksManager
from optimum.bettertransformer import BetterTransformer

from src.backend.base import Backend
from src.backend.config import BackendConfig

BACKEND_NAME = "pytorch"

LOGGER = getLogger(BACKEND_NAME)

@dataclass
class PyTorchOptimizationConfig:
    bettertransformer: bool = False
    torch_compile: bool = False
    grad_enabled: bool = False
    eval_mode: bool = False

@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME
    optimization: PyTorchOptimizationConfig = PyTorchOptimizationConfig()

    @staticmethod
    def version() -> str:
        return torch.__version__

    @staticmethod
    def supported_keys() -> Set[str]:
        return BackendConfig.supported_keys().union(
            set(PyTorchOptimizationConfig.__dataclass_fields__.keys())
        )


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        super().__init__(model)
        LOGGER.info(
            f"Allocated pytorch backend for model: {self.model} on task: {self.task}"
        )

    def configure(self, config: PyTorchConfig):
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
        LOGGER.info(f"\t+ Moving Module to device {config.device}")
        self.pretrained_model.to(config.device)

        # Disable gradients
        if not config.optimization.grad_enabled:
            LOGGER.info("\t+ Disabling gradients")
            torch.set_grad_enabled(False)

        # Turn on eval mode
        if config.optimization.eval_mode:
            LOGGER.info("\t+ Turning eval mode on Module")
            self.pretrained_model.eval()

        # Turn on better transformer inference
        if config.optimization.bettertransformer:
            LOGGER.info("\t+ Using BetterTransformer fastpath")
            self.pretrained_model = BetterTransformer.transform(
                self.pretrained_model, keep_original_model=False)

        # Compile model
        if config.optimization.torch_compile:
            LOGGER.info("\t+ Using compiled Module")
            self.pretrained_model = torch.compile(self.pretrained_model)