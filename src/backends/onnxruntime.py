from dataclasses import dataclass
from logging import getLogger
from typing import Set

import onnxruntime
from tempfile import TemporaryDirectory
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from optimum.onnxruntime.trainer import ORTFeaturesManager

from src.backends.base import Backend
from src.backends.config import BackendConfig

BACKEND_NAME = 'onnxruntime'

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME

    optimization_level: int = 1
    enable_transformers_specific_optimizations: bool = True
    optimize_for_gpu: bool = False

    @staticmethod
    def version() -> str:
        return onnxruntime.__version__

    @staticmethod
    def supported_keys() -> Set[str]:
        return BackendConfig.supported_keys().union({
            'optimization_level',
            'enable_transformers_specific_optimizations',
            'optimize_for_gpu'
        })


class ORTBackend(Backend[ORTConfig]):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        super().__init__(model)
        LOGGER.info(
            f"Allocated onnxruntime backend for model: {self.model} on task: {self.task}"
        )

    def configure(self, backend_config: ORTConfig):
        LOGGER.info("Configuring onnxruntime backend:")
        super().configure(backend_config)

        session_options = onnxruntime.SessionOptions()

        if backend_config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime intra_op_num_threads({session_options.intra_op_num_threads})"
            )
            session_options.intra_op_num_threads = backend_config.intra_op_num_threads

        if backend_config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime inter_op_num_threads({session_options.inter_op_num_threads})"
            )
            session_options.inter_op_num_threads = backend_config.inter_op_num_threads

        ortmodel_class = ORTFeaturesManager.get_model_class_for_feature(
            self.task)
        LOGGER.info(f"\t+ Loading model {self.model} for task {self.task}")
        self.pretrained_model = ortmodel_class.from_pretrained(
            self.model, session_options=session_options, export=True)

        # Move model to device (can be done when loading the model)
        LOGGER.info(
            f"\t+ Moving module to {backend_config.device} execution provider")
        self.pretrained_model.to(backend_config.device)

        # Optimize model
        LOGGER.info("\t+ Optimizing model")
        optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)
        optimization_config = OptimizationConfig(
            optimization_level=backend_config.optimization_level,
            enable_transformers_specific_optimizations=backend_config.enable_transformers_specific_optimizations,
            optimize_for_gpu=backend_config.optimize_for_gpu
        )
        with TemporaryDirectory() as tmpdirname:
            optimizer.optimize(
                save_dir=f'{tmpdirname}/{self.model}.onnx',
                optimization_config=optimization_config
            )

            self.pretrained_model = ortmodel_class.from_pretrained(
                f'{tmpdirname}/{self.model}.onnx',
                session_options=session_options,
            )
