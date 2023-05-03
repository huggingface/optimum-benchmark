from dataclasses import dataclass
from logging import getLogger
from typing import Set

import onnxruntime
from tempfile import TemporaryDirectory
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from optimum.onnxruntime.trainer import ORTFeaturesManager

from src.backend.base import Backend
from src.backend.config import BackendConfig

BACKEND_NAME = 'onnxruntime'

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class ORTOptimizationConfig:
    optimization_level: int = 1
    optimize_for_gpu: bool = False
    fp16: bool = False
    enable_transformers_specific_optimizations: bool = False
    disable_gelu_fusion: bool = False
    disable_layer_norm_fusion: bool = False
    disable_attention_fusion: bool = False
    disable_skip_layer_norm_fusion: bool = False
    disable_bias_skip_layer_norm_fusion: bool = False
    disable_bias_gelu_fusion: bool = False
    disable_embed_layer_norm_fusion: bool = False
    enable_gelu_approximation: bool = False
    use_mask_index: bool = False
    no_attention_mask: bool = False
    disable_shape_inference: bool = False
    # still experimental
    # use_multi_head_attention: bool = False
    # enable_gemm_fast_gelu: bool = False
    # use_raw_attention_mask: bool = False
    # disable_group_norm: bool = False
    # disable_packed_kv: bool = False


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME
    optimization: ORTOptimizationConfig = ORTOptimizationConfig()

    @staticmethod
    def version() -> str:
        return onnxruntime.__version__

    @staticmethod
    def supported_keys() -> Set[str]:
        return BackendConfig.supported_keys().union(
            set(ORTOptimizationConfig.__dataclass_fields__.keys())
        )


class ORTBackend(Backend[ORTConfig]):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        super().__init__(model)
        LOGGER.info(
            f"Allocated onnxruntime backend for model: {self.model} on task: {self.task}"
        )

    def configure(self, config: ORTConfig):
        LOGGER.info("Configuring onnxruntime backend:")
        super().configure(config)

        print(type(config.optimization))

        session_options = onnxruntime.SessionOptions()

        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime intra_op_num_threads({session_options.intra_op_num_threads})"
            )
            session_options.intra_op_num_threads = config.intra_op_num_threads

        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime inter_op_num_threads({session_options.inter_op_num_threads})"
            )
            session_options.inter_op_num_threads = config.inter_op_num_threads

        ortmodel_class = ORTFeaturesManager.get_model_class_for_feature(
            self.task)
        LOGGER.info(f"\t+ Loading model {self.model} for task {self.task}")
        self.pretrained_model = ortmodel_class.from_pretrained(
            self.model, session_options=session_options, export=True)

        # Move model to device (can be done when loading the model)
        LOGGER.info(
            f"\t+ Moving module to {config.device} execution provider")
        self.pretrained_model.to(config.device)

        # Optimize model
        LOGGER.info("\t+ Optimizing model")
        optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)
        with TemporaryDirectory() as tmpdirname:
            optimizer.optimize(
                save_dir=f'{tmpdirname}/{self.model}.onnx',
                optimization_config=OptimizationConfig(
                    **config.optimization),
            )

            self.pretrained_model = ortmodel_class.from_pretrained(
                f'{tmpdirname}/{self.model}.onnx',
                session_options=session_options,
            )
