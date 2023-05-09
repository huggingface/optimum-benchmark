from logging import getLogger
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import List, Optional
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.trainer import ORTFeaturesManager
from optimum.onnxruntime.configuration import OptimizationConfig,  \
    AutoOptimizationConfig, QuantizationConfig, AutoQuantizationConfig

from onnxruntime import SessionOptions, __version__ as ort_version

from src.backend.base import Backend, BackendConfig

BACKEND_NAME = 'onnxruntime'

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class ORTOptimizationConfig:
    # default options from optimum docs
    optimization_level: int = 1
    optimize_for_gpu: bool = False
    fp16: bool = False
    enable_transformers_specific_optimizations: bool = True
    disable_gelu_fusion: bool = False
    disable_layer_norm_fusion: bool = False
    disable_attention_fusion: bool = False
    disable_skip_layer_norm_fusion: bool = False
    disable_bias_skip_layer_norm_fusion: bool = False
    disable_bias_gelu_fusion: bool = False
    disable_embed_layer_norm_fusion: bool = True
    enable_gelu_approximation: bool = False
    use_mask_index: bool = False
    no_attention_mask: bool = False
    disable_shape_inference: bool = False
    # still experimental
    # use_multi_head_attention: bool = False
    # enable_gemm_fast_gelu: bool = False
    # use_raw_attention_mask: bool = False
    # disable_group_norm: bool = True
    # disable_packed_kv: bool = True


@dataclass
class ORTQuantizationConfig:
    is_static: bool = False
    use_symmetric_activations: bool = False
    use_symmetric_weights: bool = True
    per_channel: bool = True
    nodes_to_quantize: Optional[List[str]] = None
    nodes_to_exclude: Optional[List[str]] = None
    operators_to_quantize: List[str] = field(
        default_factory=lambda: ['MatMul', 'Add'])


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = ort_version

    # basic options
    use_io_binding: bool = False
    # graph optimization options
    optimization_strategy: Optional[str] = None
    optimization_config: ORTOptimizationConfig = ORTOptimizationConfig()
    # auto quantization options
    auto_quantization_strategy: Optional[str] = None
    auto_quantization_config: ORTQuantizationConfig = ORTQuantizationConfig()


class ORTBackend(Backend):
    NAME = BACKEND_NAME

    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

    def configure(self, config: ORTConfig) -> None:
        LOGGER.info("Configuring onnxruntime backend:")
        super().configure(config)

        session_options = SessionOptions()
        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime intra_op_num_threads({config.intra_op_num_threads})"
            )
            session_options.intra_op_num_threads = config.intra_op_num_threads
        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime inter_op_num_threads({config.inter_op_num_threads})"
            )
            session_options.inter_op_num_threads = config.inter_op_num_threads

        try:
            # for now, I'm using ORTFeaturesManager to get the model class
            # but it only works with tasks that are supported in training
            ortmodel_class = ORTFeaturesManager.get_model_class_for_feature(
                self.task)
        except KeyError:
            raise NotImplementedError(
                f"Task {self.task} not supported by onnxruntime backend (not really)")

        LOGGER.info(f"\t+ Loading model {self.model} for task {self.task}")
        self.pretrained_model = ortmodel_class.from_pretrained(
            self.model,
            session_options=session_options,
            use_io_binding=config.use_io_binding,
            export=True,
        )

        LOGGER.info(
            f"\t+ Moving module to {self.device} execution provider")
        self.pretrained_model.to(self.device)

        LOGGER.info("\t+ Optimizing model")
        optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)

        if config.optimization_strategy is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime optimization config with strategy {config.optimization_strategy}"
            )
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=config.optimization_strategy,
                for_gpu=(self.device == 'cuda'),
            )
        else:
            LOGGER.info(
                f"\t+ Setting onnxruntime optimization config with custom config"
            )
            optimization_config = OptimizationConfig(
                **config.optimization_config,
            )

        with TemporaryDirectory() as tmpdirname:
            optimizer.optimize(
                save_dir=f'{tmpdirname}/{self.model}.onnx',
                optimization_config=optimization_config,
            )

            self.pretrained_model = ortmodel_class.from_pretrained(
                f'{tmpdirname}/{self.model}.onnx',
                session_options=session_options,
            )

        if config.auto_quantization_strategy is not None:
            LOGGER.info("\t+ Quantizing model")
            quantizer = ORTQuantizer.from_pretrained(self.pretrained_model)

            LOGGER.info(
                f"\t+ Setting onnxruntime quantization config with strategy {config.auto_quantization_strategy}"
            )
            quantization_config = getattr(AutoQuantizationConfig, config.auto_quantization_strategy)(
                **config.auto_quantization_config,
            )

            with TemporaryDirectory() as tmpdirname:
                quantizer.quantize(
                    save_dir=f'{tmpdirname}/{self.model}.onnx',
                    quantization_config=quantization_config,
                )

                self.pretrained_model = ortmodel_class.from_pretrained(
                    f'{tmpdirname}/{self.model}.onnx',
                    session_options=session_options,
                )

    def run_inference(self, inputs):
        return self.pretrained_model(**inputs)

    def clean(self) -> None:
        LOGGER.info("Cleaning onnxruntime backend:")
        pass
