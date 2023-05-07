from logging import getLogger
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.trainer import ORTFeaturesManager
from optimum.onnxruntime.configuration import OptimizationConfig
from onnxruntime import SessionOptions, __version__ as ort_version

from src.backend.base import Backend, BackendConfig

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
    version: str = ort_version

    # basic options
    use_io_binding: bool = False
    # graph optimization options
    optimization: ORTOptimizationConfig = ORTOptimizationConfig()


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

    def run_inference(self, inputs):
        return self.pretrained_model(**inputs)

    def clean(self) -> None:
        LOGGER.info("Cleaning onnxruntime backend:")
        pass
