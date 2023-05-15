from logging import getLogger
from dataclasses import dataclass, field
from omegaconf.dictconfig import DictConfig
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.trainer import ORTFeaturesManager
from optimum.onnxruntime.configuration import OptimizationConfig,  \
    AutoOptimizationConfig, QuantizationConfig, AutoQuantizationConfig

from onnxruntime import SessionOptions, __version__ as ort_version
from torch import Tensor

from src.backend.base import Backend, BackendConfig

BACKEND_NAME = 'onnxruntime'

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = ort_version

    # basic options
    use_io_binding: bool = False
    provider: str = 'CPUExecutionProvider'
    # graph optimization options
    optimization_level: Optional[str] = None
    optimization_parameters: DictConfig = DictConfig({})
    # auto quantization options
    quantization_strategy: Optional[str] = None
    quantization_parameters: DictConfig = DictConfig({})


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
            provider=config.provider,
            export=True,
        )

        if config.optimization_level is not None:
            LOGGER.info("\t+ Optimizing model")
            optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)

            custom_opt_config = {
                key: value
                for (key, value)
                in config.optimization_parameters.items()
                if value is not None
            }

            LOGGER.info(
                f"\t+ Setting onnxruntime optimization level with "
                f"backend.optimization_level({config.optimization_level}) "
                f"and overriding optimization config with custom "
                f"backend.optimization_parameters({custom_opt_config})"
            )
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=config.optimization_level,
                **custom_opt_config
            )

            with TemporaryDirectory() as tmpdirname:
                optimizer.optimize(
                    save_dir=f'{tmpdirname}/{self.model}.onnx',
                    optimization_config=optimization_config,
                )
                self.pretrained_model = ortmodel_class.from_pretrained(
                    f'{tmpdirname}/{self.model}.onnx',
                    session_options=session_options,
                    provider=config.provider,
                )

        if config.quantization_strategy is not None:
            LOGGER.info("\t+ Quantizing model")
            quantizer = ORTQuantizer.from_pretrained(self.pretrained_model)

            custom_qnt_config = {
                key: value
                for (key, value)
                in config.quantization_parameters.items()
                if value is not None
            }

            LOGGER.info(
                f"\t+ Setting onnxruntime quantization strategy with "
                f"backend.quantization_strategy({config.quantization_strategy})"
                f"and overriding quantization config with custom "
                f"backend.quantization_parameters({custom_opt_config})"
            )

            quantization_config = AutoQuantizationConfig.__getattribute__(
                config.quantization_strategy)(**custom_qnt_config)

            with TemporaryDirectory() as tmpdirname:
                quantizer.quantize(
                    save_dir=f'{tmpdirname}/{self.model}.onnx',
                    quantization_config=quantization_config,
                )
                self.pretrained_model = ortmodel_class.from_pretrained(
                    f'{tmpdirname}/{self.model}.onnx',
                    session_options=session_options,
                    provider=config.provider,
                )

    def run_inference_with_model(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.pretrained_model(**inputs)

    def symbolic_trace_model(self, inputs: Dict[str, Tensor]) -> None:
        return super().symbolic_trace_model(inputs)

    def run_inference_with_profiler(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return super().run_inference_with_profiler(inputs)
