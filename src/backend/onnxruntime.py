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
    pass


@dataclass
class ORTQuantizationConfig:
    pass


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = ort_version

    # basic options
    use_io_binding: bool = False
    # graph optimization options
    auto_optimization_level: Optional[str] = None
    auto_optimization_config: ORTOptimizationConfig = ORTOptimizationConfig()
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

        custom_opt_config = {
            key: value
            for (key, value)
            in config.auto_optimization_config.items()
            if value is not None
        }

        if (config.auto_optimization_level is not None) or custom_opt_config:
            LOGGER.info("\t+ Optimizing model")
            optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)

            if self.device == 'cuda' or config.auto_optimization_config.get('optimize_for_gpu'):
                LOGGER.info(
                    f"\t+ Enabling onnxruntime optimization for gpu (optimize_for_gpu=True)"
                )
                config.auto_optimization_config['optimize_for_gpu'] = True
            else:
                LOGGER.info(
                    f"\t+ Disabling onnxruntime optimization for gpu (optimize_for_gpu=False)"
                )
                config.auto_optimization_config['optimize_for_gpu'] = False

            if 'optimize_for_gpu' in custom_opt_config:
                custom_opt_config['optimize_for_gpu'] = config.auto_optimization_config['optimize_for_gpu']

            if config.auto_optimization_level is not None:
                LOGGER.info(
                    f"\t+ Setting onnxruntime optimization level with "
                    f"backend.auto_optimization_level({config.auto_optimization_level}) "
                    f"and overriding optimization config with custom "
                    f"backend.auto_optimization_config({custom_opt_config})"
                )
                optimization_config = AutoOptimizationConfig.with_optimization_level(
                    optimization_level=config.auto_optimization_level,
                    for_gpu=config.auto_optimization_config['optimize_for_gpu'],
                    **custom_opt_config
                )
            elif custom_opt_config:
                LOGGER.info(
                    f"\t+ Setting onnxruntime optimization config with custom "
                    f"backend.auto_optimization_config({custom_opt_config})"
                )
                optimization_config = OptimizationConfig(
                    **custom_opt_config,
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

        custom_qnt_config = {
            key: value
            for (key, value)
            in config.auto_quantization_config.items()
            if value is not None
        }

        if (config.auto_quantization_strategy is not None) and custom_qnt_config:
            LOGGER.info("\t+ Quantizing model")
            quantizer = ORTQuantizer.from_pretrained(self.pretrained_model)

            LOGGER.info(
                f"\t+ Setting onnxruntime quantization strategy with "
                f"backend.auto_quantization_strategy({config.auto_quantization_strategy})"
                f"and overriding quantization config with custom "
                f"backend.auto_quantization_config({custom_opt_config})"
            )
            quantization_config = AutoQuantizationConfig.__getattribute__(
                config.auto_quantization_strategy)(**custom_qnt_config)

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
