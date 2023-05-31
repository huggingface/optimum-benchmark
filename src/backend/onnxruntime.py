import os
import gc
from typing import Dict, List
from logging import getLogger
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from omegaconf import ListConfig

import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.pipelines import ORT_SUPPORTED_TASKS
from optimum.onnxruntime.configuration import (
    OptimizationConfig,
    QuantizationConfig,
    # AutoOptimizationConfig,
    # AutoQuantizationConfig,
)

from omegaconf.dictconfig import DictConfig
from torch import Tensor

from src.backend.base import Backend, BackendConfig
from src.profilers import ORTProfilingWrapper

BACKEND_NAME = "onnxruntime"
LOGGER = getLogger(BACKEND_NAME)


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = onnxruntime.__version__

    # basic options
    provider: str = "CPUExecutionProvider"
    use_io_binding: bool = False
    enable_profiling: bool = False

    # optimization options
    enable_optimization: bool = False
    optimization: DictConfig = DictConfig({})

    # quantization options
    enable_quantization: bool = False
    quantization: DictConfig = DictConfig({})


class ORTBackend(Backend):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)
        self.ortmodel_class = ORT_SUPPORTED_TASKS[self.task]["class"][0]

    def configure(self, config: ORTConfig) -> None:
        LOGGER.info("Configuring onnxruntime backend")
        super().configure(config)

        self.session_options = onnxruntime.SessionOptions()

        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime intra_op_num_threads({config.intra_op_num_threads})"
            )
            self.session_options.intra_op_num_threads = config.intra_op_num_threads

        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime inter_op_num_threads({config.inter_op_num_threads})"
            )
            self.session_options.inter_op_num_threads = config.inter_op_num_threads

        if config.enable_profiling:
            LOGGER.info("\t+ Enabling onnxruntime profiling")
            self.session_options.enable_profiling = True

        LOGGER.info(f"Loading model {self.model} with {self.ortmodel_class.__name__}")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=self.model,
            session_options=self.session_options,
            provider=config.provider,
            use_io_binding=config.use_io_binding,
            export=True,
        )

        with TemporaryDirectory() as tmpdirname:
            if config.enable_optimization:
                self.optimize_model(config, tmpdirname)

            if config.enable_quantization:
                self.quantize_model(config, tmpdirname)

    def optimize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        LOGGER.info("\t+ Setting onnxruntime optimization parameters:")
        for key, value in config.optimization.items():
            LOGGER.info(f"\t+ {key}: {value}")

        optimization_config = OptimizationConfig(**config.optimization)  # type: ignore

        LOGGER.info("\t+ Attempting optimization")
        optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)  # type: ignore
        optimizer.optimize(
            save_dir=f"{tmpdirname}/optimized",
            optimization_config=optimization_config,
        )
        LOGGER.info("\t+ Loading optimized model")
        del self.pretrained_model
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/optimized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
        )

    def quantize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        LOGGER.info("\t+ Setting onnxruntime quantization parameters:")
        for key, value in config.quantization.items():
            LOGGER.info(f"\t+ {key}: {value}")

        # should be handeled by Pydantic
        if config.quantization.get("format", None) is not None:
            config.quantization["format"] = QuantFormat.from_string(
                config.quantization["format"]
            )
        if config.quantization.get("mode", None) is not None:
            config.quantization["mode"] = QuantizationMode.from_string(
                config.quantization["mode"]
            )
        if config.quantization.get("activations_dtype", None) is not None:
            config.quantization["activations_dtype"] = QuantType.from_string(
                config.quantization["activations_dtype"]
            )
        if config.quantization.get("weights_dtype", None) is not None:
            config.quantization["weights_dtype"] = QuantType.from_string(
                config.quantization["weights_dtype"]
            )
        if config.quantization.get("operators_to_quantize", None) is not None:
            config.quantization["operators_to_quantize"] = list(
                config.quantization["operators_to_quantize"]
            )
            print(type(config.quantization["operators_to_quantize"]))

        quantization_config = QuantizationConfig(
            **{
                key: list(value) if type(value) == ListConfig else value
                for key, value in config.quantization.items()
            }  # type: ignore
        )

        LOGGER.info("\t+ Attempting quantization")
        model_dir = self.pretrained_model.model_save_dir  # type: ignore
        components = [file for file in os.listdir(model_dir) if file.endswith(".onnx")]
        for component in components:
            LOGGER.info(f"\t+ Quantizing {component}")
            quantizer = ORTQuantizer.from_pretrained(model_dir, file_name=component)
            quantizer.quantize(
                save_dir=f"{tmpdirname}/quantized",
                quantization_config=quantization_config,
            )

        LOGGER.info("\t+ Loading quantized model")
        del self.pretrained_model
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/quantized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
        )

    def forward(self, input: Dict[str, Tensor]):
        output = self.pretrained_model(**input)
        return output

    def generate(self, input: Dict[str, Tensor]):
        output = self.pretrained_model.generate(**input) # type: ignore
        return output

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing for profiling")
        LOGGER.info("\t+ Wrapping model with profiler")
        self.pretrained_model = ORTProfilingWrapper(self.pretrained_model)  # type: ignore

    def clean(self) -> None:
        LOGGER.info("Cleaning onnxruntime backend")
        del self.pretrained_model
        gc.collect()
