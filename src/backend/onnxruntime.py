import os
import gc
from logging import getLogger
from dataclasses import dataclass
from typing import Dict, List, Optional
from tempfile import TemporaryDirectory

import torch
import onnxruntime
from transformers import GenerationMixin
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.pipelines import ORT_SUPPORTED_TASKS
from optimum.onnxruntime.configuration import (
    OptimizationConfig,
    QuantizationConfig,
    AutoOptimizationConfig,
    AutoQuantizationConfig,
)

from torch import Tensor
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from src.backend.base import Backend, BackendConfig
from src.profiler.ort_profiler import ORTProfilingWrapper
from src.utils import get_used_memory

BACKEND_NAME = "onnxruntime"
LOGGER = getLogger(BACKEND_NAME)

PROVIDER_OPTIONS = {"arena_extend_strategy": "kSameAsRequested"}


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
    auto_optimization: Optional[str] = None
    optimization_config: DictConfig = DictConfig({})

    # quantization options
    enable_quantization: bool = False
    auto_quantization: Optional[str] = None
    quantization_config: DictConfig = DictConfig({})


class ORTBackend(Backend):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)
        self.ortmodel_class = ORT_SUPPORTED_TASKS[self.task]["class"][0]

    def configure(self, config: ORTConfig) -> None:
        LOGGER.info("Configuring onnxruntime backend:")
        super().configure(config)
        self.session_options = onnxruntime.SessionOptions()
        # self.session_options.enable_cpu_mem_arena = False

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

        LOGGER.info(
            f"\t+ Loading model {self.model} with {self.ortmodel_class.__name__}"
        )
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=self.model,
            session_options=self.session_options,
            provider=config.provider,
            provider_options=PROVIDER_OPTIONS,
            use_io_binding=config.use_io_binding,
            export=True,
        )
        LOGGER.info(f"\t+ Device used memory: {get_used_memory(device=self.device)}")

        with TemporaryDirectory() as tmpdirname:
            if config.enable_optimization:
                self.optimize_model(config, tmpdirname)
                LOGGER.info(
                    f"\t+ Device used memory: {get_used_memory(device=self.device)}"
                )

            if config.enable_quantization:
                self.quantize_model(config, tmpdirname)
                LOGGER.info(
                    f"\t+ Device used memory: {get_used_memory(device=self.device)}"
                )

    def optimize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        if config.auto_optimization is not None:
            LOGGER.info(f"\t+ Setting auto optimization {config.auto_optimization}")
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=config.auto_optimization,
                for_gpu=config.optimization_config.optimize_for_gpu,
                disable_shape_inference=config.optimization_config.disable_shape_inference,
            )
        else:
            LOGGER.info("\t+ Setting optimization parameters:")
            for key, value in config.optimization_config.items():
                LOGGER.info(f"\t\t+ {key}: {value}")

            optimization_config = OptimizationConfig(
                **OmegaConf.to_container(config.optimization_config, resolve=True)
            )

        LOGGER.info("\t+ Attempting optimization")
        optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)  # type: ignore
        optimizer.optimize(
            save_dir=f"{tmpdirname}/optimized",
            optimization_config=optimization_config,
        )
        LOGGER.info("\t+ Loading optimized model")
        del self.pretrained_model
        gc.collect()
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/optimized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=PROVIDER_OPTIONS,
        )

    def quantize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        LOGGER.info("\t+ Setting quantization parameters:")
        for key, value in config.quantization_config.items():
            LOGGER.info(f"\t\t+ {key}: {value}")

        if config.auto_quantization is not None:
            quantization_config = getattr(
                AutoQuantizationConfig, config.auto_quantization
            )(
                is_static=config.quantization_config.is_static,
                per_channel=config.quantization_config.per_channel,
                operators_to_quantize=OmegaConf.to_container(
                    config.quantization_config.operators_to_quantize, resolve=True
                ),
            )

        else:
            # should be handeled by Pydantic
            if config.quantization_config.get("format", None) is not None:
                config.quantization_config["format"] = QuantFormat.from_string(
                    config.quantization_config["format"]
                )
            if config.quantization_config.get("mode", None) is not None:
                config.quantization_config["mode"] = QuantizationMode.from_string(
                    config.quantization_config["mode"]
                )
            if config.quantization_config.get("activations_dtype", None) is not None:
                config.quantization_config["activations_dtype"] = QuantType.from_string(
                    config.quantization_config["activations_dtype"]
                )
            if config.quantization_config.get("weights_dtype", None) is not None:
                config.quantization_config["weights_dtype"] = QuantType.from_string(
                    config.quantization_config["weights_dtype"]
                )

            quantization_config = QuantizationConfig(
                **OmegaConf.to_container(config.quantization_config, resolve=True)
            )

        LOGGER.info("\t+ Attempting quantization")
        model_dir = self.pretrained_model.model_save_dir
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
        gc.collect()
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/quantized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=PROVIDER_OPTIONS,
        )

    def forward(self, input: Dict[str, Tensor]):
        output = self.pretrained_model(**input)
        return output

    @property
    def is_generator(self) -> bool:
        return isinstance(self.pretrained_model, GenerationMixin)

    def generate(self, input: Dict[str, Tensor]):
        output = self.pretrained_model.generate(
            **input,
            max_new_tokens=self.pretrained_model.config.max_length,
            pad_token_id=self.pretrained_model.config.eos_token_id,
        )
        return output

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing for profiling")
        LOGGER.info("\t+ Wrapping model with profiler")
        self.pretrained_model = ORTProfilingWrapper(self.pretrained_model)

    def clean(self) -> None:
        LOGGER.info("Cleaning onnxruntime backend")
        del self.pretrained_model
        gc.collect()
        torch.cuda.empty_cache()
