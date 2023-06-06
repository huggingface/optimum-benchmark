import inspect
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

PROVIDER_OPTIONS = {}  # {"arena_extend_strategy": "kSameAsRequested"}


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = onnxruntime.__version__

    # basic options
    provider: str = "CPUExecutionProvider"
    use_io_binding: bool = False
    enable_profiling: bool = False

    # optimization options
    optimization: bool = False
    optimization_config: DictConfig = DictConfig({})

    auto_optimization: Optional[str] = None
    auto_optimization_config: DictConfig = DictConfig({})

    # quantization options
    quantization: bool = False
    quantization_config: DictConfig = DictConfig({})

    auto_quantization: Optional[str] = None
    auto_quantization_config: DictConfig = DictConfig({})


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
            if config.optimization or config.auto_optimization is not None:
                self.optimize_model(config, tmpdirname)
                LOGGER.info(
                    f"\t+ Device used memory: {get_used_memory(device=self.device)}"
                )

            if config.quantization or config.auto_quantization is not None:
                self.quantize_model(config, tmpdirname)
                LOGGER.info(
                    f"\t+ Device used memory: {get_used_memory(device=self.device)}"
                )

    def optimize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        if config.auto_optimization is not None:
            LOGGER.info(f"\t+ Using auto optimization {config.auto_optimization}")
            optimization_dict = OmegaConf.to_container(
                config.auto_optimization_config, resolve=True
            )
            LOGGER.info("\t+ Setting optimization parameters:")
            for key, value in optimization_dict.items():
                LOGGER.info(f"\t\t+ {key}: {value}")

            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=config.auto_optimization, **optimization_dict
            )
        else:
            optimization_dict = OmegaConf.to_container(
                config.optimization_config, resolve=True
            )
            LOGGER.info("\t+ Setting optimization parameters:")
            for key, value in optimization_dict.items():
                LOGGER.info(f"\t\t+ {key}: {value}")
            optimization_config = OptimizationConfig(**optimization_dict)

        LOGGER.info("\t+ Attempting optimization")
        optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)  # type: ignore
        optimizer.optimize(
            save_dir=f"{tmpdirname}/optimized",
            optimization_config=optimization_config,
        )
        del self.pretrained_model
        gc.collect()
        torch.cuda.empty_cache()

        LOGGER.info("\t+ Loading optimized model")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/optimized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=PROVIDER_OPTIONS,
        )

    def quantize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        if config.auto_quantization is not None:
            LOGGER.info(f"\t+ Using auto quantization {config.auto_quantization}")
            auto_quantization_class = getattr(
                AutoQuantizationConfig, config.auto_quantization
            )
            quantization_dict = OmegaConf.to_container(
                config.auto_quantization_config, resolve=True
            )

            LOGGER.info("\t+ Setting quantization parameters:")
            for key, value in quantization_dict.items():
                LOGGER.info(f"\t\t+ {key}: {value}")

            quantization_config = auto_quantization_class(**quantization_dict)

        else:
            quantization_dict = OmegaConf.to_container(
                config.quantization_config, resolve=True
            )
            # should be handeled by Pydantic later
            if quantization_dict.get("format", None) is not None:
                quantization_dict["format"] = QuantFormat.from_string(
                    quantization_dict["format"]
                )
            if quantization_dict.get("mode", None) is not None:
                quantization_dict["mode"] = QuantizationMode.from_string(
                    quantization_dict["mode"]
                )
            if quantization_dict.get("activations_dtype", None) is not None:
                quantization_dict["activations_dtype"] = QuantType.from_string(
                    quantization_dict["activations_dtype"]
                )
            if quantization_dict.get("weights_dtype", None) is not None:
                quantization_dict["weights_dtype"] = QuantType.from_string(
                    quantization_dict["weights_dtype"]
                )

            quantization_config = QuantizationConfig(
                **quantization_dict,
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

        del self.pretrained_model
        gc.collect()
        torch.cuda.empty_cache()

        LOGGER.info("\t+ Loading quantized model")
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

    def generate(self, input: Dict[str, Tensor], new_tokens: int) -> Tensor:
        output = self.pretrained_model.generate(
            **input,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
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
