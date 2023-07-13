import os
import torch
import onnxruntime
from torch import Tensor
from pathlib import Path
from logging import getLogger
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Optional
from tempfile import TemporaryDirectory
from omegaconf.dictconfig import DictConfig


from optimum.exporters import TasksManager
from optimum.pipelines import ORT_SUPPORTED_TASKS
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime.configuration import (
    OptimizationConfig,
    QuantizationConfig,
    AutoOptimizationConfig,
    AutoQuantizationConfig,
)


from src.profiler.ort_profiler import ORTProfilingWrapper
from src.backend.base import Backend, BackendConfig
from src.backend.utils import dummy_export
from src.utils import infer_device_id

OmegaConf.register_new_resolver(
    "is_gpu",
    lambda device: torch.device(device).type == "cuda",
)
OmegaConf.register_new_resolver(
    "infer_provider",
    lambda device: f"{torch.device(device).type.upper()}ExecutionProvider",
)
OmegaConf.register_new_resolver(
    "infer_device_id", lambda device: infer_device_id(device)
)

LOGGER = getLogger("onnxruntime")


@dataclass
class ORTConfig(BackendConfig):
    name: str = "onnxruntime"
    version: str = onnxruntime.__version__
    _target_: str = "src.backend.onnxruntime.ORTBackend"

    # export options
    export: bool = True
    no_weights: bool = False
    use_merged: Optional[bool] = None
    torch_dtype: Optional[str] = None

    # provider options
    provider: str = "${infer_provider:${device}}"
    device_id: Optional[int] = "${infer_device_id:${device}}"  # type: ignore

    # inference options
    use_io_binding: bool = "${is_gpu:${device}}"  # type: ignore
    enable_profiling: bool = "${benchmark.profile}"  # type: ignore

    # optimization options
    optimization: bool = False
    optimization_config: DictConfig = DictConfig(
        {
            "optimization_level": 1,  # 0, 1, 2, 99
            "optimize_for_gpu": "${is_gpu:${device}}",
            "fp16": False,
            "enable_transformers_specific_optimizations": True,
            "enable_gelu_approximation": False,
            "disable_gelu_fusion": False,
            "disable_layer_norm_fusion": False,
            "disable_attention_fusion": False,
            "disable_skip_layer_norm_fusion": True,
            "disable_bias_skip_layer_norm_fusion": False,
            "disable_bias_gelu_fusion": False,
            "use_mask_index": False,
            "no_attention_mask": False,
            "disable_embed_layer_norm_fusion": True,
            "disable_shape_inference": False,
            "use_multi_head_attention": False,
            "enable_gemm_fast_gelu_fusion": False,
            "use_raw_attention_mask": False,
            "disable_group_norm_fusion": True,
            "disable_packed_kv": True,
        }
    )

    auto_optimization: Optional[str] = None  # O1, O2, O3, O4
    auto_optimization_config: DictConfig = DictConfig(
        {
            "for_gpu": "${is_gpu:${device}}",
            # add auto optimization specific options
        }
    )

    # quantization options
    quantization: bool = False
    quantization_config: DictConfig = DictConfig(
        {
            "is_static": False,
            "format": "QOperator",  # QOperator, QDQ
            "mode": "IntegerOps",  # QLinearOps, IntegerOps
            "activations_dtype": "QUInt8",  # QInt8, QUInt8
            "activations_symmetric": False,
            "weights_dtype": "QInt8",  # QInt8, QUInt8
            "weights_symmetric": True,
            "per_channel": False,
            "reduce_range": False,
            "operators_to_quantize": [
                "MatMul",
                "Add",
            ],
        }
    )

    # arm64,avx2,avx512,avx512_vnni,tensorrt
    auto_quantization: Optional[str] = None
    auto_quantization_config: DictConfig = DictConfig(
        {
            # for now, only dynamic quantization is supported
            "is_static": False
            # add auto quantization specific options
        }
    )


class ORTBackend(Backend):
    def __init__(self, model: str, device: str, cache_kwargs: DictConfig) -> None:
        super().__init__(model, device, cache_kwargs)

        self.ortmodel_class = ORT_SUPPORTED_TASKS[self.task]["class"][0]

        LOGGER.info(
            f"\t+ Infered ORTModel class {self.ortmodel_class.__name__} "
            f"for task {self.task} and model_type {self.pretrained_config.model_type}"
        )

    def configure(self, config: ORTConfig) -> None:
        super().configure(config)

        # session options
        self.session_options = onnxruntime.SessionOptions()
        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime session intra_op_num_threads({config.intra_op_num_threads})"
            )
            self.session_options.intra_op_num_threads = config.intra_op_num_threads
        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime session inter_op_num_threads({config.inter_op_num_threads})"
            )
            self.session_options.inter_op_num_threads = config.inter_op_num_threads
        if config.enable_profiling:
            LOGGER.info("\t+ Enabling onnxruntime profiling")
            self.session_options.enable_profiling = True

        # provider options
        self.provider_options = {}
        if config.device_id is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime provider device_id({config.device_id})"
            )
            self.provider_options["device_id"] = config.device_id

        # Set torch dtype
        self.torch_dtype = (
            getattr(torch, config.torch_dtype)  # in case of torch.dtype
            if config.torch_dtype is not None and hasattr(torch, config.torch_dtype)
            else None  # in case of string or None
        )
        LOGGER.info(
            f"\t+ Using torch dtype({self.torch_dtype}) for weights loading and export"
        )

        with TemporaryDirectory() as tmpdirname:
            if config.no_weights:
                self.load_model_from_config(config, tmpdirname)
            else:
                self.load_model_from_pretrained(config)

            if (
                (config.optimization or config.auto_optimization is not None)
                and not config.use_merged
                and not config.no_weights
            ):
                self.optimize_model(config, tmpdirname)
            elif (
                (config.optimization or config.auto_optimization is not None)
                and config.use_merged
                and not config.no_weights
            ):
                raise NotImplementedError(
                    "Optimization on merged model is only supported during export (no_weights=True)"
                )
            if config.quantization or config.auto_quantization is not None:
                self.quantize_model(config, tmpdirname)

    def load_model_from_config(self, config: ORTConfig, tmpdirname: str) -> None:
        LOGGER.info(
            f"\t+ Loading model from config in {config.torch_dtype} on {self.device}"
        )

        dummy_export(
            # dummy init options
            automodel_class=self.automodel_class,
            pretrained_config=self.pretrained_config,
            # export options
            output_dir=tmpdirname,
            device=self.device,
            torch_dtype=self.torch_dtype,
            auto_optimization=config.auto_optimization,
            use_merged=config.use_merged,
            **self.cache_kwargs,
        )
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading exported model in onnxruntime")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=Path(tmpdirname),
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=self.provider_options,
            use_merged=config.use_merged,
            export=False,
            **self.cache_kwargs,
        )

    def load_model_from_pretrained(self, config: ORTConfig) -> None:
        if self.torch_dtype is not None and self.torch_dtype != torch.float32:
            raise NotImplementedError(
                "Loading from pretrained is only supported with torch_dtype float32 for now"
            )

        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=self.model,
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=self.provider_options,
            use_merged=config.use_merged,
            export=config.export,
            **self.cache_kwargs,
        )

    def optimize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        if config.auto_optimization is not None:
            LOGGER.info(f"\t+ Using auto optimization {config.auto_optimization}")
            optimization_dict = OmegaConf.to_container(
                config.auto_optimization_config, resolve=True
            )
            LOGGER.info("\t+ Setting auto optimization parameters:")
            for key, value in optimization_dict.items():  # type: ignore
                LOGGER.info(f"\t\t+ {key}: {value}")

            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=config.auto_optimization, **optimization_dict
            )
        else:
            optimization_dict = OmegaConf.to_container(
                config.optimization_config, resolve=True
            )
            LOGGER.info("\t+ Setting optimization parameters:")
            for key, value in optimization_dict.items():  # type: ignore
                LOGGER.info(f"\t\t+ {key}: {value}")
            optimization_config = OptimizationConfig(**optimization_dict)

        LOGGER.info("\t+ Attempting optimization")
        optimizer = ORTOptimizer.from_pretrained(self.pretrained_model)
        optimizer.optimize(
            save_dir=f"{tmpdirname}/optimized",
            optimization_config=optimization_config,
        )
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading optimized model")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/optimized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=self.provider_options,
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
            for key, value in quantization_dict.items():  # type: ignore
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
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading quantized model")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/quantized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=self.provider_options,
        )

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        output = self.pretrained_model(**input)[0]

        return output

    def generate(self, input: Dict[str, Tensor], new_tokens: int) -> Tensor:
        output = self.pretrained_model.generate(
            **input,
            pad_token_id=self.pretrained_model.config.eos_token_id,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            num_beams=1,
        )[0]
        return output

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing for profiling")
        LOGGER.info("\t+ Wrapping model with profiler")
        self.pretrained_model = ORTProfilingWrapper(
            self.pretrained_model
        )  # type: ignore
