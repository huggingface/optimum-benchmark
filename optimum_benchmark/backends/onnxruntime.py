import os
import torch
import onnxruntime
from torch import Tensor
from pathlib import Path
from logging import getLogger
from omegaconf import OmegaConf
from dataclasses import dataclass
from hydra.utils import get_class
from typing import Dict, List, Optional
from tempfile import TemporaryDirectory
from omegaconf.dictconfig import DictConfig

from transformers import GenerationMixin
from optimum.pipelines import ORT_SUPPORTED_TASKS
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime.configuration import (
    OptimizationConfig,
    QuantizationConfig,
    AutoCalibrationConfig,
    AutoOptimizationConfig,
    AutoQuantizationConfig,
)


from optimum_benchmark.profilers.ort_profiler import ORTProfilingWrapper
from optimum_benchmark.backends.base import Backend, BackendConfig
from optimum_benchmark.backends.utils import export_dummy_model
from optimum_benchmark.utils import infer_device_id

OmegaConf.register_new_resolver(
    "is_gpu",
    lambda device: torch.device(device).type == "cuda",
)
OmegaConf.register_new_resolver(
    "infer_execution_provider",
    lambda device: f"{torch.device(device).type.upper()}ExecutionProvider",
)
OmegaConf.register_new_resolver(
    "infer_device_id",
    lambda device: infer_device_id(device),
)
OmegaConf.register_new_resolver(
    "perform_calibration",
    lambda *static_quants: any(static_quants),
)

LOGGER = getLogger("onnxruntime")


@dataclass
class ORTConfig(BackendConfig):
    name: str = "onnxruntime"
    version: str = onnxruntime.__version__
    _target_: str = "optimum_benchmark.backends.onnxruntime.ORTBackend"

    # export options
    export: bool = True
    no_weights: bool = False
    use_merged: Optional[bool] = None
    torch_dtype: Optional[str] = None

    # provider options
    provider: str = "${infer_execution_provider:${device}}"
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

    # O1, O2, O3, O4
    auto_optimization: Optional[str] = None
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

    # calibration options
    calibration: bool = "${perform_calibration:${backend.auto_quantization_config.is_static}, ${backend.quantization_config.is_static}}"
    calibration_config: DictConfig = DictConfig(
        {
            "dataset_name": "glue",
            "num_samples": 300,
            "dataset_config_name": "sst2",
            "dataset_split": "train",
            "preprocess_batch": True,
            "preprocess_class": "optimum_benchmark.preprocessors.glue.GluePreprocessor",
        }
    )


class ORTBackend(Backend):
    def __init__(
        self, model: str, task: str, device: str, hub_kwargs: DictConfig
    ) -> None:
        super().__init__(model, task, device, hub_kwargs)

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
                    "Optimization on merged model is only supported during export (no_weights=True) for now."
                )
            if config.quantization or config.auto_quantization is not None:
                self.quantize_model(config, tmpdirname)

    def load_model_from_config(self, config: ORTConfig, tmpdirname: str) -> None:
        LOGGER.info(
            f"\t+ Loading model from config in {config.torch_dtype} on {self.device}"
        )

        export_dummy_model(
            # dummy init options
            automodel_class=self.automodel_class,
            pretrained_config=self.pretrained_config,
            # export options
            output_dir=tmpdirname,
            device=self.device,
            torch_dtype=self.torch_dtype,
            auto_optimization=config.auto_optimization,
            use_merged=config.use_merged,
            **self.hub_kwargs,
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
            **self.hub_kwargs,
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
            export=config.export,
            # TODO: is this the right way to do it?
            **({"use_merged": config.use_merged} if self.can_be_merged() else {}),
            **self.hub_kwargs,
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

            if config.calibration:
                preprocess_class = get_class(config.calibration_config.preprocess_class)
                preprocess_function = preprocess_class(model_name_or_path=self.model)

                calibration_dataset = quantizer.get_calibration_dataset(
                    dataset_name=config.calibration_config.dataset_name,
                    num_samples=config.calibration_config.num_samples,
                    dataset_config_name=config.calibration_config.dataset_config_name,
                    dataset_split=config.calibration_config.dataset_split,
                    preprocess_function=preprocess_function,
                )

                # Create the calibration configuration containing the parameters related to calibration.
                calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

                # Perform the calibration step: computes the activations quantization ranges
                calibration_tensors_range = quantizer.fit(
                    dataset=calibration_dataset,
                    calibration_config=calibration_config,
                    operators_to_quantize=quantization_config.operators_to_quantize,
                )

            quantizer.quantize(
                save_dir=f"{tmpdirname}/quantized",
                calibration_tensors_range=calibration_tensors_range,
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
            pad_token_id=0,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            num_beams=1,
        )[0]
        return output

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing model for profiling")
        LOGGER.info("\t+ Wrapping model inside profiler")
        self.pretrained_model = ORTProfilingWrapper(self.pretrained_model)

    def can_be_merged(self) -> bool:
        # TODO: check if this is the right way to do it
        return issubclass(self.ortmodel_class, GenerationMixin)
