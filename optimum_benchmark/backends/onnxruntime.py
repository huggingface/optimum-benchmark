import os
import torch
from torch import Tensor
from datasets import Dataset
from logging import getLogger
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from hydra.utils import get_class
from tempfile import TemporaryDirectory
from omegaconf.dictconfig import DictConfig
from typing import Any, Callable, Dict, List, Optional


try:
    from onnxruntime import __version__ as onnxruntime_version
except ImportError:
    onnxruntime_version = "Not installed"

from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import (
    OptimizationConfig,
    QuantizationConfig,
    AutoCalibrationConfig,
    AutoOptimizationConfig,
    AutoQuantizationConfig,
)


from optimum_benchmark.backends.base import Backend, BackendConfig
from optimum_benchmark.backends.utils import main_export, randomize_weights
from optimum_benchmark.profilers.ort_profiler import ORTProfilingWrapper
from optimum_benchmark.utils import infer_device_id

OmegaConf.register_new_resolver(
    "is_gpu",
    lambda device: torch.device(device).type == "cuda",
)
OmegaConf.register_new_resolver(
    "is_profiling",
    lambda benchmark_name: benchmark_name == "profiling",
)
OmegaConf.register_new_resolver(
    "infer_provider",
    lambda device: f"{torch.device(device).type.upper()}ExecutionProvider",
)
OmegaConf.register_new_resolver(
    "infer_device_id",
    lambda device: infer_device_id(device),
)
OmegaConf.register_new_resolver(
    "requires_calibration",
    lambda *static_quants: any(static_quants),
)

LOGGER = getLogger("onnxruntime")


@dataclass
class ORTConfig(BackendConfig):
    name: str = "onnxruntime"
    version: str = onnxruntime_version
    _target_: str = "optimum_benchmark.backends.onnxruntime.ORTBackend"

    # export options
    export: bool = True
    no_weights: bool = False
    use_merged: bool = False
    use_cache: bool = True
    torch_dtype: Optional[str] = None

    # provider options
    provider: str = "${infer_provider:${device}}"
    device_id: Optional[int] = "${infer_device_id:${device}}"

    # inference options
    use_io_binding: bool = "${is_gpu:${device}}"
    enable_profiling: bool = "${is_profiling:${benchmark.name}}"

    # optimization options
    optimization: bool = False
    optimization_config: Dict = field(default_factory=lambda: {
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
    auto_optimization_config: Dict = field(default_factory=lambda: {
            "for_gpu": "${is_gpu:${device}}",
            # add auto optimization specific options in config file or cli
            # using +backend.auto_optimization_config.option_name: value
        }
    )

    # quantization options
    quantization: bool = False
    quantization_config: Dict = field(default_factory=lambda: {
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
    auto_quantization_config: Dict = field(default_factory=lambda: {
            "is_static": False
            # add auto quantization specific options in config file or cli
            # using +backend.auto_quantization_config.option_name: value
        }
    )

    # calibration options
    calibration: bool = "${requires_calibration:${backend.auto_quantization_config.is_static}, ${backend.quantization_config.is_static}}"
    calibration_config: Dict = field(default_factory=lambda: {
            "dataset_name": "glue",
            "num_samples": 300,
            "dataset_config_name": "sst2",
            "dataset_split": "train",
            "preprocess_batch": True,
            "preprocess_class": "optimum_benchmark.preprocessors.glue.GluePreprocessor",
        }
    )

    # this will skip exporting the model and will use automodel instead
    use_ortmodel: bool = "${is_inference:${benchmark.name}}"


class ORTBackend(Backend):
    def __init__(
        self, model: str, task: str, device: str, hub_kwargs: DictConfig
    ) -> None:
        super().__init__(model, task, device, hub_kwargs)

        from optimum.pipelines import ORT_SUPPORTED_TASKS

        if self.task == "stable-diffusion":
            self.ortmodel_class = get_class(
                "optimum.onnxruntime.ORTStableDiffusionPipeline"
            )
        elif self.task == "stable-diffusion-xl":
            self.ortmodel_class = get_class(
                "optimum.onnxruntime.ORTStableDiffusionXLPipeline"
            )
        elif self.task in ORT_SUPPORTED_TASKS:
            self.ortmodel_class = ORT_SUPPORTED_TASKS[self.task]["class"][0]
        else:
            raise ValueError(f"Unsupported task {self.task}")

        LOGGER.info(
            f"\t+ Infered ORTModel class {self.ortmodel_class.__name__} "
            f"for task {self.task} and model_type {self.model_type}"
        )

    def configure(self, config: ORTConfig) -> None:
        super().configure(config)

        import onnxruntime

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
            else config.torch_dtype
        )
        LOGGER.info(
            f"\t+ Using torch dtype({self.torch_dtype}) for weights loading and export"
        )

        with TemporaryDirectory() as tmpdirname:
            if config.use_ortmodel:
                if config.no_weights:
                    self.load_ortmodel_from_config(config, tmpdirname)
                else:
                    self.load_ortmodel_from_pretrained(config, tmpdirname)
            else:
                if config.no_weights:
                    self.load_automodel_from_config(config)
                else:
                    self.load_automodel_from_pretrained(config)

    def load_ortmodel_from_config(self, config: ORTConfig, tmpdirname: str) -> None:
        LOGGER.info(
            f"\t+ Loading model from config in {config.torch_dtype} on {self.device}"
        )

        self.load_automodel_from_config(config)
        main_export(
            model_name_or_path=self.model,
            output=f"{tmpdirname}/exported_model",
            task=self.task + "-with-past"
            if self.can_generate() and config.use_cache
            else self.task,
            device=self.device.type,
            fp16=self.torch_dtype == torch.float16,
            optimize=config.auto_optimization,
            no_post_process=not config.use_merged,
            for_ort=True,
            do_validation=False,
            **self.hub_kwargs,
            # we hijack the model instantiation and use our random weights model
            model=self.pretrained_model,
        )
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading exported model in onnxruntime")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/exported_model",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=self.provider_options,
            **(
                {
                    "use_merged": config.use_merged,
                    "use_cache": config.use_cache,
                }
                if self.can_generate()
                else {}
            ),
            export=False,
            **self.hub_kwargs,
        )

        if config.optimization:
            raise NotImplementedError(
                "Only AutoOptimization is supported when loading a model with random weights"
            )

        if config.quantization or config.auto_quantization is not None:
            self.quantize(config, tmpdirname)

    def load_ortmodel_from_pretrained(self, config: ORTConfig, tmpdirname: str) -> None:
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
            **(
                {
                    "use_merged": config.use_merged,
                    "use_cache": config.use_cache,
                }
                if self.can_generate()
                else {}
            ),
            **self.hub_kwargs,
        )

        if config.optimization or config.auto_optimization is not None:
            self.optimize(config, tmpdirname)

        if config.quantization or config.auto_quantization is not None:
            self.quantize(config, tmpdirname)

    def optimize(self, config: ORTConfig, tmpdirname: str) -> None:
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

    def quantize(self, config: ORTConfig, tmpdirname: str) -> None:
        if config.auto_quantization is not None:
            LOGGER.info(
                f"\t+ Using auto quantization {config.auto_quantization} and its config"
            )
            auto_quantization_config_class = getattr(
                AutoQuantizationConfig, config.auto_quantization
            )
            quantization_dict = OmegaConf.to_container(
                config.auto_quantization_config, resolve=True
            )
            quantization_dict = format_ort_quantization_dict(quantization_dict)
            quantization_config = auto_quantization_config_class(**quantization_dict)

        else:
            LOGGER.info("\t+ Using manual quantization and its config")
            from optimum_benchmark.backends.utils import format_ort_quantization_dict

            quantization_dict = OmegaConf.to_container(
                config.quantization_config, resolve=True
            )
            quantization_dict = format_ort_quantization_dict(quantization_dict)
            quantization_config = QuantizationConfig(**quantization_dict)

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

    def load_automodel_from_config(self, config: ORTConfig) -> None:
        from accelerate import init_empty_weights

        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                config=self.pretrained_config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.hub_kwargs.get("trust_remote_code", False),
            )
        self.pretrained_model.to_empty(device=self.device)
        randomize_weights(self.pretrained_model)

    def load_automodel_from_pretrained(self, config: ORTConfig) -> None:
        with self.device:
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                torch_dtype=self.torch_dtype,
                **self.hub_kwargs,
            )

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing model for profiling")
        LOGGER.info("\t+ Wrapping model inside profiler")
        self.pretrained_model = ORTProfilingWrapper(self.pretrained_model)

    def prepare_for_training(
        self,
        training_dataset: Dataset,
        training_data_collator: Callable,
        training_arguments: Dict[str, Any],
    ) -> None:
        LOGGER.info("Preparing model for training")
        LOGGER.info("\t+ Wrapping model inside trainer")

        from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments

        training_arguments = ORTTrainingArguments(**training_arguments)
        self.trainer = ORTTrainer(
            model=self.pretrained_model,
            args=training_arguments,
            train_dataset=training_dataset,
            data_collator=training_data_collator,
            feature=self.task,
        )

    def forward(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
        output = self.pretrained_model(**input, **kwargs)[0]

        return output

    def generate(self, input: Dict[str, Tensor], **kwargs) -> Tensor:
        output = self.pretrained_model.generate(**input, **kwargs)[0]
        return output

    def train(self) -> None:
        LOGGER.info("Training model")
        results = self.trainer.train()

        return results
