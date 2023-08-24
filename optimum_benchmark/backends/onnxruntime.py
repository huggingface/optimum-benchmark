from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from logging import getLogger
from datasets import Dataset
import os


import torch
from torch import Tensor
from omegaconf import OmegaConf
from hydra.utils import get_class
from onnxruntime import SessionOptions
from accelerate import init_empty_weights
from optimum.pipelines import ORT_SUPPORTED_TASKS
from onnxruntime import __version__ as onnxruntime_version
from optimum.onnxruntime.configuration import (
    OptimizationConfig,
    QuantizationConfig,
    AutoCalibrationConfig,
    AutoOptimizationConfig,
    AutoQuantizationConfig,
)
from optimum.onnxruntime import (
    ORTOptimizer,
    ORTQuantizer,
    ORTTrainer,
    ORTTrainingArguments,
)

if TYPE_CHECKING:
    from transformers import TrainerCallback, TrainerState
    from transformers.modeling_outputs import ModelOutput


from .base import Backend, BackendConfig
from .utils.optimum_utils import main_export
from .utils.pytorch_utils import randomize_weights
from ..profilers.ort_profiler import ORTProfilingWrapper
from .utils.onnxruntime_utils import (
    format_ort_quantization_dict,
    infer_device_id,
    DEFAULT_OPTIMIZATION_CONFIG,
    DEFAULT_QUANTIZATION_CONFIG,
    DEFAULT_CALIBRATION_CONFIG,
)


OmegaConf.register_new_resolver(
    "is_gpu",
    lambda device: "cuda" in device.lower() or "tensorrt" in device.lower(),
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
    provider_options: Optional[Dict] = None
    # TODO: deprecate device_id in favor of provider_options
    device_id: Optional[int] = "${infer_device_id:${device}}"

    # inference options
    use_io_binding: bool = "${is_gpu:${device}}"
    session_options: Optional[Dict] = None
    # TODO: deprecate enable_profiling in favor of session_options
    enable_profiling: bool = "${is_profiling:${benchmark.name}}"

    # optimization options
    optimization: bool = False
    optimization_config: Optional[Dict] = None

    # O1, O2, O3, O4
    auto_optimization: Optional[str] = None
    auto_optimization_config: Optional[Dict] = None

    # quantization options
    quantization: bool = False
    quantization_config: Optional[Dict] = None

    # arm64,avx2,avx512,avx512_vnni,tensorrt
    auto_quantization: Optional[str] = None
    auto_quantization_config: Optional[Dict] = None

    # calibration options
    calibration: bool = False
    calibration_config: Optional[Dict] = None

    # this will skip exporting the model and will use automodel with trainer
    use_ortmodel: bool = "${is_inference:${benchmark.name}}"

    def __post_init__(self):
        if self.optimization:
            self.optimization_config = OmegaConf.merge(
                self.optimization_config or {},
                DEFAULT_OPTIMIZATION_CONFIG,
            )

        if self.auto_optimization is not None:
            self.auto_optimization_config = OmegaConf.merge(
                self.auto_optimization_config or {},
                DEFAULT_OPTIMIZATION_CONFIG,
            )
            self.auto_optimization_config.pop("optimization_level", None)
            self.auto_optimization_config[
                "for_gpu"
            ] = self.auto_optimization_config.pop("optimize_for_gpu")

        if self.quantization:
            self.quantization_config = OmegaConf.merge(
                self.quantization_config or {},
                DEFAULT_QUANTIZATION_CONFIG,
            )

        # auto quantization is needs specific config for each type
        # if self.auto_quantization is not None:
        #     self.auto_quantization_config = OmegaConf.merge(
        #         self.auto_quantization_config or {},
        #         DEFAULT_QUANTIZATION_CONFIG,
        #     )

        if self.quantization_config is not None:
            self.calibration = self.quantization_config["is_static"]

        if self.auto_quantization_config is not None:
            self.calibration = self.auto_quantization_config["is_static"]

        if self.calibration:
            self.calibration_config = OmegaConf.merge(
                self.calibration_config or {},
                DEFAULT_CALIBRATION_CONFIG,
            )

        if self.device_id is not None:
            LOGGER.warning(
                "device_id is deprecated, please use provider_options instead"
            )
            self.provider_options = OmegaConf.merge(
                self.provider_options or {},
                {"device_id": self.device_id},
            )

        if self.enable_profiling is not None:
            LOGGER.warning(
                "enable_profiling is deprecated, please use session_options instead"
            )
            self.session_options = OmegaConf.merge(
                self.session_options or {},
                {"enable_profiling": self.enable_profiling},
            )


class ORTBackend(Backend):
    name: str = "onnxruntime"
    config: ORTConfig

    def __init__(
        self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(model, task, device, hub_kwargs)
        self.device = torch.device(device)

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

        # session options
        session_options = SessionOptions()
        if self.config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting intra_op_num_threads({config.intra_op_num_threads})"
            )
            self.config.session_options.intra_op_num_threads = (
                self.config.intra_op_num_threads
            )
        if self.config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting inter_op_num_threads({config.inter_op_num_threads})"
            )
            self.config.session_options.inter_op_num_threads = (
                self.config.inter_op_num_threads
            )
        for key, value in self.config.session_options.items():
            setattr(session_options, key, value)
        self.config.session_options = session_options

        # Set torch dtype
        self.config.torch_dtype = (
            getattr(torch, self.config.torch_dtype)  # in case of torch.dtype
            if self.config.torch_dtype is not None
            and hasattr(torch, self.config.torch_dtype)
            else self.config.torch_dtype
        )

        with TemporaryDirectory() as tmpdirname:
            if self.config.use_ortmodel:
                if self.config.no_weights:
                    self.load_ortmodel_from_config(tmpdirname)
                else:
                    self.load_ortmodel_from_pretrained(tmpdirname)
            else:
                if self.config.no_weights:
                    self.load_automodel_from_config()
                else:
                    self.load_automodel_from_pretrained()

    def load_ortmodel_from_config(self, tmpdirname: str) -> None:
        LOGGER.info("\t+ Creating random weights model")
        self.load_automodel_from_config()

        LOGGER.info("\t+ Exporting model to onnx")
        main_export(
            model_name_or_path=self.model,
            output=f"{tmpdirname}/exported_model",
            # with "auto" the taks manager will infer the same task
            # we're using but will add "-with-past" when possible
            task="auto",
            device=self.device.type,
            fp16=self.config.torch_dtype == torch.float16,
            optimize=self.config.auto_optimization,
            no_post_process=not self.config.use_merged,
            do_validation=False,
            **self.hub_kwargs,
            # we hijack the model instantiation and use our random weights model
            model=self.pretrained_model,
        )
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading exported model with ORTModel")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/exported_model",
            session_options=self.config.session_options,
            use_io_binding=self.config.use_io_binding,
            provider=self.config.provider,
            provider_options=self.config.provider_options,
            **(
                {
                    "use_merged": self.config.use_merged,
                    "use_cache": self.config.use_cache,
                }
                if self.is_text_generation_model()
                else {}
            ),
            export=False,
            **self.hub_kwargs,
        )

        if self.config.optimization:
            raise NotImplementedError(
                "Only AutoOptimization is supported when "
                "loading a model with random weights"
            )

        if self.config.quantization or self.config.auto_quantization is not None:
            self.quantize(tmpdirname)

    def load_ortmodel_from_pretrained(self, tmpdirname: str) -> None:
        if (
            self.config.torch_dtype is not None
            and self.config.torch_dtype != torch.float32
        ):
            raise NotImplementedError(
                "Loading with ORTModel is only supported "
                "with torch_dtype float32 for now"
            )

        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=self.model,
            session_options=self.config.session_options,
            use_io_binding=self.config.use_io_binding,
            provider=self.config.provider,
            provider_options=self.config.provider_options,
            export=self.config.export,
            **(
                {
                    "use_merged": self.config.use_merged,
                    "use_cache": self.config.use_cache,
                }
                if self.is_text_generation_model()
                else {}
            ),
            **self.hub_kwargs,
        )

        if self.config.optimization or self.config.auto_optimization is not None:
            self.optimize(tmpdirname)

        if self.config.quantization or self.config.auto_quantization is not None:
            self.quantize(tmpdirname)

    def optimize(self, tmpdirname: str) -> None:
        if self.config.auto_optimization is not None:
            LOGGER.info(f"\t+ Using auto optimization {self.config.auto_optimization}")
            optimization_dict = OmegaConf.to_container(
                self.config.auto_optimization_config, resolve=True
            )
            LOGGER.info("\t+ Setting auto optimization parameters:")
            for key, value in optimization_dict.items():  # type: ignore
                LOGGER.info(f"\t\t+ {key}: {value}")

            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=self.config.auto_optimization, **optimization_dict
            )
        else:
            optimization_dict = OmegaConf.to_container(
                self.config.optimization_config, resolve=True
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
            session_options=self.config.session_options,
            use_io_binding=self.config.use_io_binding,
            provider=self.config.provider,
            provider_options=self.config.provider_options,
        )

    def quantize(self, tmpdirname: str) -> None:
        if self.config.auto_quantization is not None:
            LOGGER.info(f"\t+ Using auto quantization {self.config.auto_quantization}")
            auto_quantization_config_class = getattr(
                AutoQuantizationConfig, self.config.auto_quantization
            )
            quantization_dict = OmegaConf.to_container(
                self.config.auto_quantization_config, resolve=True
            )
            quantization_dict = format_ort_quantization_dict(quantization_dict)
            quantization_config = auto_quantization_config_class(**quantization_dict)

        else:
            LOGGER.info("\t+ Using manual quantization")
            quantization_dict = OmegaConf.to_container(
                self.config.quantization_config, resolve=True
            )
            quantization_dict = format_ort_quantization_dict(quantization_dict)
            quantization_config = QuantizationConfig(**quantization_dict)

        LOGGER.info("\t+ Attempting quantization")
        model_dir = self.pretrained_model.model_save_dir
        components = [file for file in os.listdir(model_dir) if file.endswith(".onnx")]
        for component in components:
            LOGGER.info(f"\t+ Quantizing {component}")
            quantizer = ORTQuantizer.from_pretrained(model_dir, file_name=component)

            if self.config.calibration:
                preprocess_class = get_class(
                    self.config.calibration_config.preprocess_class
                )
                preprocess_function = preprocess_class(model_name_or_path=self.model)

                calibration_dataset = quantizer.get_calibration_dataset(
                    dataset_name=self.config.calibration_config.dataset_name,
                    num_samples=self.config.calibration_config.num_samples,
                    dataset_config_name=self.config.calibration_config.dataset_config_name,
                    dataset_split=self.config.calibration_config.dataset_split,
                    preprocess_function=preprocess_function,
                )

                # Create the calibration configuration
                # containing the parameters related to calibration.
                calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

                # Perform the calibration step:
                # computes the activations quantization ranges
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
            session_options=self.config.session_options,
            use_io_binding=self.config.use_io_binding,
            provider=self.config.provider,
            provider_options=self.config.provider_options,
        )

    def load_automodel_from_config(self) -> None:
        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                config=self.pretrained_config,
                torch_dtype=self.config.torch_dtype,
                trust_remote_code=self.hub_kwargs.get("trust_remote_code", False),
            )
        self.pretrained_model.to_empty(device=self.device)
        randomize_weights(self.pretrained_model)

    def load_automodel_from_pretrained(self) -> None:
        with self.device:
            self.pretrained_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                torch_dtype=self.config.torch_dtype,
                **self.hub_kwargs,
            )

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing model for profiling")
        LOGGER.info("\t+ Wrapping model inside profiler")
        self.pretrained_model = ORTProfilingWrapper(self.pretrained_model)

    def forward(self, input: Dict[str, Tensor], **kwargs) -> "ModelOutput":
        output = self.pretrained_model(**input, **kwargs)

        return output

    def generate(self, input: Dict[str, Tensor], **kwargs) -> "ModelOutput":
        output = self.pretrained_model.generate(**input, **kwargs)
        return output

    def train(
        self,
        training_dataset: "Dataset",
        training_arguments: Dict[str, Any],
        training_callbacks: List["TrainerCallback"],
        training_data_collator: Callable,
    ) -> "TrainerState":
        LOGGER.info("\t+ Setting dataset format to `torch`.")
        training_dataset.set_format(
            type="torch", columns=list(training_dataset.features.keys())
        )

        LOGGER.info(
            "\t+ Wrapping training arguments with "
            "optimum.onnxruntime.ORTTrainingArguments"
        )
        training_arguments = ORTTrainingArguments(**training_arguments)

        LOGGER.info("\t+ Wrapping model with optimum.onnxruntime.ORTTrainer")
        trainer = ORTTrainer(
            model=self.pretrained_model,
            args=training_arguments,
            callbacks=training_callbacks,
            train_dataset=training_dataset,
            data_collator=training_data_collator,
        )

        LOGGER.info("\t+ Starting training")
        trainer.train()
        LOGGER.info("\t+ Training finished successfully")
        trainer_state = trainer.state

        return trainer_state
