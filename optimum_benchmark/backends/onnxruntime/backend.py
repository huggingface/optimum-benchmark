import gc
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch
from hydra.utils import get_class
from onnxruntime import SessionOptions
from optimum.onnxruntime import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ORTOptimizer,
    ORTQuantizer,
    ORTTrainer,
    ORTTrainingArguments,
)
from optimum.onnxruntime.configuration import (
    AutoCalibrationConfig,
    AutoOptimizationConfig,
    AutoQuantizationConfig,
    OptimizationConfig,
    QuantizationConfig,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import TrainerCallback, TrainerState

from ...profilers.ort_profiler import ORTProfilingWrapper
from ..base import Backend
from ..ddp_utils import record_if_available, training_worker
from ..optimum_utils import main_export
from ..pytorch.utils import randomize_weights
from .config import ORTConfig
from .utils import TASKS_TO_ORTMODELS, TASKS_TO_ORTSD, format_quantization_config

LOGGER = getLogger("onnxruntime")


class ORTBackend(Backend[ORTConfig]):
    NAME: str = "onnxruntime"

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]) -> None:
        super().__init__(model, task, device, hub_kwargs)
        self.validate_device()
        self.validate_task()

        if self.is_diffusion_pipeline():
            self.ortmodel_class = get_class(TASKS_TO_ORTSD[self.task])
        elif self.task in TASKS_TO_ORTMODELS:
            self.ortmodel_class = TASKS_TO_ORTMODELS[self.task]

        ortmodel_name = self.ortmodel_class.__name__
        LOGGER.info(f"Inferred ORTModel class {ortmodel_name} for task {self.task} and model_type {self.model_type}")

    def validate_device(self) -> None:
        if self.device.type not in ["cpu", "cuda"]:
            raise ValueError(f"ORTBackend only supports CPU and CUDA devices, got {self.device.type}")

    def validate_task(self) -> None:
        if self.task not in TASKS_TO_ORTMODELS and self.task not in TASKS_TO_ORTSD:
            raise NotImplementedError(f"ORTBackend does not support task {self.task}")

    def configure(self, config: ORTConfig) -> None:
        super().configure(config)

        # Process torch dtype
        self.torch_dtype = getattr(torch, self.config.torch_dtype) if self.config.torch_dtype is not None else None

        ###### Training with ORTModule ######
        # ort-training is basically a different package so we might need to separate these two backends in the future
        if not self.config.use_inference_session:
            if self.config.no_weights:
                self.load_automodel_from_config()
            else:
                self.load_automodel_from_pretrained()

            if self.config.peft_strategy is not None:
                LOGGER.info("\t+ Applying PEFT")
                from peft import get_peft_model

                from ..peft_utils import get_peft_config_class

                peft_config_class = get_peft_config_class(self.config.peft_strategy)
                peft_config = peft_config_class(**self.config.peft_config)
                self.pretrained_model = get_peft_model(self.pretrained_model, peft_config=peft_config)
            # early exit because nothing of the following can be applied to training
            return

        ###### Inference with ORTModelForxxx ######
        # Inference session options
        self.session_options = SessionOptions()
        for key, value in self.config.session_options.items():
            setattr(self.session_options, key, value)

        # Exporting, optimizing, post-processing and quantizing with ORTModelForxxx
        self.tmpdir = TemporaryDirectory()

        # Some statefullness to handle the different combinations of options
        self.export = self.config.export
        self.use_merged = self.config.use_merged
        self.provider_options = self.config.provider_options.copy()

        if self.is_diffusion_pipeline():
            self.load_ortmodel()
            # early exit because nothing of the following can be applied to diffusion pipelines
            return

        if self.config.no_weights:
            self.load_automodel_from_config()  # creates dummy automodel
            self.export_automodel()  # exports automodel
            self.export = False
        else:
            if self.config.export:
                self.use_merged = False  # merging is handled separately
                self.load_automodel_from_pretrained()  # creates automodel from pretrained
                self.export_automodel()  # exports automodel
                self.export = False

        self.delete_pretrained_model()  # deletes automodel

        if self.config.auto_optimization or self.config.optimization:
            self.optimize_onnx_files()

        if self.config.use_merged:
            self.merge_onnx_files()
            self.use_merged = True

        if self.config.auto_quantization or self.config.quantization:
            self.quantize_onnx_files()

        if not (self.config.provider == "TensorrtExecutionProvider" and self.is_text_generation_model()):
            self.load_ortmodel()
            self.tmpdir.cleanup()

    def load_automodel_from_config(self) -> None:
        from accelerate import init_empty_weights

        LOGGER.info("\t+ Loading AutoModel from config")
        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                self.pretrained_config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.hub_kwargs.get("trust_remote_code", False),
            )
        self.pretrained_model.to_empty(device=self.device)
        randomize_weights(self.pretrained_model)

    def load_automodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading AutoModel from pretrained")
        with self.device:
            self.pretrained_model = self.automodel_class.from_pretrained(
                self.model,
                torch_dtype=self.torch_dtype,
                **self.hub_kwargs,
            )

    def load_ortmodel(self) -> None:
        LOGGER.info("\t+ Loading ORTModel")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            self.model,
            export=self.export,
            provider=self.config.provider,
            session_options=self.session_options,
            provider_options=self.provider_options,
            use_io_binding=self.config.use_io_binding,
            **self.ortmodel_kwargs,
            **self.hub_kwargs,
        )
        # exported or not, the onnx model is/was here
        self.model = self.pretrained_model.model_save_dir

    @property
    def ortmodel_kwargs(self) -> Dict[str, Any]:
        if self.is_text_generation_model():
            return {"use_cache": self.config.use_cache, "use_merged": self.use_merged}
        else:
            return {}

    @property
    def export_task(self) -> str:
        return self.task + "-with-past" if self.config.use_cache and self.is_text_generation_model() else self.task

    def export_automodel(self) -> None:
        LOGGER.info("\t+ Exporting AutoModel to ONNX")
        exported_model_dir = f"{self.tmpdir.name}/exported_model"
        self.merging_config, self.models_and_onnx_configs = main_export(
            self.model,
            output=exported_model_dir,
            task=self.export_task,
            device=self.device.type,
            fp16=self.torch_dtype == torch.float16,
            **self.hub_kwargs,
            # we hijack the model instantiation and use our random weights model
            model=self.pretrained_model,
        )
        self.model = exported_model_dir

    def merge_onnx_files(self) -> None:
        LOGGER.info("\t+ Post-processing the exported model")
        self.merging_config.post_process_exported_models(self.model, self.models_and_onnx_configs, None)

    @property
    def onnx_files_names(self):
        assert os.path.isdir(self.model), f"{self.model} is not a directory"
        return [file for file in os.listdir(self.model) if file.endswith(".onnx")]

    def optimize_onnx_files(self) -> None:
        LOGGER.info("\t+ Attempting optimization")
        optimized_model_path = f"{self.tmpdir.name}/optimized"
        LOGGER.info("\t+ Processing optimization config")
        if self.config.auto_optimization is not None:
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=self.config.auto_optimization,
                for_gpu=self.device.type == "cuda",
                **self.config.auto_optimization_config,
            )
        elif self.config.optimization:
            optimization_config = OptimizationConfig(
                optimize_for_gpu=self.device.type == "cuda", **self.config.optimization_config
            )
        LOGGER.info("\t+ Creating optimizer")
        optimizer = ORTOptimizer.from_pretrained(self.model, file_names=self.onnx_files_names)
        LOGGER.info("\t+ Optimizing ORTModel")
        optimizer.optimize(
            optimization_config,
            save_dir=optimized_model_path,
            file_suffix="",
            # TODO: add support for these
            use_external_data_format=None,
            one_external_file=True,
        )
        self.model = optimized_model_path

    @property
    def onnx_files_names_to_quantize(self):
        assert os.path.isdir(self.model), f"{self.model} is not a directory"
        if self.config.use_merged:
            # we filter merging components since they're not used for inference
            # this also allows for calibration of one merged component models (like gpt2)
            return [
                model
                for model in self.onnx_files_names
                if model not in [ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME]
            ]
        else:
            return self.onnx_files_names

    def quantize_onnx_files(self) -> None:
        LOGGER.info("\t+ Attempting quantization")
        quantized_model_path = f"{self.tmpdir.name}/quantized"
        LOGGER.info("\t+ Processing quantization config")
        if self.config.calibration and len(self.onnx_files_names_to_quantize) > 1:
            raise NotImplementedError("Calibration is not supported for models with multiple components")
        if self.config.auto_quantization is not None:
            self.config.auto_quantization_config = format_quantization_config(self.config.auto_quantization_config)
            auto_quantization_config_class = getattr(AutoQuantizationConfig, self.config.auto_quantization)
            quantization_config = auto_quantization_config_class(**self.config.auto_quantization_config)
        elif self.config.quantization:
            self.config.quantization_config = format_quantization_config(self.config.quantization_config)
            quantization_config = QuantizationConfig(**self.config.quantization_config)
        LOGGER.info(f"\t+ Model has {len(self.onnx_files_names_to_quantize)} components to quantize")
        if len(self.onnx_files_names_to_quantize) == 1:
            LOGGER.info("\t+ Creating quantizer")
            quantizer = ORTQuantizer.from_pretrained(self.model, file_name=self.onnx_files_names_to_quantize[0])
            if self.config.calibration:
                LOGGER.info("\t+ Processing calibration config")
                preprocess_class = get_class(self.config.calibration_config.pop("preprocess_class"))
                self.config.calibration_config["preprocess_function"] = preprocess_class(model_name_or_path=self.model)
                LOGGER.info("\t+ Loading calibration dataset")
                calibration_dataset = quantizer.get_calibration_dataset(**self.config.calibration_config)
                LOGGER.info("\t+ Creating calibration config")
                calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
                LOGGER.info("\t+ Fitting calibration tensors range")
                calibration_tensors_range = quantizer.fit(
                    dataset=calibration_dataset,
                    calibration_config=calibration_config,
                    operators_to_quantize=quantization_config.operators_to_quantize,
                    use_gpu=self.device.type == "cuda",
                    # TODO: add support for these
                    batch_size=1,
                    use_external_data_format=False,
                    force_symmetric_range=False,
                )
            else:
                calibration_tensors_range = None
            LOGGER.info("\t+ Quantizing model")
            quantizer.quantize(
                save_dir=quantized_model_path,
                quantization_config=quantization_config,
                calibration_tensors_range=calibration_tensors_range,
                # TODO: add support for these
                use_external_data_format=False,
                preprocessor=None,
            )
        else:
            for onnx_file_name_to_quantize in self.onnx_files_names_to_quantize:
                LOGGER.info(f"\t+ Creating quantizer for {onnx_file_name_to_quantize}")
                quantizer = ORTQuantizer.from_pretrained(self.model, file_name=onnx_file_name_to_quantize)
                LOGGER.info(f"\t+ Quantizing {onnx_file_name_to_quantize}")
                quantizer.quantize(
                    save_dir=quantized_model_path,
                    quantization_config=quantization_config,
                    calibration_tensors_range=None,
                    file_suffix="",
                    # TODO: add support for these
                    use_external_data_format=False,
                    preprocessor=None,
                )
        self.model = quantized_model_path

    def prepare_for_inference(self, **kwargs) -> None:
        if self.config.provider == "TensorrtExecutionProvider" and self.is_text_generation_model():
            max_new_tokens = kwargs["max_new_tokens"]
            batch_size = kwargs["input_shapes"]["batch_size"]
            sequence_length = kwargs["input_shapes"]["sequence_length"]

            LOGGER.info("\t+ Creating dynamic shapes for Tensorrt engine, loading will take a while")
            self.provider_options = {
                **self.provider_options,
                "trt_profile_min_shapes": f"input_ids:{batch_size}x{sequence_length},attention_mask:{batch_size}x{sequence_length}",
                "trt_profile_max_shapes": f"input_ids:{batch_size}x{sequence_length + max_new_tokens},attention_mask:{batch_size}x{sequence_length + max_new_tokens}",
                "trt_profile_opt_shapes": f"input_ids:{batch_size}x{sequence_length + max_new_tokens},attention_mask:{batch_size}x{sequence_length + max_new_tokens}",
            }

            self.load_ortmodel()
            self.tmpdir.cleanup()

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing model for profiling")
        LOGGER.info("\t+ Wrapping model inside profiler")
        self.pretrained_model = ORTProfilingWrapper(self.pretrained_model)

    @record_if_available
    def train(
        self,
        training_dataset: "Dataset",
        training_arguments: Dict[str, Any],
        training_callbacks: List["TrainerCallback"],
        training_data_collator: Callable,
    ) -> "TrainerState":
        worker_args = (
            "torch",
            LOGGER,
            ORTTrainer,
            ORTTrainingArguments,
            self.config.use_ddp,
            training_dataset,
            training_arguments,
            training_data_collator,
            training_callbacks,
            self.pretrained_model,
        )

        if self.config.use_ddp:
            from torch.distributed.launcher.api import LaunchConfig, elastic_launch

            # For DDP, we log only the state of the first rank as transformers does.
            # since the batch size used in measuring the throughput is the one of world size.
            ddp_config = LaunchConfig(**self.config.ddp_config)
            results = elastic_launch(config=ddp_config, entrypoint=training_worker)(worker_args)[0]
        else:
            # For DP, we can still use training_worker, simply not wrapped by the elastic_launch class.
            results = training_worker(worker_args)

        return results

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            self.tmpdir.cleanup()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
