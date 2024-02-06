import gc
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from hydra.utils import get_class
from onnxruntime import SessionOptions
from safetensors.torch import save_file
from transformers.utils import ModelOutput
from transformers import TrainerCallback, TrainerState
from transformers.modeling_utils import no_init_weights
from transformers.utils.logging import set_verbosity_error
from optimum.onnxruntime.configuration import (
    AutoOptimizationConfig,
    AutoQuantizationConfig,
    AutoCalibrationConfig,
    OptimizationConfig,
    QuantizationConfig,
    CalibrationConfig,
)
from optimum.onnxruntime import (
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_DECODER_NAME,
    ORTOptimizer,
    ORTQuantizer,
)

from ...generators.dataset_generator import DatasetGenerator
from ...task_utils import TEXT_GENERATION_TASKS
from .config import ORTConfig
from ..base import Backend
from .utils import (
    format_calibration_config,
    format_quantization_config,
    TASKS_TO_ORTMODELS,
    TASKS_TO_ORTSD,
)

# disable transformers logging
set_verbosity_error()

LOGGER = getLogger("onnxruntime")


class ORTBackend(Backend[ORTConfig]):
    NAME: str = "onnxruntime"

    def __init__(self, config: ORTConfig) -> None:
        super().__init__(config)
        self.validate_task()

        if self.config.library == "diffusers":
            self.ortmodel_class = get_class(TASKS_TO_ORTSD[self.config.task])
            LOGGER.info(f"Using ORTDiffusion class {self.ortmodel_class.__name__}")
        elif self.config.task in TASKS_TO_ORTMODELS:
            self.ortmodel_class = get_class(TASKS_TO_ORTMODELS[self.config.task])
            LOGGER.info(f"Using ORTModel class {self.ortmodel_class.__name__}")
        else:
            raise NotImplementedError(f"ORTBackend does not support task {self.config.task}")

        self.set_session_options()
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            self.load_ortmodel_with_no_weights()
        else:
            self.load_ortmodel_from_pretrained()

        if self.is_deferred_trt_loading():
            return

        if self.is_optimized or self.is_quantized:
            original_model = self.config.model
            self.config.model = self.pretrained_model.model_save_dir

        if self.is_optimized:
            self.optimize_onnx_files()

        if self.is_quantized:
            self.quantize_onnx_files()

        if self.is_optimized or self.is_quantized:
            original_export = self.config.export
            self.load_ortmodel_from_pretrained()  # load optimized/quantized model
            self.config.export = original_export
            self.config.model = original_model

        self.validate_provider()

    def validate_task(self) -> None:
        if self.config.task not in {**TASKS_TO_ORTMODELS, **TASKS_TO_ORTSD}:
            raise NotImplementedError(f"ORTBackend does not support task {self.config.task}")

    def validate_provider(self) -> None:
        assert (
            self.pretrained_model.providers[0] == self.config.provider
        ), f"{self.config.provider} is not first in providers list: {self.pretrained_model.providers}"

    def is_deferred_trt_loading(self) -> bool:
        return self.config.provider == "TensorrtExecutionProvider" and self.config.task in TEXT_GENERATION_TASKS

    def set_session_options(self) -> None:
        self.session_options = SessionOptions()
        for key, value in self.config.session_options.items():
            setattr(self.session_options, key, value)

    def load_ortmodel_with_no_weights(self) -> None:
        LOGGER.info("\t+ Creating no weights model directory")
        no_weights_model = os.path.join(self.tmpdir.name, "no_weights")
        os.makedirs(no_weights_model, exist_ok=True)

        LOGGER.info("\t+ Saving pretrained config")
        self.pretrained_config.save_pretrained(save_directory=no_weights_model)

        LOGGER.info("\t+ Creating no weights model weights")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        LOGGER.info("\t+ Saving no weights model weights")
        save_file(
            filename=os.path.join(no_weights_model, "model.safetensors"),
            metadata={"format": "pt"},
            tensors=state_dict,
        )

        with no_init_weights():
            original_model = self.config.model
            self.config.model = no_weights_model
            LOGGER.info("\t+ Loading no weights model")
            self.load_ortmodel_from_pretrained()
            self.config.model = original_model

    def load_ortmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            self.config.model,
            export=self.config.export,
            session_options=self.session_options,
            provider_options=self.config.provider_options,
            use_io_binding=self.config.use_io_binding,
            provider=self.config.provider,
            **self.config.hub_kwargs,
            **self.ortmodel_kwargs,
        )

    @property
    def is_optimized(self) -> bool:
        return (self.config.auto_optimization is not None) or self.config.optimization

    @property
    def is_quantized(self) -> bool:
        return (self.config.auto_quantization is not None) or self.config.quantization

    @property
    def is_calibrated(self) -> bool:
        return (self.config.auto_calibration is not None) or self.config.calibration

    @property
    def ortmodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.task in TEXT_GENERATION_TASKS:
            kwargs["use_cache"] = self.config.use_cache
            kwargs["use_merged"] = self.config.use_merged

        return kwargs

    @property
    def onnx_files_names(self):
        assert os.path.isdir(self.config.model), f"{self.config.model} is not a directory"
        if self.config.use_merged:
            return [
                model
                for model in os.listdir(self.config.model)
                if model not in [ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME] and model.endswith(".onnx")
            ]
        else:
            return [file for file in os.listdir(self.config.model) if file.endswith(".onnx")]

    @property
    def inputs_names(self) -> List[str]:
        if hasattr(self.pretrained_model, "inputs_names"):
            return self.pretrained_model.inputs_names
        elif hasattr(self.pretrained_model, "input_names"):
            return self.pretrained_model.input_names
        else:
            return []

    def optimize_onnx_files(self) -> None:
        LOGGER.info("\t+ Attempting optimization")
        optimized_model_path = os.path.join(self.tmpdir.name, "optimized")
        LOGGER.info("\t+ Processing optimization config")
        if self.config.auto_optimization is not None:
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=self.config.auto_optimization,
                for_gpu=(self.config.device == "cuda"),
                **self.config.auto_optimization_config,
            )
        elif self.config.optimization:
            optimization_config = OptimizationConfig(
                optimize_for_gpu=(self.config.device == "cuda"),
                **self.config.optimization_config,
            )
        LOGGER.info("\t+ Creating optimizer")
        optimizer = ORTOptimizer.from_pretrained(self.config.model, file_names=self.onnx_files_names)
        LOGGER.info("\t+ Optimizing ORTModel")
        optimizer.optimize(
            optimization_config,
            save_dir=optimized_model_path,
            # TODO: add support for these
            use_external_data_format=None,
            one_external_file=True,
            file_suffix="",
        )

        if self.pretrained_processor is not None:
            self.pretrained_processor.save_pretrained(optimized_model_path)

        if self.pretrained_config is not None:
            self.pretrained_config.save_pretrained(optimized_model_path)

        self.config.model = optimized_model_path

    def quantize_onnx_files(self) -> None:
        LOGGER.info("\t+ Attempting quantization")
        quantized_model_path = f"{self.tmpdir.name}/quantized"

        if self.is_calibrated and len(self.onnx_files_names) > 1:
            raise NotImplementedError(
                "Calibrated/Static Quantization is not supported for models with multiple components. "
                f"Found {len(self.onnx_files_names)} components."
            )

        LOGGER.info("\t+ Processing quantization config")
        if self.config.auto_quantization is not None:
            auto_quantization_config = format_quantization_config(self.config.auto_quantization_config)
            auto_quantization_class = getattr(AutoQuantizationConfig, self.config.auto_quantization)
            quantization_config = auto_quantization_class(**auto_quantization_config)
        elif self.config.quantization:
            quantization_config = format_quantization_config(self.config.quantization_config)
            quantization_config = QuantizationConfig(**quantization_config)

        if self.is_calibrated:
            LOGGER.info("\t+ Generating calibration dataset")
            dataset_shapes = {
                "dataset_size": 1,
                "sequence_length": 1,
                **self.model_shapes,
            }
            calibration_dataset = DatasetGenerator(
                task=self.config.task,
                dataset_shapes=dataset_shapes,
                model_shapes=self.model_shapes,
            ).generate()
            columns_to_be_removed = list(set(calibration_dataset.column_names) - set(self.inputs_names))
            calibration_dataset = calibration_dataset.remove_columns(columns_to_be_removed)

            LOGGER.info("\t+ Processing calibration config")
            if self.config.auto_calibration is not None:
                LOGGER.info("\t+ Processing calibration config")
                auto_calibration_method = getattr(AutoCalibrationConfig, self.config.auto_calibration)
                calibration_config = auto_calibration_method(
                    calibration_dataset,
                    **self.config.auto_calibration_config,
                )
            elif self.config.calibration:
                LOGGER.info("\t+ Processing calibration config")
                calibration_config = format_calibration_config(self.config.calibration_config)
                calibration_config = CalibrationConfig(
                    dataset_name="calibration_dataset",
                    dataset_split=calibration_dataset.split,
                    dataset_num_samples=calibration_dataset.num_rows,
                    dataset_config_name=calibration_dataset.config_name,
                    **self.config.calibration_config,
                )

        for onnx_file_name in self.onnx_files_names:
            LOGGER.info(f"\t+ Creating quantizer for {onnx_file_name}")
            quantizer = ORTQuantizer.from_pretrained(self.config.model, file_name=onnx_file_name)

            if self.is_calibrated:
                LOGGER.info("\t+ Fitting calibration tensors range")
                calibration_tensors_range = quantizer.fit(
                    dataset=calibration_dataset,
                    use_gpu=(self.config.device == "cuda"),
                    calibration_config=calibration_config,
                    operators_to_quantize=quantization_config.operators_to_quantize,
                    # TODO: add support for these (maybe)
                    use_external_data_format=False,
                    force_symmetric_range=False,
                    batch_size=1,
                )
            else:
                calibration_tensors_range = None

            LOGGER.info("\t+ Quantizing model")
            quantizer.quantize(
                save_dir=quantized_model_path,
                quantization_config=quantization_config,
                calibration_tensors_range=calibration_tensors_range,
                # TODO: add support for these (maybe)
                use_external_data_format=False,
                preprocessor=None,
                file_suffix="",
            )

        if self.pretrained_processor is not None:
            self.pretrained_processor.save_pretrained(quantized_model_path)

        if self.pretrained_config is not None:
            self.pretrained_config.save_pretrained(quantized_model_path)

        self.config.model = quantized_model_path

    def prepare_for_inference(self, **kwargs) -> None:
        if self.is_deferred_trt_loading():
            LOGGER.info("\t+ Creating dynamic shapes for Tensorrt engine. Engine creation might take a while.")
            batch_size = kwargs["batch_size"]
            max_new_tokens = kwargs["max_new_tokens"]
            sequence_length = kwargs["sequence_length"]
            self.config.provider_options = {
                **self.config.provider_options,
                "trt_profile_min_shapes": (
                    f"input_ids:{batch_size}x{sequence_length},"
                    f"attention_mask:{batch_size}x{sequence_length},"
                    f"position_ids:{batch_size}x{sequence_length}"
                ),
                "trt_profile_max_shapes": (
                    f"input_ids:{batch_size}x{sequence_length + max_new_tokens},"
                    f"attention_mask:{batch_size}x{sequence_length + max_new_tokens},"
                    f"position_ids:{batch_size}x{sequence_length + max_new_tokens}"
                ),
                "trt_profile_opt_shapes": (
                    f"input_ids:{batch_size}x{sequence_length + max_new_tokens},"
                    f"attention_mask:{batch_size}x{sequence_length + max_new_tokens},"
                    f"position_ids:{batch_size}x{sequence_length + max_new_tokens}"
                ),
            }
            self.load_ortmodel_from_pretrained()
            self.validate_provider()

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.library == "diffusers":
            return {"prompt": inputs["prompt"]}

        LOGGER.info(f"\t+ Moving inputs tensors to device {self.config.device}")
        for key, value in list(inputs.items()):
            if key in self.inputs_names:
                inputs[key] = value.to(self.config.device)
            else:
                inputs.pop(key)

        return inputs

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model(**inputs, **kwargs)

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model.generate(**inputs, **kwargs)

    def train(
        self,
        training_dataset: Dataset,
        training_arguments: Dict[str, Any],
        training_callbacks: List[TrainerCallback],
        training_data_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    ) -> TrainerState:
        from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments

        LOGGER.info("\t+ Setting dataset format to `torch`")
        training_dataset.set_format(type="torch", columns=list(training_dataset.features.keys()))
        LOGGER.info("\t+ Wrapping training arguments with optimum.onnxruntime.ORTTrainingArguments")
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

        return trainer.state

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            LOGGER.info("\t+ Cleaning temporary directory")
            self.tmpdir.cleanup()

        if self.config.device == "cuda":
            LOGGER.info("\t+ Emptying CUDA cache")
            torch.cuda.empty_cache()

        gc.collect()
