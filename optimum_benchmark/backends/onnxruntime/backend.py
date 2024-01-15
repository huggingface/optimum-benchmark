import gc
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from hydra.utils import get_class
from onnxruntime import SessionOptions
from optimum.onnxruntime import (
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ORTOptimizer,
    ORTQuantizer,
)
from optimum.onnxruntime.configuration import (
    AutoCalibrationConfig,
    AutoOptimizationConfig,
    AutoQuantizationConfig,
    CalibrationConfig,
    OptimizationConfig,
    QuantizationConfig,
)
from safetensors.torch import save_file
from transformers import TrainerCallback, TrainerState
from transformers.modeling_utils import no_init_weights
from transformers.utils.logging import set_verbosity_error

from ...generators.dataset_generator import DatasetGenerator
from ..base import Backend
from ..peft_utils import get_peft_config_class
from ..pytorch.utils import randomize_weights
from .config import ORTConfig
from .utils import (
    TASKS_TO_ORTMODELS,
    TASKS_TO_ORTSD,
    format_calibration_config,
    format_quantization_config,
)

# disable transformers logging
set_verbosity_error()

LOGGER = getLogger("onnxruntime")


class ORTBackend(Backend[ORTConfig]):
    NAME: str = "onnxruntime"

    def __init__(self, model: str, task: str, library: str, device: str, hub_kwargs: Dict[str, Any]) -> None:
        super().__init__(model, task, library, device, hub_kwargs)
        self.validate_device()
        self.validate_task()

    def validate_device(self) -> None:
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"ORTBackend only supports CPU and CUDA devices, got {self.device}")

    def validate_task(self) -> None:
        if self.task not in TASKS_TO_ORTMODELS and self.task not in TASKS_TO_ORTSD:
            raise NotImplementedError(f"ORTBackend does not support task {self.task}")

    def configure(self, config: ORTConfig) -> None:
        super().configure(config)

        if self.library == "diffusers":
            self.ortmodel_class = get_class(TASKS_TO_ORTSD[self.task])
        elif self.task in TASKS_TO_ORTMODELS:
            self.ortmodel_class = TASKS_TO_ORTMODELS[self.task]

        ortmodel_name = self.ortmodel_class.__name__
        LOGGER.info(f"Inferred ORTModel class {ortmodel_name} for task {self.task} and model_type {self.model_type}")

        ######## Training with ORTModule ########
        if not self.config.use_inference_session:
            if self.config.no_weights:
                self.load_automodel_with_no_weights()
            else:
                self.load_automodel_from_pretrained()

            if self.config.peft_strategy is not None:
                LOGGER.info("\t+ Applying PEFT")
                from peft import get_peft_model

                peft_config_class = get_peft_config_class(self.config.peft_strategy)
                peft_config = peft_config_class(**self.config.peft_config)
                self.pretrained_model = get_peft_model(self.pretrained_model, peft_config=peft_config)

            return  # early exit because nothing of the following can be applied to training

        ######## Inference with ORTModelForxxx ########
        self.export = self.config.export
        self.tmpdir = TemporaryDirectory()
        self.session_options = SessionOptions()
        self.provider_options = self.config.provider_options

        for key, value in self.config.session_options.items():
            setattr(self.session_options, key, value)

        if self.config.no_weights:
            self.load_ortmodel_with_no_weights()
        else:
            self.load_ortmodel_from_pretrained()

        if self.config.provider == "TensorrtExecutionProvider" and self.is_text_generation_model():
            return  # deferred loading for trt text generation models

        if self.is_optimized or self.is_quantized:
            original_model = self.model
            original_export = self.export

            self.model = self.pretrained_model.model_save_dir  # self.model will point to a directory from here on
            self.export = False  # we disable export because we'll load the optimized/quantized model now

        if self.is_optimized:
            self.optimize_onnx_files()

        if self.is_quantized:
            self.quantize_onnx_files()

        if self.is_optimized or self.is_quantized:
            self.load_ortmodel_from_pretrained()  # load optimized/quantized model
            self.export = original_export
            self.model = original_model

        self.validate_provider()

    def validate_provider(self) -> None:
        assert (
            self.pretrained_model.providers[0] == self.config.provider
        ), f"{self.config.provider} is not first in providers list: {self.pretrained_model.providers}"

    def load_automodel_with_no_weights(self) -> None:
        original_model = self.model
        no_weights_model = os.path.join(self.tmpdir.name, "no_weights")

        if not os.path.exists(no_weights_model):
            LOGGER.info("\t+ Creating no weights model directory")
            os.makedirs(no_weights_model)

        LOGGER.info("\t+ Saving pretrained config")
        self.pretrained_config.save_pretrained(save_directory=no_weights_model)

        LOGGER.info("\t+ Creating no weights model")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        LOGGER.info("\t+ Saving no weights model")
        save_file(
            filename=os.path.join(no_weights_model, "model.safetensors"),
            metadata={"format": "pt"},
            tensors=state_dict,
        )

        LOGGER.info("\t+ Loading no weights model")
        with no_init_weights():
            self.model = no_weights_model
            self.load_automodel_from_pretrained()
            self.model = original_model

        LOGGER.info("\t+ Randomizing weights")
        randomize_weights(self.pretrained_model)
        LOGGER.info("\t+ Tying model weights after randomization")
        self.pretrained_model.tie_weights()

    def load_automodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading AutoModel from pretrained")
        self.pretrained_model = self.automodel_class.from_pretrained(
            self.model,
            **self.automodel_kwargs,
            **self.hub_kwargs,
        ).to(self.device)

    def load_ortmodel_with_no_weights(self) -> None:
        no_weights_model = os.path.join(self.tmpdir.name, "no_weights")

        LOGGER.info("\t+ Loading AutoModel with no weights")
        self.load_automodel_with_no_weights()
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading ORTModel with no weights")
        with no_init_weights():
            original_model = self.model
            self.model = no_weights_model
            self.load_ortmodel_from_pretrained()
            self.model = original_model

    def load_ortmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            self.model,
            export=self.export,
            session_options=self.session_options,
            provider_options=self.provider_options,
            use_io_binding=self.config.use_io_binding,
            provider=self.config.provider,
            **self.ortmodel_kwargs,
            **self.hub_kwargs,
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
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.torch_dtype is not None and hasattr(torch, self.config.torch_dtype):
            kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)
        else:
            kwargs["torch_dtype"] = self.config.torch_dtype

        return kwargs

    @property
    def ortmodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.is_text_generation_model():
            kwargs["use_cache"] = self.config.use_cache
            kwargs["use_merged"] = self.config.use_merged

        return kwargs

    @property
    def onnx_files_names(self):
        assert os.path.isdir(self.model), f"{self.model} is not a directory"
        return [file for file in os.listdir(self.model) if file.endswith(".onnx")]

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

    def optimize_onnx_files(self) -> None:
        LOGGER.info("\t+ Attempting optimization")
        optimized_model_path = os.path.join(self.tmpdir.name, "optimized")
        LOGGER.info("\t+ Processing optimization config")
        if self.config.auto_optimization is not None:
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                for_gpu=(self.device == "cuda"),
                optimization_level=self.config.auto_optimization,
                **self.config.auto_optimization_config,
            )
        elif self.config.optimization:
            optimization_config = OptimizationConfig(
                optimize_for_gpu=(self.device == "cuda"),
                **self.config.optimization_config,
            )
        LOGGER.info("\t+ Creating optimizer")
        optimizer = ORTOptimizer.from_pretrained(self.model, file_names=self.onnx_files_names)
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

        self.model = optimized_model_path

    def quantize_onnx_files(self) -> None:
        LOGGER.info("\t+ Attempting quantization")
        quantized_model_path = f"{self.tmpdir.name}/quantized"

        if self.is_calibrated and len(self.onnx_files_names_to_quantize) > 1:
            raise NotImplementedError(
                "Calibrated/Static Quantization is not supported for models with multiple components. "
                f"Found {len(self.onnx_files_names_to_quantize)} components."
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
            dataset_shapes = {"dataset_size": 1, "sequence_length": 1, **self.model_shapes}
            calibration_dataset = DatasetGenerator(task=self.task, dataset_shapes=dataset_shapes).generate()
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

        for onnx_file_name_to_quantize in self.onnx_files_names_to_quantize:
            LOGGER.info(f"\t+ Creating quantizer for {onnx_file_name_to_quantize}")
            quantizer = ORTQuantizer.from_pretrained(self.model, file_name=onnx_file_name_to_quantize)

            if self.is_calibrated:
                LOGGER.info("\t+ Fitting calibration tensors range")
                calibration_tensors_range = quantizer.fit(
                    dataset=calibration_dataset,
                    use_gpu=(self.device == "cuda"),
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

        self.model = quantized_model_path

    @property
    def inputs_names(self) -> List[str]:
        if hasattr(self.pretrained_model, "inputs_names"):
            return self.pretrained_model.inputs_names
        elif hasattr(self.pretrained_model, "input_names"):
            return self.pretrained_model.input_names
        else:
            return []

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.library == "diffusers":
            return {"prompt": inputs["prompt"]}

        LOGGER.info(f"\t+ Moving inputs tensors to device {self.device}")
        for key, value in list(inputs.items()):
            if key in self.inputs_names:
                inputs[key] = value.to(self.device)
            else:
                inputs.pop(key)

        return inputs

    def prepare_for_inference(self, **kwargs) -> None:
        if self.config.provider == "TensorrtExecutionProvider" and self.is_text_generation_model():
            batch_size = kwargs["batch_size"]
            max_new_tokens = kwargs["max_new_tokens"]
            sequence_length = kwargs["sequence_length"]

            LOGGER.info("\t+ Creating dynamic shapes for Tensorrt engine, loading will take a while")
            self.provider_options = {
                **self.provider_options,
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

        if self.device == "cuda":
            LOGGER.info("\t+ Emptying CUDA cache")
            torch.cuda.empty_cache()

        gc.collect()
