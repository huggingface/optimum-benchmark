import os
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Dict

import torch
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

from ...generators.dataset_generator import DatasetGenerator
from ...import_utils import is_accelerate_available, is_torch_distributed_available
from ..base import Backend
from ..transformers_utils import fast_weights_init
from .config import ORTConfig
from .utils import (
    TASKS_TO_ORTMODELS,
    TASKS_TO_ORTPIPELINES,
    format_calibration_config,
    format_quantization_config,
)

if is_accelerate_available():
    from accelerate import Accelerator

if is_torch_distributed_available():
    import torch.distributed


class ORTBackend(Backend[ORTConfig]):
    NAME: str = "onnxruntime"

    def __init__(self, config: ORTConfig) -> None:
        super().__init__(config)

        if self.config.library != "diffusers" and self.config.task in TASKS_TO_ORTMODELS:
            self.ort_model_loader = get_class(TASKS_TO_ORTMODELS[self.config.task])
            self.logger.info(f"Using ORTModel class {self.ort_model_loader.__name__}")
        elif self.config.library == "diffusers" and self.config.task in TASKS_TO_ORTPIPELINES:
            self.ort_model_loader = get_class(TASKS_TO_ORTPIPELINES[self.config.task])
            self.logger.info(f"Using ORTDiffusionPipeline class {self.ort_model_loader.__name__}")
        else:
            raise NotImplementedError(f"ORTBackend does not support task {self.config.task}")

    def validate_execution_provider(self) -> None:
        if not self.pretrained_model.providers[0] == self.config.provider:
            raise ValueError(
                f"{self.config.provider} is not first in providers list: {self.pretrained_model.providers}"
            )

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            self.logger.info("\t+ Creating no weights ORTModel")
            self.create_no_weights_model_fast()
            self.logger.info("\t+ Loading no weights ORTModel")
            self.load_ortmodel_with_no_weights()
        else:
            self.logger.info("\t+ Loading pretrained ORTModel")
            self.load_ortmodel_from_pretrained()

        if self.is_optimized or self.is_quantized:
            original_model, self.config.model = self.config.model, self.pretrained_model.model_save_dir

        if self.is_optimized:
            self.logger.info("\t+ Applying ORT optimization")
            self.optimize_onnx_files()
            self.config.model = self.optimized_model

        if self.is_quantized:
            self.logger.info("\t+ Applying ORT quantization")
            self.quantize_onnx_files()
            self.config.model = self.quantized_model

        if self.is_optimized or self.is_quantized:
            original_export, self.config.export = self.config.export, False
            self.logger.info("\t+ Loading optimized/quantized model")
            self.load_ortmodel_from_pretrained()
            self.config.export = original_export
            self.config.model = original_model

        self.logger.info("\t+ Validating requested Execution Provider")
        self.validate_execution_provider()

        self.logger.info("\t+ Cleaning up backend temporary directory")
        self.tmpdir.cleanup()

    def load_ortmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.ort_model_loader.from_pretrained(
            self.config.model,
            **self.config.model_kwargs,
            **self.ortmodel_kwargs,
        )

    def load_ortmodel_with_no_weights(self) -> None:
        with fast_weights_init():
            original_model, self.config.model = self.config.model, self.no_weights_model_path.as_posix()
            original_export, self.config.export = self.config.export, True
            self.logger.info("\t+ Loading no weights ORTModel")
            self.load_ortmodel_from_pretrained()
            self.config.export = original_export
            self.config.model = original_model

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

        if self.config.export is not None:
            kwargs["export"] = self.config.export

        if self.config.provider is not None:
            kwargs["provider"] = self.config.provider

        if self.config.use_cache is not None:
            kwargs["use_cache"] = self.config.use_cache

        if self.config.use_merged is not None:
            kwargs["use_merged"] = self.config.use_merged

        if self.config.torch_dtype is not None:
            kwargs["torch_dtype"] = self.config.torch_dtype

        if self.config.use_io_binding is not None:
            kwargs["use_io_binding"] = self.config.use_io_binding

        if self.config.session_options:
            kwargs["session_options"] = SessionOptions()
            for key, value in self.config.session_options.items():
                setattr(kwargs["session_options"], key, value)

        if self.config.provider_options:
            kwargs["provider_options"] = self.config.provider_options

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

    def optimize_onnx_files(self) -> None:
        self.logger.info("\t+ Attempting optimization")
        self.optimized_model = os.path.join(self.tmpdir.name, "optimized")
        self.logger.info("\t+ Processing optimization config")
        if self.config.auto_optimization is not None:
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=self.config.auto_optimization,
                for_gpu=(self.config.device == "cuda"),
                **self.config.auto_optimization_config,
            )
        elif self.config.optimization:
            optimization_config = OptimizationConfig(
                optimize_for_gpu=(self.config.device == "cuda"), **self.config.optimization_config
            )
        self.logger.info("\t+ Creating optimizer")
        optimizer = ORTOptimizer.from_pretrained(self.config.model, file_names=self.onnx_files_names)
        self.logger.info("\t+ Optimizing ORTModel")
        optimizer.optimize(
            optimization_config,
            save_dir=self.optimized_model,
            # TODO: add support for these
            use_external_data_format=None,
            one_external_file=True,
            file_suffix="",
        )

        if self.pretrained_processor is not None:
            self.pretrained_processor.save_pretrained(self.optimized_model)
        if self.pretrained_config is not None:
            self.pretrained_config.save_pretrained(self.optimized_model)

    def quantize_onnx_files(self) -> None:
        self.logger.info("\t+ Attempting quantization")
        self.quantized_model = f"{self.tmpdir.name}/quantized_model"

        if self.is_calibrated and len(self.onnx_files_names) > 1:
            raise NotImplementedError(
                "Calibrated/Static Quantization is not supported for models with multiple components. "
                f"Found {len(self.onnx_files_names)} components."
            )

        self.logger.info("\t+ Processing quantization config")
        if self.config.auto_quantization is not None:
            auto_quantization_config = format_quantization_config(self.config.auto_quantization_config)
            auto_quantization_class = getattr(AutoQuantizationConfig, self.config.auto_quantization)
            quantization_config = auto_quantization_class(**auto_quantization_config)
        elif self.config.quantization:
            quantization_config = format_quantization_config(self.config.quantization_config)
            quantization_config = QuantizationConfig(**quantization_config)

        if self.is_calibrated:
            self.logger.info("\t+ Generating calibration dataset")
            dataset_shapes = {"dataset_size": 2, "sequence_length": 2, "num_choices": 2}
            calibration_dataset = DatasetGenerator(
                task=self.config.task, dataset_shapes=dataset_shapes, model_shapes=self.model_shapes
            )()
            columns_to_be_removed = list(set(calibration_dataset.column_names) - set(self.pretrained_model.input_names))
            calibration_dataset = calibration_dataset.remove_columns(columns_to_be_removed)

            self.logger.info("\t+ Processing calibration config")
            if self.config.auto_calibration is not None:
                self.logger.info("\t+ Processing calibration config")
                auto_calibration_method = getattr(AutoCalibrationConfig, self.config.auto_calibration)
                calibration_config = auto_calibration_method(calibration_dataset, **self.config.auto_calibration_config)
            elif self.config.calibration:
                self.logger.info("\t+ Processing calibration config")
                calibration_config = format_calibration_config(self.config.calibration_config)
                calibration_config = CalibrationConfig(
                    dataset_name="calibration_dataset",
                    dataset_split=calibration_dataset.split,
                    dataset_num_samples=calibration_dataset.num_rows,
                    dataset_config_name=calibration_dataset.config_name,
                    **self.config.calibration_config,
                )

        for onnx_file_name in self.onnx_files_names:
            self.logger.info(f"\t+ Creating quantizer for {onnx_file_name}")
            quantizer = ORTQuantizer.from_pretrained(self.config.model, file_name=onnx_file_name)

            if self.is_calibrated:
                self.logger.info("\t+ Fitting calibration tensors range")
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

            self.logger.info("\t+ Quantizing model")
            quantizer.quantize(
                save_dir=self.quantized_model,
                quantization_config=quantization_config,
                calibration_tensors_range=calibration_tensors_range,
                # TODO: add support for these (maybe)
                use_external_data_format=False,
                preprocessor=None,
                file_suffix="",
            )

        if self.pretrained_processor is not None:
            self.pretrained_processor.save_pretrained(self.quantized_model)
        if self.pretrained_config is not None:
            self.pretrained_config.save_pretrained(self.quantized_model)

    @property
    def split_between_processes(self) -> bool:
        return is_torch_distributed_available() and torch.distributed.is_initialized()

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.split_between_processes:
            with Accelerator().split_between_processes(inputs=inputs, apply_padding=False) as process_inputs:
                inputs = process_inputs

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.config.device)

        for input_name in list(inputs.keys()):
            if hasattr(self.pretrained_model, "input_names") and input_name not in self.pretrained_model.input_names:
                inputs.pop(input_name)

        return inputs

    @torch.inference_mode()
    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.forward(**inputs, **kwargs)

    @torch.inference_mode()
    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(**inputs, **kwargs)

    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(**inputs, **kwargs)

    @torch.inference_mode()
    def call(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model(**inputs, **kwargs)
