import gc
import os
from collections import OrderedDict
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict

import torch
from hydra.utils import get_class
from onnxruntime import SessionOptions
from optimum.amd.ryzenai import AutoQuantizationConfig, QuantizationConfig, RyzenAIOnnxQuantizer
from optimum.exporters.onnx import main_export
from safetensors.torch import save_file
from transformers.utils.logging import set_verbosity_error

from ...generators.dataset_generator import DatasetGenerator
from ...task_utils import IMAGE_PROCESSING_TASKS, TEXT_GENERATION_TASKS
from ..base import Backend
from ..transformers_utils import random_init_weights
from .config import RyzenAIConfig
from .utils import TASKS_TO_RYZENAIMODEL

# disable transformers logging
set_verbosity_error()

LOGGER = getLogger("ryzenai")


class RyzenAIBackend(Backend[RyzenAIConfig]):
    NAME: str = "ryzenai"

    def __init__(self, config: RyzenAIConfig) -> None:
        super().__init__(config)

        self.ryzenaimodel_class = get_class(TASKS_TO_RYZENAIMODEL[self.config.task])
        LOGGER.info(f"\t+ Using RyzenAIModel class {self.ryzenaimodel_class.__name__}")

        self.session_options = SessionOptions()
        if self.config.session_options:
            LOGGER.info("\t+ Processing session options")
            for key, value in self.config.session_options.items():
                setattr(self.session_options, key, value)

        LOGGER.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.is_quantized:
            if self.config.no_weights:
                LOGGER.info("\t+ Loading no weights AutoModel")
                self.load_automodel_with_no_weights()
            else:
                LOGGER.info("\t+ Loading pretrained AutoModel")
                self.load_automodel_from_pretrained()

            original_model, original_export = self.config.model, self.config.export

            LOGGER.info("\t+ Exporting model to ONNX")
            self.export_onnx_model()
            self.config.model = self.exported_model

            LOGGER.info("\t+ Applying RyzenAI quantization")
            self.quantize_onnx_files()
            self.config.model = self.quantized_model

            self.config.export = False
            LOGGER.info("\t+ Loading quantized RyzenAIModel")
            self.load_ryzenaimodel_from_pretrained()

            self.config.model, self.config.export = original_model, original_export

        elif self.config.no_weights:
            raise NotImplementedError("`no_weights` is only supported when RyzenAI model is quantized from scratch")
        else:
            LOGGER.info("\t+ Loading pretrained RyzenAIModel")
            self.load_ryzenaimodel_from_pretrained()

        self.tmpdir.cleanup()

    def create_no_weights_model(self) -> None:
        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights_model")
        LOGGER.info("\t+ Creating no weights model directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        LOGGER.info("\t+ Creating no weights model state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()
        LOGGER.info("\t+ Saving no weights model safetensors")
        safetensors = os.path.join(self.no_weights_model, "model.safetensors")
        save_file(tensors=state_dict, filename=safetensors, metadata={"format": "pt"})

        if self.config.library == "transformers":
            LOGGER.info("\t+ Saving no weights model pretrained config")
            self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

    def load_automodel_with_no_weights(self) -> None:
        LOGGER.info("\t+ Creating no weights model")
        self.create_no_weights_model()
        with random_init_weights():
            original_model, self.config.model = self.config.model, self.no_weights_model
            LOGGER.info("\t+ Loading no weights AutoModel")
            self.load_automodel_from_pretrained()
            self.config.model = original_model

        if self.config.library == "transformers":
            LOGGER.info("\t+ Tying weights")
            self.pretrained_model.tie_weights()

    def export_onnx_model(self) -> None:
        self.exported_model = f"{self.tmpdir.name}/exported_model"
        main_export(
            model_name_or_path=self.config.model,
            output=self.exported_model,
            task=self.config.task,
            no_dynamic_axes=True,
            batch_size=1,
            opset=13,
        )

    def load_automodel_from_pretrained(self) -> None:
        if self.config.library == "timm":
            self.pretrained_model = self.automodel_class(model_name=self.config.model)
        else:
            self.pretrained_model = self.automodel_class.from_pretrained(self.config.model, **self.config.hub_kwargs)

    def load_ryzenaimodel_with_no_weights(self) -> None:
        LOGGER.info("\t+ Creating no weights model")
        self.create_no_weights_model()
        with random_init_weights():
            original_model, original_export = self.config.model, self.config.export

            self.config.model, self.config.export = self.no_weights_model, False
            LOGGER.info("\t+ Loading no weights RyzenAIModel")
            self.load_ryzenaimodel_from_pretrained()

            self.config.model, self.config.export = original_model, original_export

    def load_ryzenaimodel_from_pretrained(self) -> None:
        self.pretrained_model = self.ryzenaimodel_class.from_pretrained(
            self.config.model,
            export=self.config.export,
            provider=self.config.provider,
            vaip_config=self.config.vaip_config,
            **self.config.hub_kwargs,
            **self.ryzenaimodel_kwargs,
        )

    def quantize_onnx_files(self) -> None:
        LOGGER.info("\t+ Attempting quantization")
        self.quantized_model = f"{self.tmpdir.name}/quantized_model"

        LOGGER.info("\t+ Processing quantization config")
        if self.config.auto_quantization is not None:
            auto_quantization_class = getattr(AutoQuantizationConfig, self.config.auto_quantization)
            quantization_config = auto_quantization_class(**self.config.auto_quantization_config)
        elif self.config.quantization:
            quantization_config = QuantizationConfig(**quantization_config)

        LOGGER.info("\t+ Generating calibration dataset")
        dataset_shapes = {"dataset_size": 1, "sequence_length": 1, **self.model_shapes}
        calibration_dataset = DatasetGenerator(
            task=self.config.task, dataset_shapes=dataset_shapes, model_shapes=self.model_shapes
        )()
        calibration_dataset = calibration_dataset.remove_columns(["labels"])

        for onnx_file_name in self.onnx_files_names:
            LOGGER.info(f"\t+ Creating quantizer for {onnx_file_name}")
            quantizer = RyzenAIOnnxQuantizer.from_pretrained(self.config.model, file_name=onnx_file_name)

            LOGGER.info("\t+ Quantizing model")
            quantizer.quantize(
                save_dir=self.quantized_model,
                quantization_config=quantization_config,
                dataset=calibration_dataset,
                # TODO: add support for these (maybe)
                batch_size=1,
                file_suffix="",
            )

        if self.pretrained_processor is not None:
            self.pretrained_processor.save_pretrained(self.quantized_model)
        if self.config.library == "transformers":
            self.pretrained_config.save_pretrained(self.quantized_model)

    @property
    def onnx_files_names(self):
        assert os.path.isdir(self.config.model), f"{self.config.model} is not a directory"
        return [file for file in os.listdir(self.config.model) if file.endswith(".onnx")]

    @property
    def is_quantized(self) -> bool:
        return self.config.quantization or self.config.auto_quantization

    @property
    def ryzenaimodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.task in TEXT_GENERATION_TASKS:
            kwargs["use_cache"] = self.config.use_cache

        return kwargs

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = super().prepare_inputs(inputs)

        if not self.config.export and self.config.task in IMAGE_PROCESSING_TASKS:
            # original amd ryzenai models expects channels first
            inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 3, 1)

        return inputs

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.forward(**inputs, **kwargs)

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(**inputs, **kwargs)

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            self.tmpdir.cleanup()

        gc.collect()
