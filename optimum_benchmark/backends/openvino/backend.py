import gc
import inspect
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict

import torch
from hydra.utils import get_class
from openvino.runtime import properties
from optimum.intel.openvino import OVConfig as OVQuantizationConfig  # naming conflict
from optimum.intel.openvino import OVQuantizer
from safetensors.torch import save_file
from transformers.modeling_utils import no_init_weights
from transformers.utils.logging import set_verbosity_error

from ...generators.dataset_generator import DatasetGenerator
from ..base import Backend
from ..pytorch.utils import randomize_weights
from .config import OVConfig
from .utils import TASKS_TO_OVMODEL

# disable transformers logging
set_verbosity_error()

LOGGER = getLogger("openvino")


class OVBackend(Backend[OVConfig]):
    NAME: str = "openvino"

    def __init__(self, model: str, task: str, library: str, device: str, hub_kwargs: Dict[str, Any]) -> None:
        super().__init__(model, task, library, device, hub_kwargs)
        self.validate_device()
        self.validate_task()

    def validate_task(self) -> None:
        if self.task not in TASKS_TO_OVMODEL:
            raise NotImplementedError(f"OVBackend does not support task {self.task}")

    def validate_device(self) -> None:
        if self.device != "cpu":
            raise ValueError(f"OVBackend only supports CPU devices, got {self.device}")

    def configure(self, config: OVConfig) -> None:
        super().configure(config)

        self.ovmodel_class = get_class(TASKS_TO_OVMODEL[self.task])
        ovmodel_name = self.ovmodel_class.__name__
        LOGGER.info(f"\t+ Inferred class {ovmodel_name} for task {self.task} and model_type {self.model_type}")

        self.openvino_config = self.config.openvino_config.copy()
        if self.config.inter_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting inter_op_num_threads to {self.config.inter_op_num_threads}")
            self.openvino_config[properties.inference_num_threads()] = self.config.inter_op_num_threads

        if self.config.intra_op_num_threads is not None:
            raise NotImplementedError("OVBackend does not support intra_op_num_threads")

        if self.library == "diffusers" and self.config.no_weights:
            raise NotImplementedError("Diffusers models can't be loaded with no weights")

        self.tmpdir = TemporaryDirectory()

        if self.config.quantization:
            if self.config.no_weights:
                self.load_automodel_with_no_weights()
            else:
                self.load_automodel_from_pretrained()
            self.quantize_automodel()
            self.delete_pretrained_model()
            self.load_ovmodel_from_pretrained()
        elif self.config.no_weights:
            self.load_ovmodel_with_no_weights()
        else:
            self.load_ovmodel_from_pretrained()

        self.tmpdir.cleanup()

    def load_ovmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.ovmodel_class.from_pretrained(
            self.model,
            ov_config=self.openvino_config,
            export=self.config.export and not self.config.quantization,
            # in case of quantization, the model will be exported by the quantizer
            **self.ovmodel_kwargs,
            **self.hub_kwargs,
        )

    def load_ovmodel_with_no_weights(self) -> None:
        no_weights_model = os.path.join(self.tmpdir.name, "no_weights")

        LOGGER.info("\t+ Loading AutoModel with no weights")
        self.load_automodel_with_no_weights()
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading OVModel with no weights")
        with no_init_weights():
            original_model = self.model
            self.model = no_weights_model
            self.load_ovmodel_from_pretrained()
            self.model = original_model

    def load_automodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading AutoModel from pretrained")
        self.pretrained_model = self.automodel_class.from_pretrained(self.model, **self.hub_kwargs)

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

    @property
    def ovmodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.is_text_generation_model():
            kwargs["use_cache"] = self.config.use_cache
            kwargs["use_merged"] = self.config.use_merged

        return kwargs

    def quantize_automodel(self) -> None:
        LOGGER.info("\t+ Attempting quantization")
        quantized_model_path = f"{self.tmpdir.name}/quantized"
        LOGGER.info("\t+ Processing quantization config")
        quantization_config = OVQuantizationConfig(**self.config.quantization_config)
        LOGGER.info("\t+ Creating quantizer")
        quantizer = OVQuantizer.from_pretrained(self.pretrained_model, task=self.task, seed=self.config.seed)

        if self.config.calibration:
            LOGGER.info("\t+ Generating calibration dataset")
            dataset_shapes = {"dataset_size": 1, "sequence_length": 1, **self.model_shapes}
            calibration_dataset = DatasetGenerator(task=self.task, dataset_shapes=dataset_shapes).generate()
            columns_to_be_removed = list(set(calibration_dataset.column_names) - set(quantizer._export_input_names))
            calibration_dataset = calibration_dataset.remove_columns(columns_to_be_removed)

        LOGGER.info("\t+ Quantizing model")
        quantizer.quantize(
            save_directory=quantized_model_path,
            quantization_config=quantization_config,
            calibration_dataset=calibration_dataset,
            # TODO: add support for these (maybe)
            remove_unused_columns=True,
            data_collator=None,
            weights_only=False,
            file_name=None,
            batch_size=1,
        )
        self.model = quantized_model_path

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.library == "diffusers":
            return {"prompt": inputs["prompt"]}

        return inputs

    def prepare_for_inference(self, **kwargs) -> None:
        if self.config.reshape:
            static_shapes = {
                key: value
                for key, value in kwargs.items()
                if key in inspect.getfullargspec(self.pretrained_model.reshape).args
            }
            if ("height" in static_shapes or "width" in static_shapes) and ("sequence_length" in static_shapes):
                static_shapes["sequence_length"] = kwargs.get("num_channels", 3)

            LOGGER.info(f"\t+ Reshaping model with static shapes: {static_shapes}")
            self.pretrained_model.reshape(**static_shapes)

        if self.config.half:
            LOGGER.info("\t+ Converting model to half precision")
            self.pretrained_model.half()

        if self.config.reshape or self.config.half:
            LOGGER.info("\t+ Compiling model")
            self.pretrained_model.compile()

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            self.tmpdir.cleanup()

        gc.collect()
