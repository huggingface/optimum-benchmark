import gc
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict

import torch
from hydra.utils import get_class
from neural_compressor.config import (
    AccuracyCriterion,
    PostTrainingQuantConfig,
    TuningCriterion,
)
from optimum.intel.neural_compressor.quantization import INCQuantizer
from transformers.modeling_utils import no_init_weights
from transformers.utils.logging import set_verbosity_error

from ...generators.dataset_generator import DatasetGenerator
from ..base import Backend
from .config import INCConfig
from .utils import TASKS_TO_INCMODELS

LOGGER = getLogger("neural-compressor")

# disable transformers logging
set_verbosity_error()


class INCBackend(Backend[INCConfig]):
    NAME: str = "neural-compressor"

    def __init__(self, model: str, task: str, library: str, device: str, hub_kwargs: Dict[str, Any]) -> None:
        super().__init__(model, task, library, device, hub_kwargs)
        self.validate_device()
        self.validate_task()

        self.incmodel_class = get_class(TASKS_TO_INCMODELS[self.task])
        LOGGER.info(
            f"Inferred INCModel {self.incmodel_class.__name__} for task {self.task} and model_type {self.model_type}"
        )

    def validate_device(self) -> None:
        if self.device != "cpu":
            raise ValueError(f"INCBackend only supports CPU devices, got {self.device}")

    def validate_task(self) -> None:
        if self.task not in TASKS_TO_INCMODELS:
            raise NotImplementedError(f"INCBackend does not support task {self.task}")

    def configure(self, config: INCConfig) -> None:
        super().configure(config)

        self.tmpdir = TemporaryDirectory()

        if self.config.ptq_quantization:
            if self.config.no_weights:
                self.load_automodel_with_no_weights()
            else:
                self.load_automodel_from_pretrained()
            self.quantize_automodel()
            self.delete_pretrained_model()
            self.load_incmodel_from_pretrained()
        elif self.config.no_weights:
            self.load_incmodel_with_no_weights()
        else:
            self.load_incmodel_from_pretrained()

        self.tmpdir.cleanup()

    def load_automodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading AutoModel from pretrained")
        self.pretrained_model = self.automodel_class.from_pretrained(self.model, **self.hub_kwargs)

    def load_automodel_with_no_weights(self) -> None:
        no_weights_model = os.path.join(self.tmpdir.name, "no_weights")

        if not os.path.exists(no_weights_model):
            LOGGER.info("\t+ Creating no weights model directory")
            os.makedirs(no_weights_model)

        LOGGER.info("\t+ Saving pretrained config")
        self.pretrained_config.save_pretrained(save_directory=no_weights_model)

        LOGGER.info("\t+ Creating no weights model")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        LOGGER.info("\t+ Saving no weights model")
        torch.save(state_dict, os.path.join(no_weights_model, "pytorch_model.bin"))

        LOGGER.info("\t+ Loading no weights model")
        with no_init_weights():
            original_model = self.model
            self.model = no_weights_model
            self.load_automodel_from_pretrained()
            self.model = original_model

    def load_incmodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading INCModel from pretrained")
        self.pretrained_model = self.incmodel_class.from_pretrained(self.model, **self.hub_kwargs)

    def load_incmodel_with_no_weights(self) -> None:
        no_weights_model = os.path.join(self.tmpdir.name, "no_weights")

        LOGGER.info("\t+ Loading AutoModel with no weights")
        self.load_automodel_with_no_weights()
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading INCModel with no weights")
        with no_init_weights():
            original_model = self.model
            self.model = no_weights_model
            self.load_incmodel_from_pretrained()
            self.model = original_model

    def quantize_automodel(self) -> None:
        LOGGER.info("\t+ Attempting to quantize model")
        quantized_model_path = f"{self.tmpdir.name}/quantized"
        LOGGER.info("\t+ Processing quantization config")
        ptq_quantization_config = self.config.ptq_quantization_config.copy()
        ptq_quantization_config["accuracy_criterion"] = AccuracyCriterion(
            **ptq_quantization_config["accuracy_criterion"]
        )
        ptq_quantization_config["tuning_criterion"] = TuningCriterion(**ptq_quantization_config["tuning_criterion"])
        ptq_quantization_config = PostTrainingQuantConfig(**ptq_quantization_config)
        LOGGER.info("\t+ Creating quantizer")
        quantizer = INCQuantizer.from_pretrained(
            task=self.task,
            seed=self.config.seed,
            model=self.pretrained_model,
            # TODO: add support for these
            calibration_fn=None,
            eval_fn=None,
        )

        if self.config.calibration:
            LOGGER.info("\t+ Generating calibration dataset")
            dataset_shapes = {"dataset_size": 1, "sequence_length": 1, **self.model_shapes}
            calibration_dataset = DatasetGenerator(task=self.task, dataset_shapes=dataset_shapes).generate()
            columns_to_be_removed = list(set(calibration_dataset.column_names) - set(quantizer._signature_columns))
            calibration_dataset = calibration_dataset.remove_columns(columns_to_be_removed)
        else:
            calibration_dataset = None

        LOGGER.info("\t+ Quantizing model")
        quantizer.quantize(
            save_directory=quantized_model_path,
            calibration_dataset=calibration_dataset,
            quantization_config=ptq_quantization_config,
            # TODO: add support for these
            remove_unused_columns=True,
            data_collator=None,
            file_name=None,
            batch_size=1,
        )
        self.model = quantized_model_path

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.library == "diffusers":
            return {"prompt": inputs["prompt"]}

        return inputs

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            self.tmpdir.cleanup()

        gc.collect()
