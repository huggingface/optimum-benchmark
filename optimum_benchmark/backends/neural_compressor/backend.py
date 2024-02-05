import os
import gc
from typing import Any, Dict
from logging import getLogger
from tempfile import TemporaryDirectory

import torch
from hydra.utils import get_class
from transformers.utils import ModelOutput
from transformers.modeling_utils import no_init_weights
from transformers.utils.logging import set_verbosity_error
from optimum.intel.neural_compressor.quantization import INCQuantizer
from neural_compressor.config import (
    PostTrainingQuantConfig,
    AccuracyCriterion,
    TuningCriterion,
)

from ...generators.dataset_generator import DatasetGenerator
from .utils import TASKS_TO_INCMODELS
from .config import INCConfig
from ..base import Backend

# disable transformers logging
set_verbosity_error()

LOGGER = getLogger("neural-compressor")


class INCBackend(Backend[INCConfig]):
    NAME: str = "neural-compressor"

    def __init__(self, config: INCConfig):
        super().__init__(config)
        self.validate_task()

        self.incmodel_class = get_class(TASKS_TO_INCMODELS[self.config.task])
        LOGGER.info(f"Using INCModel class {self.incmodel_class.__name__}")

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

    def validate_task(self) -> None:
        if self.config.task not in TASKS_TO_INCMODELS:
            raise NotImplementedError(
                f"INCBackend does not support task {self.config.task}"
            )

    def load_automodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading AutoModel from pretrained")
        self.pretrained_model = self.automodel_class.from_pretrained(
            self.config.model, **self.config.hub_kwargs
        )

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
            original_model = self.config.model
            self.config.model = no_weights_model
            self.load_automodel_from_pretrained()
            self.config.model = original_model

    def load_incmodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading INCModel from pretrained")
        self.pretrained_model = self.incmodel_class.from_pretrained(
            self.config.model, **self.config.hub_kwargs
        )

    def load_incmodel_with_no_weights(self) -> None:
        no_weights_model = os.path.join(self.tmpdir.name, "no_weights")

        LOGGER.info("\t+ Loading AutoModel with no weights")
        self.load_automodel_with_no_weights()
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading INCModel with no weights")
        with no_init_weights():
            original_model = self.config.model
            self.config.model = no_weights_model
            self.load_incmodel_from_pretrained()
            self.config.model = original_model

    def quantize_automodel(self) -> None:
        LOGGER.info("\t+ Attempting to quantize model")
        quantized_model_path = f"{self.tmpdir.name}/quantized"
        LOGGER.info("\t+ Processing quantization config")
        ptq_quantization_config = self.config.ptq_quantization_config.copy()
        ptq_quantization_config["accuracy_criterion"] = AccuracyCriterion(
            **ptq_quantization_config["accuracy_criterion"]
        )
        ptq_quantization_config["tuning_criterion"] = TuningCriterion(
            **ptq_quantization_config["tuning_criterion"]
        )
        ptq_quantization_config = PostTrainingQuantConfig(**ptq_quantization_config)
        LOGGER.info("\t+ Creating quantizer")
        quantizer = INCQuantizer.from_pretrained(
            model=self.pretrained_model,
            task=self.config.task,
            seed=self.config.seed,
            # TODO: add support for these
            calibration_fn=None,
            eval_fn=None,
        )

        if self.config.calibration:
            LOGGER.info("\t+ Generating calibration dataset")
            dataset_shapes = {
                "dataset_size": 1,
                "sequence_length": 1,
                **self.model_shapes,
            }
            calibration_dataset = DatasetGenerator(
                task=self.config.task, dataset_shapes=dataset_shapes
            ).generate()
            columns_to_be_removed = list(
                set(calibration_dataset.column_names)
                - set(quantizer._signature_columns)
            )
            calibration_dataset = calibration_dataset.remove_columns(
                columns_to_be_removed
            )
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
        self.config.model = quantized_model_path

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.library == "diffusers":
            return {"prompt": inputs["prompt"]}

        return inputs

    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model(**input, **kwargs)

    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model.generate(**input, **kwargs)

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            self.tmpdir.cleanup()

        gc.collect()
