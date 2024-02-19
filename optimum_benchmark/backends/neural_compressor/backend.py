import gc
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict

import torch
from hydra.utils import get_class
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor.quantization import INCQuantizer
from transformers.modeling_utils import no_init_weights
from transformers.utils import ModelOutput
from transformers.utils.logging import set_verbosity_error

from ...generators.dataset_generator import DatasetGenerator
from ..base import Backend
from ..transformers_utils import randomize_weights
from .config import INCConfig
from .utils import TASKS_TO_INCMODELS

# disable transformers logging
set_verbosity_error()

LOGGER = getLogger("neural-compressor")


class INCBackend(Backend[INCConfig]):
    NAME: str = "neural-compressor"

    def __init__(self, config: INCConfig):
        super().__init__(config)
        self.validate_task()

        LOGGER.info("\t+ Creating backend temporary directory")
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

    def validate_task(self) -> None:
        if self.config.task not in TASKS_TO_INCMODELS:
            raise NotImplementedError(f"INCBackend does not support task {self.config.task}")

        self.incmodel_class = get_class(TASKS_TO_INCMODELS[self.config.task])
        LOGGER.info(f"Using INCModel class {self.incmodel_class.__name__}")

    def load_automodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading AutoModel from pretrained")
        self.pretrained_model = self.automodel_class.from_pretrained(self.config.model, **self.config.hub_kwargs)

    def create_no_weights_model(self) -> None:
        LOGGER.info("\t+ Creating no weights model state_dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        LOGGER.info("\t+ Creating no weights model directory")
        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights")
        os.makedirs(self.no_weights_model, exist_ok=True)

        LOGGER.info("\t+ Saving no weights model pretrained config")
        self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

        LOGGER.info("\t+ Saving no weights model state_dict")
        torch.save(state_dict, os.path.join(self.no_weights_model, "pytorch_model.bin"))

    def load_automodel_with_no_weights(self) -> None:
        self.create_no_weights_model()

        with no_init_weights():
            original_model = self.config.model
            self.config.model = self.no_weights_model
            LOGGER.info("\t+ Loading no weights model")
            self.load_automodel_from_pretrained()
            self.config.model = original_model

        LOGGER.info("\t+ Randomizing model weights")
        randomize_weights(self.pretrained_model)
        LOGGER.info("\t+ Tying model weights")
        self.pretrained_model.tie_weights()

    def load_incmodel_from_pretrained(self) -> None:
        LOGGER.info("\t+ Loading INCModel from pretrained")
        self.pretrained_model = self.incmodel_class.from_pretrained(self.config.model, **self.config.hub_kwargs)

    def load_incmodel_with_no_weights(self) -> None:
        self.create_no_weights_model()

        with no_init_weights():
            original_model = self.config.model
            self.config.model = self.no_weights_model
            LOGGER.info("\t+ Loading no weights model")
            self.load_incmodel_from_pretrained()
            self.config.model = original_model

        LOGGER.info("\t+ Randomizing model weights")
        randomize_weights(self.pretrained_model.model)
        LOGGER.info("\t+ Tying model weights")
        self.pretrained_model.model.tie_weights()

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
            model=self.pretrained_model,
            task=self.config.task,
            seed=self.config.seed,
            # TODO: add support for these
            calibration_fn=None,
            eval_fn=None,
        )

        if self.config.calibration:
            LOGGER.info("\t+ Generating calibration dataset")
            dataset_shapes = {"dataset_size": 1, "sequence_length": 1, **self.model_shapes}
            calibration_dataset = DatasetGenerator(
                task=self.config.task, dataset_shapes=dataset_shapes, model_shapes=self.model_shapes
            )()
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
            LOGGER.info("\t+ Cleaning backend temporary directory")
            self.tmpdir.cleanup()

        gc.collect()
