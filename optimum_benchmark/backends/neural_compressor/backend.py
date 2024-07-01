import os
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Dict

import torch
from hydra.utils import get_class
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
from optimum.intel.neural_compressor.quantization import INCQuantizer

from ...generators.dataset_generator import DatasetGenerator
from ..base import Backend
from ..transformers_utils import random_init_weights
from .config import INCConfig
from .utils import TASKS_TO_INCMODELS


class INCBackend(Backend[INCConfig]):
    NAME: str = "neural-compressor"

    def __init__(self, config: INCConfig):
        super().__init__(config)
        self.validate_task()

        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.ptq_quantization:
            if self.config.no_weights:
                self.logger.info("\t+ Loading no weights AutoModel")
                self.load_automodel_with_no_weights()
            else:
                self.logger.info("\t+ Loading pretrained AutoModel")
                self.load_automodel_from_pretrained()

            self.logger.info("\t+ Applying post-training quantization")
            self.quantize_automodel()

            self.logger.info("\t+ Loading quantized INCModel")
            original_model, self.config.model = self.config.model, self.quantized_model
            self.load_incmodel_from_pretrained()
            self.config.model = original_model

        elif self.config.no_weights:
            self.logger.info("\t+ Loading no weights INCModel")
            self.load_incmodel_with_no_weights()

        else:
            self.logger.info("\t+ Loading pretrained INCModel")
            self.load_incmodel_from_pretrained()

        self.tmpdir.cleanup()

    def validate_task(self) -> None:
        if self.config.task not in TASKS_TO_INCMODELS:
            raise NotImplementedError(f"INCBackend does not support task {self.config.task}")

        self.incmodel_class = get_class(TASKS_TO_INCMODELS[self.config.task])
        self.logger.info(f"Using INCModel class {self.incmodel_class.__name__}")

    def load_automodel_from_pretrained(self) -> None:
        self.pretrained_model = self.automodel_class.from_pretrained(self.config.model, **self.config.model_kwargs)

    def create_no_weights_model(self) -> None:
        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights_model")
        self.logger.info("\t+ Creating no weights model directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        self.logger.info("\t+ Creating no weights model state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()
        self.logger.info("\t+ Saving no weights model pytorch_model.bin")
        torch.save(state_dict, os.path.join(self.no_weights_model, "pytorch_model.bin"))

        if self.config.library == "transformers":
            self.logger.info("\t+ Saving no weights model pretrained config")
            self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

    def load_automodel_with_no_weights(self) -> None:
        self.logger.info("\t+ Creating no weights model")
        self.create_no_weights_model()

        with random_init_weights():
            original_model, self.config.model = self.config.model, self.no_weights_model
            self.logger.info("\t+ Loading no weights AutoModel")
            self.load_automodel_from_pretrained()
            self.config.model = original_model

        self.logger.info("\t+ Tying model weights")
        self.pretrained_model.tie_weights()

    def load_incmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.incmodel_class.from_pretrained(self.config.model, **self.config.model_kwargs)

    def load_incmodel_with_no_weights(self) -> None:
        self.logger.info("\t+ Creating no weights model")
        self.create_no_weights_model()

        with random_init_weights():
            original_model, self.config.model = self.config.model, self.no_weights_model
            self.logger.info("\t+ Loading no weights INCModel")
            self.load_incmodel_from_pretrained()
            self.config.model = original_model

        self.logger.info("\t+ Tying model weights")
        self.pretrained_model.model.tie_weights()

    def quantize_automodel(self) -> None:
        self.quantized_model = f"{self.tmpdir.name}/quantized_model"
        self.logger.info("\t+ Processing quantization config")
        ptq_quantization_config = self.config.ptq_quantization_config.copy()
        ptq_quantization_config["accuracy_criterion"] = AccuracyCriterion(
            **ptq_quantization_config["accuracy_criterion"]
        )
        ptq_quantization_config["tuning_criterion"] = TuningCriterion(**ptq_quantization_config["tuning_criterion"])
        ptq_quantization_config = PostTrainingQuantConfig(**ptq_quantization_config)
        self.logger.info("\t+ Creating quantizer")
        quantizer = INCQuantizer.from_pretrained(
            model=self.pretrained_model,
            task=self.config.task,
            seed=self.config.seed,
            # TODO: add support for these
            calibration_fn=None,
            eval_fn=None,
        )

        if self.config.calibration:
            self.logger.info("\t+ Generating calibration dataset")
            dataset_shapes = {"dataset_size": 1, "sequence_length": 1, **self.model_shapes}
            calibration_dataset = DatasetGenerator(
                task=self.config.task, dataset_shapes=dataset_shapes, model_shapes=self.model_shapes
            )()
            columns_to_be_removed = list(set(calibration_dataset.column_names) - set(quantizer._signature_columns))
            calibration_dataset = calibration_dataset.remove_columns(columns_to_be_removed)
        else:
            calibration_dataset = None

        self.logger.info("\t+ Quantizing model")
        quantizer.quantize(
            save_directory=self.quantized_model,
            calibration_dataset=calibration_dataset,
            quantization_config=ptq_quantization_config,
            # TODO: add support for these
            remove_unused_columns=True,
            data_collator=None,
            file_name=None,
            batch_size=1,
        )

    @torch.inference_mode()
    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model(**input, **kwargs)

    @torch.inference_mode()
    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(**inputs, **kwargs)

    @torch.inference_mode()
    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(**input, **kwargs)
