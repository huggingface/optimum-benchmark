import gc
import os
from collections import OrderedDict
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict

import torch
from hydra.utils import get_class
from safetensors.torch import save_file
from transformers.utils.logging import set_verbosity_error

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
        self.validate_task()

        LOGGER.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            LOGGER.info("\t+ Loading no weights RyzenAIModel")
            self.load_ryzenaimodel_with_no_weights()
        else:
            LOGGER.info("\t+ Loading pretrained RyzenAIModel")
            self.load_ryzenaimodel_from_pretrained()

        self.tmpdir.cleanup()

    def validate_task(self) -> None:
        if self.config.task not in TASKS_TO_RYZENAIMODEL:
            raise NotImplementedError(f"RyzenAIBackend does not support task {self.config.task}")

        self.ryzenaimodel_class = get_class(TASKS_TO_RYZENAIMODEL[self.config.task])
        LOGGER.info(f"\t+ Using RyzenAIModel class {self.ryzenaimodel_class.__name__}")

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

        LOGGER.info("\t+ Tying model weights")
        self.pretrained_model.tie_weights()

    def load_automodel_from_pretrained(self) -> None:
        self.pretrained_model = self.automodel_class.from_pretrained(self.config.model, **self.config.hub_kwargs)

    def load_ryzenaimodel_with_no_weights(self) -> None:
        LOGGER.info("\t+ Creating no weights model")
        self.create_no_weights_model()

        with random_init_weights():
            original_model, self.config.model = self.config.model, self.no_weights_model
            original_export, self.config.export = self.config.export, True
            LOGGER.info("\t+ Loading no weights RyzenAIModel")
            self.load_ryzenaimodel_from_pretrained()
            self.config.model = original_model
            self.config.export = original_export

    def load_ryzenaimodel_from_pretrained(self) -> None:
        self.pretrained_model = self.ryzenaimodel_class.from_pretrained(
            self.config.model,
            export=self.config.export,
            provider=self.config.provider,
            vaip_config=self.config.vaip_config,
            **self.config.hub_kwargs,
            **self.ryzenaimodel_kwargs,
        )

    @property
    def ryzenaimodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.task in TEXT_GENERATION_TASKS:
            kwargs["use_cache"] = self.config.use_cache

        return kwargs

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = super().prepare_inputs(inputs)

        if self.config.task in IMAGE_PROCESSING_TASKS:
            # channels last
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
