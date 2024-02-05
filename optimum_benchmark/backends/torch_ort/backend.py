import gc
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import TrainerCallback, TrainerState
from transformers.modeling_utils import no_init_weights
from transformers.utils.logging import set_verbosity_error

from ..transformers_utils import randomize_weights
from ..peft_utils import get_peft_config_class
from .config import TorchORTConfig
from ..base import Backend

# disable transformers logging
set_verbosity_error()

LOGGER = getLogger("onnxruntime")


class TorchORTBackend(Backend[TorchORTConfig]):
    NAME: str = "onnxruntime"

    def __init__(self, config: TorchORTConfig):
        super().__init__(config)

        LOGGER.info(f"Using AutoModel: {self.automodel_class.__name__}")

        self.tmpdir = TemporaryDirectory()

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

        self.tmpdir.cleanup()

    def create_no_weights_model(self) -> None:
        LOGGER.info("\t+ Creating no weights model directory")
        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights")
        os.makedirs(self.no_weights_model, exist_ok=True)

        LOGGER.info("\t+ Saving pretrained config")
        self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

        LOGGER.info("\t+ Creating no weights model state_dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()

        LOGGER.info("\t+ Saving no weights model state_dict")
        save_file(
            filename=os.path.join(self.no_weights_model, "model.safetensors"),
            metadata={"format": "pt"},
            tensors=state_dict,
        )

    def load_automodel_with_no_weights(self) -> None:
        self.create_no_weights_model()

        with no_init_weights():
            original_model = self.config.model
            self.config.model = self.no_weights_model
            LOGGER.info("\t+ Loading no weights model")
            self.load_automodel_from_pretrained()
            self.config.model = original_model

        LOGGER.info("\t+ Randomizing weights")
        randomize_weights(self.pretrained_model)
        LOGGER.info("\t+ Tying model weights after randomization")
        self.pretrained_model.tie_weights()

    def load_automodel_from_pretrained(self) -> None:
        self.pretrained_model = self.automodel_class.from_pretrained(
            self.config.model,
            **self.automodel_kwargs,
            **self.config.hub_kwargs,
        ).to(self.config.device)

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.torch_dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

        return kwargs

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
