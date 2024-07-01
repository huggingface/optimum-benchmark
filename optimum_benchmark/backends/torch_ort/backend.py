import os
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments
from safetensors.torch import save_file
from transformers import TrainerCallback

from ..base import Backend
from ..peft_utils import apply_peft
from ..transformers_utils import random_init_weights
from .config import TorchORTConfig


class TorchORTBackend(Backend[TorchORTConfig]):
    NAME: str = "torch-ort"

    def __init__(self, config: TorchORTConfig):
        super().__init__(config)
        self.validate_library()

        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            self.logger.info("\t+ Loading no weights AutoModel")
            self.load_automodel_with_no_weights()
        else:
            self.logger.info("\t+ Loading pretrained AutoModel")
            self.load_automodel_from_pretrained()

        if self.config.peft_type is not None:
            self.logger.info("\t+ Applying PEFT")
            self.pretrained_model = apply_peft(self.pretrained_model, self.config.peft_type, self.config.peft_config)

        self.tmpdir.cleanup()

    def validate_library(self) -> None:
        if self.config.library == "transformers":
            self.logger.info(f"Using AutoModel class {self.automodel_class.__name__}")
        else:
            raise NotImplementedError(f"TorchORTBackend does not support {self.config.library} library")

    def create_no_weights_model(self) -> None:
        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights_model")
        self.logger.info("\t+ Creating no weights model directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        self.logger.info("\t+ Creating no weights model state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()
        self.logger.info("\t+ Saving no weights model safetensors")
        safetensors = os.path.join(self.no_weights_model, "model.safetensors")
        save_file(tensors=state_dict, filename=safetensors, metadata={"format": "pt"})

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

    def load_automodel_from_pretrained(self) -> None:
        self.pretrained_model = self.automodel_class.from_pretrained(
            self.config.model, **self.automodel_kwargs, **self.config.model_kwargs
        ).to(self.config.device)

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.torch_dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation

        return kwargs

    def train(
        self,
        training_dataset: Dataset,
        training_arguments: Dict[str, Any],
        training_callbacks: List[TrainerCallback],
        training_data_collator: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    ):
        self.logger.info(f"\t+ Wrapping training arguments with {ORTTrainingArguments.__name__}")
        training_arguments = ORTTrainingArguments(**training_arguments)
        self.logger.info(f"\t+ Wrapping model with {ORTTrainer.__name__}")
        trainer = ORTTrainer(
            model=self.pretrained_model,
            args=training_arguments,
            callbacks=training_callbacks,
            train_dataset=training_dataset,
            data_collator=training_data_collator,
        )
        self.logger.info("\t+ Starting training")
        trainer.train()
        self.logger.info("\t+ Finished training")
