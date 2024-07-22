from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List

import torch
from datasets import Dataset
from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments
from transformers import TrainerCallback

from ..base import Backend
from ..peft_utils import apply_peft
from ..transformers_utils import fast_weights_init
from .config import TorchORTConfig


class TorchORTBackend(Backend[TorchORTConfig]):
    NAME: str = "torch-ort"

    def __init__(self, config: TorchORTConfig):
        super().__init__(config)

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            self.logger.info("\t+ Creating no weights AutoModel")
            self.create_no_weights_model()
            self.logger.info("\t+ Loading no weights AutoModel")
            self.load_automodel_with_no_weights()
        else:
            self.logger.info("\t+ Loading pretrained AutoModel")
            self.load_automodel_from_pretrained()

        if self.config.peft_type is not None:
            self.logger.info("\t+ Applying PEFT")
            self.pretrained_model = apply_peft(self.pretrained_model, self.config.peft_type, self.config.peft_config)

        self.logger.info("\t+ Cleaning up backend temporary directory")
        self.tmpdir.cleanup()

    def load_automodel_with_no_weights(self) -> None:
        original_model, self.config.model = self.config.model, self.no_weights_model

        with fast_weights_init():
            self.load_automodel_from_pretrained()

        self.logger.info("\t+ Tying model weights")
        self.pretrained_model.tie_weights()

        self.config.model = original_model

    def load_automodel_from_pretrained(self) -> None:
        self.pretrained_model = self.automodel_loader.from_pretrained(
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
