from logging import getLogger
from typing import Any, Dict

from hydra.utils import get_class
from transformers.utils import ModelOutput

from ..base import Backend
from .config import TRTConfig
from .utils import MODEL_TYPE_TO_TRTMODEL

LOGGER = getLogger("tensorrt")


class TRTBackend(Backend):
    NAME: str = "tensorrt"

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]) -> None:
        super().__init__(model, task, device, hub_kwargs)
        self.validate_device()
        self.validate_model_type()

    def validate_model_type(self) -> None:
        if self.model_type not in MODEL_TYPE_TO_TRTMODEL:
            raise NotImplementedError(f"TRTBackend does not support model_type {self.model_type}")

    def validate_device(self) -> None:
        if self.device != "cuda":
            raise NotImplementedError(f"TRTBackend only supports device cuda, got {self.device}")

    def configure(self, config: TRTConfig) -> None:
        super().configure(config)

        self.trtmodel_class = get_class(MODEL_TYPE_TO_TRTMODEL[self.model_type])
        ortmodel_name = self.trtmodel_class.__name__
        LOGGER.info(
            f"\t+ Inferred TRTModel class {ortmodel_name} for task {self.task} and model_type {self.model_type}"
        )

        # TODO: save engine path for reuse, then maybe re build with max_prompt_size
        self.load_trtmodel_from_pretrained()

    @property
    def trtmodel_kwargs(self) -> Dict[str, Any]:
        return {}

    def load_trtmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.trtmodel_class.from_pretrained(
            self.model,
            **self.trtmodel_kwargs,
            **self.hub_kwargs,
        )

    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model.generate(
            input_ids=input.get("input_ids", None),
            attention_mask=input.get("attention_mask", None),
            max_new_tokens=1,
        )

    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model.generate(
            # spelling args to avoid conflict
            input_ids=input.get("inputs", None),  # diff api
            attention_mask=input.get("attention_mask", None),
            max_new_tokens=kwargs.get("max_new_tokens", -1),
            min_length=kwargs.get("min_new_tokens", -1),  # diff api
            num_beams=kwargs.get("num_beams", 1),
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            seed=kwargs.get("seed", 42),
            pad_token_id=kwargs.get("pad_token_id", 0),
            bos_token_id=kwargs.get("bos_token_id", 1),
            eos_token_id=kwargs.get("eos_token_id", 2),
        )
