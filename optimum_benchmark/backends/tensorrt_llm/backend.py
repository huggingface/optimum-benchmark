from logging import getLogger
from typing import Any, Dict

from hydra.utils import get_class
from transformers.utils import ModelOutput

from ..base import Backend
from .config import TRTLLMConfig
from .utils import MODEL_TYPE_TO_TRTLLMMODEL

LOGGER = getLogger("tensorrt-llm")


class TRTLLMBackend(Backend):
    NAME = "tensorrt-llm"

    def __init__(self, model: str, task: str, library: str, device: str, hub_kwargs: Dict[str, Any]):
        super().__init__(model, task, library, device, hub_kwargs)
        self.validate_device()
        self.validate_model_type()

    def validate_model_type(self) -> None:
        if self.model_type not in MODEL_TYPE_TO_TRTLLMMODEL:
            raise NotImplementedError(f"TRTLLMBackend does not support model_type {self.model_type}")

    def validate_device(self) -> None:
        if self.device != "cuda":
            raise NotImplementedError(f"TRTLLMBackend only supports device cuda, got {self.device}")

    def configure(self, config: TRTLLMConfig) -> None:
        super().configure(config)

        self.trtmodel_class = get_class(MODEL_TYPE_TO_TRTLLMMODEL[self.model_type])
        ortmodel_name = self.trtmodel_class.__name__
        LOGGER.info(
            f"\t+ Inferred TRTLLMModel class {ortmodel_name} for task {self.task} and model_type {self.model_type}"
        )

        self.load_trtmodel_from_pretrained()

    def load_trtmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.trtmodel_class.from_pretrained(
            self.model,
            tp=self.config.tp,
            pp=self.config.pp,
            dtype=self.config.dtype,
            use_fp8=self.config.use_fp8,
            world_size=self.config.world_size,
            gpus_per_node=self.config.gpus_per_node,
            use_cuda_graph=self.config.use_cuda_graph,
            optimization_level=self.config.optimization_level,
            max_prompt_length=self.config.max_prompt_length,
            max_batch_size=self.config.max_batch_size,
            max_new_tokens=self.config.max_new_tokens,
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
            input_ids=input.get("inputs", None),  # diff names
            attention_mask=input.get("attention_mask", None),
            # important for benchmarking
            max_new_tokens=kwargs.get("max_new_tokens", -1),
            min_length=kwargs.get("min_new_tokens", -1),  # why different ?
            num_beams=kwargs.get("num_beams", 1),
            # not really important but just in case
            repetition_penalty=kwargs.get("repetition_penalty", 0.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            pad_token_id=kwargs.get("pad_token_id", 0),
            bos_token_id=kwargs.get("bos_token_id", 1),
            eos_token_id=kwargs.get("eos_token_id", 2),
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 1.0),
            seed=kwargs.get("seed", 42),
        )
