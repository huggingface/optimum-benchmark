from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Dict

from hydra.utils import get_class

from ..base import Backend
from .config import TRTLLMConfig
from .utils import MODEL_TYPE_TO_TRTLLMMODEL


class TRTLLMBackend(Backend[TRTLLMConfig]):
    NAME = "tensorrt-llm"

    def __init__(self, config: TRTLLMConfig):
        super().__init__(config)

        if self.config.model_type in MODEL_TYPE_TO_TRTLLMMODEL:
            self.trtllm_loader = get_class(MODEL_TYPE_TO_TRTLLMMODEL[self.config.model_type])
            self.logger.info(f"\t+ Using TRTLLMModel class {self.trtllm_loader.__name__}")
        else:
            raise NotImplementedError(f"TRTLLMBackend does not support model_type {self.config.model_type}")

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        self.logger.info("\t+ Loading pretrained TRTLLMModel")
        self.load_trtmodel_from_pretrained()

        self.logger.info("\t+ Cleaning up backend temporary directory")
        self.tmpdir.cleanup()

    def load_trtmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.trtllm_loader.from_pretrained(
            self.config.model,
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
            max_beam_width=self.config.max_beam_width,
            **self.config.model_kwargs,
        )

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            min_length=kwargs.get("min_new_tokens", -1),
            max_new_tokens=kwargs.get("max_new_tokens", -1),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            pad_token_id=kwargs.get("pad_token_id", 0),
            bos_token_id=kwargs.get("bos_token_id", 1),
            eos_token_id=kwargs.get("eos_token_id", 2),
            temperature=kwargs.get("temperature", 1.0),
            num_beams=kwargs.get("num_beams", 1),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", 50),
            seed=kwargs.get("seed", 42),
        )

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            min_length=kwargs.get("min_new_tokens", -1),
            max_new_tokens=kwargs.get("max_new_tokens", -1),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            pad_token_id=kwargs.get("pad_token_id", 0),
            bos_token_id=kwargs.get("bos_token_id", 1),
            eos_token_id=kwargs.get("eos_token_id", 2),
            temperature=kwargs.get("temperature", 1.0),
            num_beams=kwargs.get("num_beams", 1),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", 50),
            seed=kwargs.get("seed", 42),
        )
