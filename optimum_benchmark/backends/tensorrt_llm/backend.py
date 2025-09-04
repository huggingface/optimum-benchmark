import shutil
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Dict

from hydra.utils import get_class

from ..base import Backend
from .config import TRTLLMConfig
from .utils import MODEL_TYPE_TO_TRTLLMMODELS


class TRTLLMBackend(Backend[TRTLLMConfig]):
    NAME = "tensorrt-llm"

    def __init__(self, config: TRTLLMConfig):
        super().__init__(config)

        if self.config.model_type in MODEL_TYPE_TO_TRTLLMMODELS:
            self.trtllm_loader = get_class(MODEL_TYPE_TO_TRTLLMMODELS[self.config.model_type])
            self.logger.info(f"\t+ Using TRTLLMModel class {self.trtllm_loader.__name__}")
        else:
            raise NotImplementedError(f"TRTLLMBackend does not support model_type {self.config.model_type}")

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            self.logger.info("\t+ Creating no weights model")
            self.create_no_weights_model_slow()
            self.logger.info("\t+ Loading no weights model")
            self.load_model_with_no_weights()
        else:
            self.logger.info("\t+ Downloading pretrained model")
            self.download_pretrained_model()
            self.logger.info("\t+ Loading pretrained model")
            self.load_model_from_pretrained()

        try:
            self.tmpdir.cleanup()
        except Exception:
            shutil.rmtree(self.tmpdir.name, ignore_errors=True)

    def load_model_with_no_weights(self) -> None:
        original_model, self.config.model = self.config.model, self.no_weights_model_path.as_posix()
        self.load_model_from_pretrained()
        self.config.model = original_model

    def load_model_from_pretrained(self) -> None:
        self.pretrained_model = self.trtllm_loader.from_pretrained(
            self.config.model,
            **self.config.model_kwargs,
            **self.trtllm_kwargs,
        )

    @property
    def trtllm_kwargs(self):
        kwargs = {}

        if self.config.tp is not None:
            kwargs["tp"] = self.config.tp

        if self.config.pp is not None:
            kwargs["pp"] = self.config.pp

        if self.config.dtype is not None:
            kwargs["dtype"] = self.config.dtype

        if self.config.use_fp8 is not None:
            kwargs["use_fp8"] = self.config.use_fp8

        if self.config.world_size is not None:
            kwargs["world_size"] = self.config.world_size

        if self.config.gpus_per_node is not None:
            kwargs["gpus_per_node"] = self.config.gpus_per_node

        if self.config.max_input_len is not None:
            kwargs["max_input_len"] = self.config.max_input_len

        if self.config.max_output_len is not None:
            kwargs["max_output_len"] = self.config.max_output_len

        if self.config.max_batch_size is not None:
            kwargs["max_batch_size"] = self.config.max_batch_size

        if self.config.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.config.max_new_tokens

        if self.config.max_prompt_length is not None:
            kwargs["max_prompt_length"] = self.config.max_prompt_length

        if self.config.optimization_level is not None:
            kwargs["optimization_level"] = self.config.optimization_level

        if self.config.use_cuda_graph is not None:
            kwargs["use_cuda_graph"] = self.config.use_cuda_graph

        return kwargs

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            pad_token_id=kwargs.get("pad_token_id", 0),
            eos_token_id=kwargs.get("eos_token_id", 1),
            **kwargs,
        )

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            pad_token_id=kwargs.get("pad_token_id", 0),
            eos_token_id=kwargs.get("eos_token_id", 1),
            **kwargs,
        )
