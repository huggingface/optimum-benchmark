import os
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Dict

import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from hydra.utils import get_class
from safetensors.torch import save_file

from ...task_utils import TEXT_GENERATION_TASKS
from ..base import Backend
from ..transformers_utils import fast_weights_init
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
            self.create_no_weights_model()
            self.logger.info("\t+ Loading no weights model")
            self.load_trtllm_with_no_weights()
        else:
            self.logger.info("\t+ Downloading pretrained model")
            self.download_pretrained_model()
            if self.config.task in TEXT_GENERATION_TASKS:
                self.logger.info("\t+ Preparing generation config")
                self.prepare_generation_config()
            self.logger.info("\t+ Loading pretrained model")
            self.load_trtllm_from_pretrained()

        self.logger.info("\t+ Cleaning up backend temporary directory")
        self.tmpdir.cleanup()

    def download_pretrained_model(self) -> None:
        with torch.device("meta"):
            self.automodel_loader.from_pretrained(self.config.model, **self.config.model_kwargs)

    def prepare_generation_config(self) -> None:
        self.generation_config.eos_token_id = None
        self.generation_config.pad_token_id = None

        model_cache_folder = f"models/{self.config.model}".replace("/", "--")
        model_cache_path = f"{HUGGINGFACE_HUB_CACHE}/{model_cache_folder}"
        snapshot_file = f"{model_cache_path}/refs/{self.config.model_kwargs.get('revision', 'main')}"
        snapshot_ref = open(snapshot_file, "r").read().strip()
        model_snapshot_path = f"{model_cache_path}/snapshots/{snapshot_ref}"
        self.generation_config.save_pretrained(save_directory=model_snapshot_path)

    def create_no_weights_model(self) -> None:
        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights_model")
        self.logger.info("\t+ Creating no weights model directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        self.logger.info("\t+ Creating no weights model state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()
        self.logger.info("\t+ Saving no weights model safetensors")
        safetensor = os.path.join(self.no_weights_model, "model.safetensors")
        save_file(tensors=state_dict, filename=safetensor, metadata={"format": "pt"})
        self.logger.info("\t+ Saving no weights model pretrained config")
        self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)
        self.logger.info("\t+ Saving no weights model pretrained processor")
        self.pretrained_processor.save_pretrained(save_directory=self.no_weights_model)
        # unlike Transformers, TRT-LLM won't accept any missing tensors so we need to materialize the model
        self.logger.info(f"\t+ Loading no weights model from {self.no_weights_model}")
        with fast_weights_init():
            self.pretrained_model = self.automodel_loader.from_pretrained(
                self.no_weights_model, **self.config.model_kwargs, device_map="auto", _fast_init=False
            )
        self.logger.info("\t+ Saving no weights model")
        self.pretrained_model.save_pretrained(save_directory=self.no_weights_model)
        del self.pretrained_model
        torch.cuda.empty_cache()

        if self.config.task in TEXT_GENERATION_TASKS:
            self.logger.info("\t+ Modifying generation config for fixed length generation")
            self.generation_config.eos_token_id = None
            self.generation_config.pad_token_id = None
            self.logger.info("\t+ Saving new pretrained generation config")
            self.generation_config.save_pretrained(save_directory=self.no_weights_model)

    def load_trtllm_with_no_weights(self) -> None:
        original_model, self.config.model = self.config.model, self.no_weights_model
        self.load_trtllm_from_pretrained()
        self.config.model = original_model

    def load_trtllm_from_pretrained(self) -> None:
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

        if self.config.use_cuda_graph is not None:
            kwargs["use_cuda_graph"] = self.config.use_cuda_graph

        if self.config.optimization_level is not None:
            kwargs["optimization_level"] = self.config.optimization_level

        if self.config.max_prompt_length is not None:
            kwargs["max_prompt_length"] = self.config.max_prompt_length

        if self.config.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.config.max_new_tokens

        if self.config.max_beam_width is not None:
            kwargs["max_beam_width"] = self.config.max_beam_width

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
