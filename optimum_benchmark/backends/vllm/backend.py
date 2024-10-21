import asyncio
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, Union

import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from safetensors.torch import save_file
from vllm import AsyncEngineArgs, AsyncLLMEngine, EngineArgs, LLMEngine, SamplingParams

from ...task_utils import TEXT_GENERATION_TASKS
from ..base import Backend
from ..transformers_utils import fast_weights_init
from .config import VLLMConfig


class VLLMBackend(Backend[VLLMConfig]):
    NAME: str = "vllm"
    pretrained_model: Union[LLMEngine, AsyncLLMEngine]

    def __init__(self, config: VLLMConfig) -> None:
        super().__init__(config)

        if self.config.task not in TEXT_GENERATION_TASKS:
            raise NotImplementedError(f"vLLM does not support task {self.config.task}")

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            self.logger.info("\t+ Creating no weights model")
            self.create_no_weights_model()
            self.logger.info("\t+ Loading no weights model")
            self.load_model_with_no_weights()
        else:
            self.logger.info("\t+ Downloading pretrained model")
            self.download_pretrained_model()
            if self.config.task in TEXT_GENERATION_TASKS:
                self.logger.info("\t+ Preparing generation config")
                self.prepare_generation_config()
            self.logger.info("\t+ Loading pretrained model")
            self.load_model_from_pretrained()

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
        self.logger.info("\t+ Saving new pretrained generation config")
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
        # unlike Transformers, vLLM won't accept any missing tensors so we need to materialize the model
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

    def load_model_with_no_weights(self) -> None:
        original_model, self.config.model = self.config.model, self.no_weights_model
        self.logger.info("\t+ Loading no weights model")
        self.load_model_from_pretrained()
        self.config.model = original_model

    def load_model_from_pretrained(self) -> None:
        if self.config.serving_mode == "offline":
            self.pretrained_model = LLMEngine.from_engine_args(EngineArgs(**self.config.to_engine_args()))
        else:
            self.pretrained_model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.config.to_engine_args()))

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.task in TEXT_GENERATION_TASKS:
            inputs = {"prompts": self.pretrained_processor.batch_decode(inputs["input_ids"])}
        else:
            raise NotImplementedError(f"vLLM does not support task {self.config.task}")

        return inputs

    def batch_offline_engine_generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        for i, prompt in enumerate(inputs["prompts"]):
            self.pretrained_model.add_request(
                inputs=prompt,
                request_id=str(i),
                params=self.get_sampling_params(kwargs),
            )

        while self.pretrained_model.has_unfinished_requests():
            self.pretrained_model.step()

    def get_sampling_params(self, kwargs: Dict[str, Any]) -> SamplingParams:
        params = SamplingParams(
            ignore_eos=True,
            detokenize=True,
            seed=self.config.seed,
            n=kwargs.get("num_return_sequences"),
            max_tokens=kwargs.get("max_new_tokens"),
            min_tokens=kwargs.get("min_new_tokens"),
            logits_processors=kwargs.get("logits_processors", None),
        )
        # following huggingface transformers implementation
        # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/generation/beam_search.py#L534
        if kwargs.get("num_beams") > 1:
            params.logprobs = 2 * kwargs.get("num_beams")
        return params

    async def single_online_engine_generate(self, prompt: str, request_id: str, kwargs: Dict[str, Any]) -> Any:
        stream = await self.pretrained_model.add_request(
            inputs=prompt,
            request_id=request_id,
            params=self.get_sampling_params(),
        )

        async for _ in stream:
            pass

    async def batch_online_engine_generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        tasks = [
            self.single_online_engine_generate(prompt, str(i), kwargs) for i, prompt in enumerate(inputs["prompts"])
        ]
        await asyncio.gather(*tasks)

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.serving_mode == "offline":
            self.batch_offline_engine_generate(inputs, kwargs)
        else:
            asyncio.run(self.batch_online_engine_generate(inputs, kwargs))

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        if self.config.serving_mode == "offline":
            self.batch_offline_engine_generate(inputs, kwargs)
        else:
            asyncio.run(self.batch_online_engine_generate(inputs, kwargs))
