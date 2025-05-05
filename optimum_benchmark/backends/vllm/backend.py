import asyncio
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Union

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import save_model
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

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
            raise NotImplementedError(f"We only support text generation tasks for VLLM, but got {self.config.task}")

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
            self.logger.info("\t+ Loading pretrained model")
            self.load_model_from_pretrained()

        try:
            self.tmpdir.cleanup()
        except Exception:
            shutil.rmtree(self.tmpdir.name, ignore_errors=True)

    def download_pretrained_model(self) -> None:
        model_snapshot_folder = snapshot_download(
            self.config.model,
            revision=self.config.model_kwargs.get("revision", None),
            cache_dir=self.config.model_kwargs.get("cache_dir", None),
            force_download=self.config.model_kwargs.get("force_download", False),
            local_files_only=self.config.model_kwargs.get("local_files_only", False),
        )

        if self.config.task in TEXT_GENERATION_TASKS:
            self.generation_config.eos_token_id = None
            self.generation_config.pad_token_id = None
            self.generation_config.save_pretrained(save_directory=model_snapshot_folder)

    def create_no_weights_model(self) -> None:
        model_path = Path(hf_hub_download(self.config.model, filename="config.json", cache_dir=self.tmpdir.name)).parent
        save_model(model=torch.nn.Linear(1, 1), filename=model_path / "model.safetensors", metadata={"format": "pt"})
        self.pretrained_processor.save_pretrained(save_directory=model_path)
        self.pretrained_config.save_pretrained(save_directory=model_path)

        with fast_weights_init():
            # unlike Transformers, TXI won't accept any missing tensors so we need to materialize the model
            dummy = self.automodel_loader.from_pretrained(model_path, device_map="auto", **self.config.model_kwargs)
            dummy.save_pretrained(model_path)
            del dummy

        torch.cuda.empty_cache()

        if self.config.task in TEXT_GENERATION_TASKS:
            self.generation_config.eos_token_id = None
            self.generation_config.pad_token_id = None
            self.generation_config.save_pretrained(save_directory=model_path)

        self.no_weights_model = model_path.as_posix()

    def load_model_with_no_weights(self) -> None:
        original_model, self.config.model = self.config.model, self.no_weights_model
        self.load_model_from_pretrained()
        self.config.model = original_model

    def load_model_from_pretrained(self) -> None:
        if self.config.serving_mode == "offline":
            self.pretrained_model = LLMEngine.from_engine_args(EngineArgs(**self.vllm_kwargs))
        else:
            self.pretrained_model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.vllm_kwargs))

    @property
    def vllm_kwargs(self):
        return {
            "model": self.config.model,
            "tokenizer": self.config.processor,
            "device": self.config.device,
            **self.config.engine_args,
        }

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.task in TEXT_GENERATION_TASKS:
            inputs = {"prompts": self.pretrained_processor.batch_decode(inputs["input_ids"])}
        else:
            raise NotImplementedError(f"vLLM does not support task {self.config.task}")

        return inputs

    def batch_offline_engine_generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        for i, prompt in enumerate(inputs["prompts"]):
            self.pretrained_model.add_request(
                prompt=prompt,
                request_id=str(i),
                params=self.get_sampling_params(kwargs),
            )

        while self.pretrained_model.has_unfinished_requests():
            self.pretrained_model.step()

    def get_sampling_params(self, kwargs: Dict[str, Any]) -> SamplingParams:
        return SamplingParams(
            ignore_eos=True,
            detokenize=True,
            seed=self.config.seed,
            n=kwargs.get("num_return_sequences"),
            max_tokens=kwargs.get("max_new_tokens"),
            min_tokens=kwargs.get("min_new_tokens"),
            logits_processors=kwargs.get("logits_processors", None),
        )

    async def single_online_engine_generate(self, prompt: str, request_id: str, kwargs: Dict[str, Any]) -> Any:
        stream = await self.pretrained_model.add_request(
            prompt=prompt,
            request_id=request_id,
            params=self.get_sampling_params(kwargs),
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
