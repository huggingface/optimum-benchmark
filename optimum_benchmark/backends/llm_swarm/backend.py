import asyncio
from typing import Any, Dict, List, Tuple

import torch
from huggingface_hub import AsyncInferenceClient
from llm_swarm import LLMSwarm
from llm_swarm import LLMSwarmConfig as LLMSwarmCfg

from ...task_utils import TEXT_GENERATION_TASKS
from ..base import Backend
from .config import LLMSwarmConfig


class LLMSwarmBackend(Backend[LLMSwarmConfig]):
    NAME: str = "llm-swarm"

    def __init__(self, config: LLMSwarmConfig) -> None:
        super().__init__(config)
        self.validate_task()

        self.logger.info("\t+ Downloading pretrained model")
        self.download_pretrained_model()
        self.logger.info("\t+ Preparing generation config")
        self.prepare_generation_config()
        self.logger.info("\t+ Loading pretrained model")
        self.load_model_from_pretrained()

    def validate_task(self) -> None:
        if self.config.task not in TEXT_GENERATION_TASKS:
            raise NotImplementedError(f"LLM Swarm does not support task {self.config.task}")

    def load_model_from_pretrained(self) -> None:
        self.llm_swarm_config = LLMSwarmCfg(
            gpus=self.config.gpus,
            model=self.config.model,
            instances=self.config.instances,
            inference_engine=self.config.inference_engine,
            slurm_template_path=self.config.slurm_template_path,
            load_balancer_template_path=self.config.load_balancer_template_path,
            per_instance_max_parallel_requests=self.config.per_instance_max_parallel_requests,
            revision=self.config.model_kwargs.get("revision", "main"),
            debug_endpoint=self.config.debug_endpoint,
        )
        self.llm_swarm = LLMSwarm(self.llm_swarm_config).__enter__()
        self.client = AsyncInferenceClient(self.llm_swarm.endpoint)

    def download_pretrained_model(self) -> None:
        with torch.device("meta"):
            self.automodel_class.from_pretrained(self.config.model, **self.config.model_kwargs)

    def prepare_generation_config(self) -> None:
        self.generation_config.eos_token_id = -100
        self.generation_config.pad_token_id = -100

        model_cache_folder = f"models/{self.config.model}".replace("/", "--")
        model_cache_path = f"{self.config.volume}/{model_cache_folder}"
        snapshot_file = f"{model_cache_path}/refs/{self.config.model_kwargs.get('revision', 'main')}"
        snapshot_ref = open(snapshot_file, "r").read().strip()
        model_snapshot_path = f"{model_cache_path}/snapshots/{snapshot_ref}"
        self.logger.info("\t+ Saving new pretrained generation config")
        self.generation_config.save_pretrained(save_directory=model_snapshot_path)

    def prepare_inputs(
        self, inputs: Dict[str, Any], input_shapes: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, input_shapes = super().prepare_inputs(inputs, input_shapes)

        if "inputs" in inputs:
            inputs = {"prompt": self.pretrained_processor.batch_decode(inputs["inputs"].tolist())}
        elif "input_ids" in inputs:
            inputs = {"prompt": self.pretrained_processor.batch_decode(inputs["input_ids"].tolist())}
        else:
            raise ValueError("inputs must contain either input_ids or inputs")

        return inputs, input_shapes

    async def single_client_call(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        return await self.client.text_generation(prompt, max_new_tokens=kwargs.get("max_new_tokens", 1))

    async def batch_client_call(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return await asyncio.gather(*(self.single_client_call(p, kwargs) for p in inputs["prompt"]))

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return asyncio.run(self.batch_client_call(inputs, kwargs))

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return asyncio.run(self.batch_client_call(inputs, kwargs))

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return asyncio.run(self.batch_client_call(inputs, kwargs))
