import os
from logging import getLogger
from typing import Any, Dict

from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import TextGenerationResponse
from transformers import AutoTokenizer

import docker

from ..base import Backend
from .config import TGIConfig

# bachend logger
LOGGER = getLogger("text-generation-inference")


class TGIBackend(Backend[TGIConfig]):
    NAME: str = "text-generation-inference"

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]):
        super().__init__(model, task, device, hub_kwargs)
        self.validate_task()

        automodel = self.automodel_class.__name__
        LOGGER.info(f"\t+ Infered AutoModel class {automodel} for task {self.task} and model_type {self.model_type}")

    def validate_task(self) -> None:
        if self.task not in ["text-generation", "text2text-generation"]:
            raise NotImplementedError(f"TGI does not support task {self.task}")

    def configure(self, config: TGIConfig) -> None:
        super().configure(config)
        self.config = config

        LOGGER.info("\t+ Starting Docker client")
        self.docker_client = docker.from_env()

        LOGGER.info("\t+ Building TGI command")
        self.command = ["--model-id", self.model]
        if self.config.quantization_scheme:
            self.command += ["--quantize", self.config.quantization_scheme]

        if self.device.type == "cuda":
            device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", self.device.index or 0)

            LOGGER.info(f"\t+ Starting TGI container on CUDA device(s): {device_ids}")
            self.tgi_container = self.docker_client.containers.run(
                image=f"ghcr.io/huggingface/text-generation-inference:{self.config.version}",
                command=self.command,
                shm_size=self.config.shm_size,
                volumes={self.config.volume: {"bind": "/data", "mode": "rw"}},
                ports={"80/tcp": (self.config.address, self.config.port)},
                device_requests=[docker.types.DeviceRequest(device_ids=[str(device_ids)], capabilities=[["gpu"]])],
                detach=True,
            )
        else:
            LOGGER.info("\t+ Starting TGI container on CPU device")
            self.tgi_container = self.docker_client.containers.run(
                image=f"ghcr.io/huggingface/text-generation-inference:{self.config.version}",
                command=self.command,
                shm_size=self.config.shm_size,
                volumes={f"{os.getcwd()}/data": {"bind": "/data", "mode": "rw"}},
                ports={f"{self.config.port}": 80},
                detach=True,
            )

        LOGGER.info("\t+ Waiting for TGI server to be ready")
        for line in self.tgi_container.logs(stream=True):
            tgi_log = line.decode("utf-8").strip()

            if not tgi_log:
                continue
            else:
                LOGGER.info(f"\t\t+ TGI log: {tgi_log}")

            if "Connected" in tgi_log:
                break

        LOGGER.info("\t+ TGI server is ready")
        self.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

        LOGGER.info("\t+ Creating InferenceClient")
        self.client = InferenceClient(
            model=f"http://{self.config.address}:{self.config.port}", timeout=self.config.timeout
        )

    def prepare_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if len(input["input_ids"]) > 1:
            raise NotImplementedError(
                "TGI client does not support batched inputs and is specifically designed for latency optimization. "
                "Please use a batch size of 1."
            )
        input = {"prompt": self.tokenizer.decode(input["input_ids"].squeeze().tolist())}
        return input

    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> TextGenerationResponse:
        return self.client.text_generation(**input, max_new_tokens=1, decoder_input_details=True, details=True)

    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> TextGenerationResponse:
        return self.client.text_generation(**input, max_new_tokens=kwargs["max_new_tokens"], details=True)

    def clean(self) -> None:
        super().clean()

        LOGGER.info("\t+ Stoping TGI container")
        self.tgi_container.stop()
        self.tgi_container.wait()

        LOGGER.info("\t+ Pruning docker containers")
        self.docker_client.containers.prune()

        LOGGER.info("\t+ Closing docker client")
        self.docker_client.close()
