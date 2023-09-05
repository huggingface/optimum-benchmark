import os
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict

from huggingface_hub import InferenceClient

if TYPE_CHECKING:
    from huggingface_hub.inference._text_generation import TextGenerationResponse

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

        # check if image exists and pull it if needed
        LOGGER.info("\t+ Checking if TGI image exists")
        try:
            self.docker_client.images.get(f"{self.config.image}:{self.config.version}")
        except docker.errors.APIError:
            LOGGER.info("\t+ Pulling TGI image")
            self.docker_client.images.pull(f"{self.config.image}:{self.config.version}")

        LOGGER.info("\t+ Building TGI command")
        self.command = [
            "--model-id",
            self.model,
            "--revision",
            self.hub_kwargs["revision"],
            "--dtype",
            str(self.config.torch_dtype),
        ]

        if self.hub_kwargs.get("trust_remote_code", False):
            self.command.append("--trust-remote-code")

        if self.config.quantization_scheme is not None:
            self.command.extend(["--quantize", self.config.quantization_scheme])

        if self.config.disable_custom_kernels:
            self.command.append("--disable-custom-kernels")

        if self.device.type == "cuda":
            device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", self.device.index or 0)
            LOGGER.info(f"\t+ Starting TGI container on CUDA device(s): {device_ids}")
            device_requests = [docker.types.DeviceRequest(device_ids=[str(device_ids)], capabilities=[["gpu"]])]
        else:
            LOGGER.info("\t+ Starting TGI container on CPU device")
            device_requests = None

        self.tgi_container = self.docker_client.containers.run(
            image=f"{self.config.image}:{self.config.version}",
            command=self.command,
            shm_size=self.config.shm_size,
            volumes={self.config.volume: {"bind": "/data", "mode": "rw"}},
            ports={"80/tcp": (self.config.address, self.config.port)},
            device_requests=device_requests,
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
                LOGGER.info("\t+ TGI server is ready")
                break

        LOGGER.info("\t+ Creating InferenceClient")
        self.client = InferenceClient(model=f"http://{self.config.address}:{self.config.port}")

    def prepare_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.stream:
            return {"prompt": self.pretrained_processor.batch_decode(input["input_ids"].tolist())}
        elif len(input["input_ids"]) > 1:
            raise NotImplementedError(
                "TGI does not support batched requests in non-stream mode. "
                "Please set `backend.stream=true` or use a batch size of 1."
            )
        else:
            return {"prompt": self.pretrained_processor.decode(input["input_ids"].squeeze().tolist())}

    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> "TextGenerationResponse":
        if self.config.stream:
            output = []
            with ThreadPoolExecutor(max_workers=len(input["prompt"])) as executor:
                futures = [
                    executor.submit(
                        self.client.text_generation,
                        decoder_input_details=True,
                        max_new_tokens=1,
                        prompt=prompt,
                        details=True,
                        stream=True,
                    )
                    for prompt in input["prompt"]
                ]
            for future in futures:
                output.append(future.result())
            return output
        else:
            return self.client.text_generation(
                input["prompt"], max_new_tokens=1, decoder_input_details=True, details=True
            )

    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> "TextGenerationResponse":
        if self.config.stream:
            output = []
            with ThreadPoolExecutor(max_workers=len(input["prompt"])) as executor:
                futures = [
                    executor.submit(
                        self.client.text_generation,
                        max_new_tokens=kwargs["max_new_tokens"],
                        do_sample=kwargs["do_sample"],
                        prompt=prompt,
                        details=True,
                    )
                    for prompt in input["prompt"]
                ]
            for future in futures:
                output.append(future.result())
            return output
        else:
            return self.client.text_generation(
                input["prompt"], do_sample=kwargs["do_sample"], max_new_tokens=kwargs["max_new_tokens"], details=True
            )

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tgi_container"):
            LOGGER.info("\t+ Stoping TGI container")
            self.tgi_container.stop()
            LOGGER.info("\t+ Waiting for TGI container to stop")
            self.tgi_container.wait()

        if hasattr(self, "docker_client"):
            LOGGER.info("\t+ Closing docker client")
            self.docker_client.close()
