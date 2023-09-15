import os
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from accelerate import init_empty_weights
from huggingface_hub import InferenceClient
from transformers import GenerationConfig

if TYPE_CHECKING:
    from huggingface_hub.inference._text_generation import TextGenerationResponse

import docker
import docker.errors
import docker.types

from ..base import Backend
from ..pytorch.utils import randomize_weights
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

        if self.config.no_weights:
            # creates dummy model
            self.load_model_from_config()
            self.save_model_snapshot()
        else:
            self.load_model_from_pretrained()
        self.delete_pretrained_model()

        LOGGER.info("\t+ Modifying generation config")
        self.modify_generation_config()

        LOGGER.info("\t+ Starting Docker client")
        self.docker_client = docker.from_env()

        try:
            LOGGER.info("\t+ Checking if TGI image exists")
            self.docker_client.images.get(f"{self.config.image}:{self.config.version}")
        except docker.errors.ImageNotFound:
            LOGGER.info("\t+ TGI image not found, pulling it")
            self.docker_client.images.pull(f"{self.config.image}:{self.config.version}")

        LOGGER.info("\t+ Building TGI command")
        self.command = [
            "--model-id",
            self.model,
            "--revision",
            self.hub_kwargs["revision"],
        ]

        if self.config.quantization_scheme is not None:
            self.command.extend(["--quantize", self.config.quantization_scheme])
        if self.config.torch_dtype is not None:
            self.command.extend(["--torch-dtype", self.config.torch_dtype])
        if self.hub_kwargs.get("trust_remote_code", False):
            self.command.append("--trust-remote-code")
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
            elif "Connected" in tgi_log:
                LOGGER.info("\t+ TGI server is ready")
                break
            else:
                LOGGER.info(f"\t {tgi_log}")

        LOGGER.info("\t+ Creating InferenceClient")
        self.client = InferenceClient(model=f"http://{self.config.address}:{self.config.port}")

    def load_model_from_config(self) -> None:
        LOGGER.info("\t+ Initializing empty weights model on device: meta")
        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                config=self.pretrained_config,
                torch_dtype=getattr(torch, self.config.torch_dtype),
                trust_remote_code=self.hub_kwargs.get("trust_remote_code", False),
            )
        # could add model dispatching to accelerate saving and support bigger models
        LOGGER.info(f"\t+ Materializing model on device: {self.device}")
        self.pretrained_model.to_empty(device=self.device)
        LOGGER.info("\t+ Randomizing model weights")
        randomize_weights(self.pretrained_model)
        LOGGER.info("\t+ Tying weights")
        self.pretrained_model.tie_weights()

    @property
    def model_snapshot_path(self) -> str:
        model_cache_folder = f"models/{self.model}".replace("/", "--")
        model_cache_path = f"{self.config.volume}/{model_cache_folder}"
        snapshot_ref = open(f"{model_cache_path}/refs/{self.hub_kwargs.get('revision', 'main')}", "r").read().strip()
        return f"{model_cache_path}/snapshots/{snapshot_ref}"

    def save_model_snapshot(self) -> None:
        LOGGER.info("\t+ Saving pretrained model snapshot")
        self.pretrained_model.save_pretrained(self.model_snapshot_path, safe_serialization=True)

    def load_model_from_pretrained(self) -> None:
        LOGGER.info("\t+ Downloading pretrained model")
        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_pretrained(self.model, **self.hub_kwargs)

    def modify_generation_config(self) -> None:
        # this should, theorically, make the generated output's sequence length fully controlled by max_new_tokens
        # instead of stopping at the first eos_token_id/pad_token_id
        generation_config = GenerationConfig.from_pretrained(self.model, **self.hub_kwargs)
        generation_config.eos_token_id = -100
        generation_config.pad_token_id = -101
        generation_config.save_pretrained(self.model_snapshot_path)

    def prepare_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return {"prompt": self.pretrained_processor.batch_decode(input["input_ids"].tolist())}

    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> List["TextGenerationResponse"]:
        output = []
        with ThreadPoolExecutor(max_workers=len(input["prompt"])) as executor:
            futures = [
                executor.submit(
                    self.client.text_generation,
                    decoder_input_details=True,
                    prompt=input["prompt"][i],
                    max_new_tokens=1,
                    details=True,
                )
                for i in range(len(input["prompt"]))
            ]
        for future in futures:
            output.append(future.result())

        return output

    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> List["TextGenerationResponse"]:
        output = []
        with ThreadPoolExecutor(max_workers=len(input["prompt"])) as executor:
            futures = [
                executor.submit(
                    self.client.text_generation,
                    max_new_tokens=kwargs["max_new_tokens"],
                    do_sample=kwargs["do_sample"],
                    prompt=input["prompt"][i],
                    details=True,
                )
                for i in range(len(input["prompt"]))
            ]
        for i in range(len(input["prompt"])):
            output.append(futures[i].result())

        return output

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
