import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import torch
from huggingface_hub import InferenceClient, snapshot_download
from huggingface_hub.inference._text_generation import TextGenerationResponse
from safetensors.torch import save_model

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

    def validate_task(self) -> None:
        if self.task not in ["text-generation", "text2text-generation"]:
            raise NotImplementedError(f"TGI does not support task {self.task}")

    def configure(self, config: TGIConfig) -> None:
        super().configure(config)

        automodel = self.automodel_class.__name__
        LOGGER.info(f"Inferred AutoModel class {automodel} for task {self.task} and model_type {self.model_type}")

        if self.config.no_weights:
            self.load_model_with_no_weights()
        else:
            self.load_model_from_pretrained()

    def load_model_from_pretrained(self) -> None:
        if not self.config.no_weights:
            LOGGER.info("\t+ Downloading pretrained model")
            snapshot_download(self.model, **self.hub_kwargs)

        LOGGER.info("\t+ Modifying pretrained generation config")
        self.pretrained_generation_config.eos_token_id = -100
        self.pretrained_generation_config.pad_token_id = -101

        LOGGER.info("\t+ Saving new pretrained generation config")
        model_cache_folder = f"models/{self.model}".replace("/", "--")
        model_cache_path = f"{self.config.volume}/{model_cache_folder}"

        snapshot_ref = open(f"{model_cache_path}/refs/{self.hub_kwargs.get('revision', 'main')}", "r").read().strip()

        model_snapshot_path = f"{model_cache_path}/snapshots/{snapshot_ref}"
        self.pretrained_generation_config.save_pretrained(save_directory=model_snapshot_path)

        self.start_tgi_server()

    def load_model_with_no_weights(self) -> None:
        self.tmp_dir = TemporaryDirectory()

        original_model = self.model
        no_weights_model = os.path.join(self.tmp_dir.name, "no_weights")

        LOGGER.info("\t+ Creating no weights model directory")
        os.makedirs(no_weights_model, exist_ok=True)

        LOGGER.info(f"\t+ Saving pretrained config to {no_weights_model}")
        self.pretrained_config.save_pretrained(save_directory=no_weights_model)

        LOGGER.info(f"\t+ Saving pretrained tokenizer to {no_weights_model}")
        self.pretrained_processor.save_pretrained(save_directory=no_weights_model)

        LOGGER.info(f"\t+ Saving no weights model to {no_weights_model}")
        save_model(
            filename=os.path.join(no_weights_model, "model.safetensors"),
            model=torch.nn.Linear(1, 1),
            metadata={"format": "pt"},
        )

        # unlike transformers api, TGI won't accept an empty model.safetensors
        # so we need to materialize the model and resave it
        LOGGER.info(f"\t+ Loading no weights model from {no_weights_model}")
        self.pretrained_model = self.automodel_class.from_pretrained(
            no_weights_model,
            **self.hub_kwargs,
            device_map="auto",
            _fast_init=False,
        )

        LOGGER.info("\t+ Randomizing weights of no weights model")
        randomize_weights(self.pretrained_model)

        LOGGER.info(f"\t+ Saving randomized weights model to {no_weights_model}")
        self.pretrained_model.save_pretrained(no_weights_model)
        del self.pretrained_model

        LOGGER.info(f"\t+ Saving generation config to {no_weights_model}")
        self.pretrained_generation_config.eos_token_id = -100
        self.pretrained_generation_config.pad_token_id = -101
        self.pretrained_generation_config.save_pretrained(save_directory=no_weights_model)

        self.model = no_weights_model
        self.start_tgi_server()
        self.model = original_model

    def start_tgi_server(self) -> None:
        LOGGER.info("\t+ Starting Python Docker client")
        self.docker_client = docker.from_env()

        try:
            LOGGER.info("\t+ Checking if TGI image exists")
            self.docker_client.images.get(self.config.image)
        except docker.errors.ImageNotFound:
            LOGGER.info("\t+ TGI image not found, pulling it")
            self.docker_client.images.pull(self.config.image)

        env = {}
        if os.environ.get("HUGGING_FACE_HUB_TOKEN", None) is not None:
            env["HUGGING_FACE_HUB_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

        LOGGER.info("\t+ Building TGI command")
        self.command = [
            "--model-id",
            self.model,
            "--revision",
            self.hub_kwargs["revision"],
        ]

        if self.config.sharded is not None:
            self.command.extend(["--sharded", str(self.config.sharded).lower()])
        if self.config.num_shard is not None:
            self.command.extend(["--num-shard", str(self.config.num_shard)])
        if self.config.quantization_scheme is not None:
            self.command.extend(["--quantize", self.config.quantization_scheme])
        if self.config.torch_dtype is not None:
            self.command.extend(["--dtype", self.config.torch_dtype])

        if self.hub_kwargs.get("trust_remote_code", False):
            self.command.append("--trust-remote-code")
        if self.config.disable_custom_kernels:
            self.command.append("--disable-custom-kernels")

        if self.device == "cuda":
            device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", self.device.index or 0)
            LOGGER.info(f"\t+ Starting TGI container on CUDA device(s): {device_ids}")
            device_requests = [docker.types.DeviceRequest(device_ids=[str(device_ids)], capabilities=[["gpu"]])]
        else:
            LOGGER.info("\t+ Starting TGI container on CPU device")
            device_requests = None

        if self.config.no_weights:
            self.volumes = {self.tmp_dir.name: {"bind": self.tmp_dir.name, "mode": "rw"}}
        else:
            self.volumes = {self.config.volume: {"bind": "/data", "mode": "rw"}}

        ports = {"80/tcp": (self.config.address, self.config.port)}

        self.tgi_container = self.docker_client.containers.run(
            device_requests=device_requests,
            command=self.command,
            volumes=self.volumes,
            shm_size=self.config.shm_size,
            image=self.config.image,
            environment=env,
            ports=ports,
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

        while True:
            try:
                LOGGER.info("\t+ Checking if TGI client is ready")
                self.client.text_generation(prompt="test", max_new_tokens=1)
                LOGGER.info("\t+ TGI client is ready")
                break
            except Exception as e:
                LOGGER.info(f"\t+ TGI client is not ready yet: {e}")
                time.sleep(0.5)

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
            LOGGER.info("\t+ Stopping TGI container")
            self.tgi_container.stop()
            LOGGER.info("\t+ Waiting for TGI container to stop")
            self.tgi_container.wait()

        if hasattr(self, "docker_client"):
            LOGGER.info("\t+ Closing docker client")
            self.docker_client.close()

        if hasattr(self, "tmp_dir"):
            LOGGER.info("\t+ Cleaning temporary directory")
            self.tmp_dir.cleanup()

        gc.collect()
