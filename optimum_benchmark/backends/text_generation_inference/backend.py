import gc
import os
import time
from logging import getLogger
from typing import Any, Dict, List
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

import torch
import docker
import docker.types
import docker.errors
from safetensors.torch import save_model
from huggingface_hub import InferenceClient, snapshot_download
from huggingface_hub.inference._text_generation import TextGenerationResponse

from ..base import Backend
from .config import TGIConfig
from ..transformers_utils import randomize_weights

# bachend logger
LOGGER = getLogger("text-generation-inference")


class TGIBackend(Backend[TGIConfig]):
    NAME: str = "text-generation-inference"

    def __init__(self, config: TGIConfig) -> None:
        super().__init__(config)
        self.validate_task()

        LOGGER.info(f"Using AutoModel class {self.automodel_class.__name__}")

        self.tmp_dir = TemporaryDirectory()

        if self.config.no_weights:
            self.load_model_with_no_weights()
        else:
            self.download_pretrained_model()
            self.load_model_from_pretrained()

    def validate_task(self) -> None:
        if self.config.task not in ["text-generation", "text2text-generation"]:
            raise NotImplementedError(f"TGI does not support task {self.config.task}")

    def download_pretrained_model(self) -> None:
        LOGGER.info("\t+ Downloading pretrained model")
        snapshot_download(self.config.model, **self.config.hub_kwargs)

    def load_model_from_pretrained(self) -> None:

        LOGGER.info("\t+ Modifying pretrained generation config")
        self.pretrained_generation_config.eos_token_id = -100
        self.pretrained_generation_config.pad_token_id = -101

        LOGGER.info("\t+ Saving new pretrained generation config")
        model_cache_folder = f"models/{self.config.model}".replace("/", "--")
        model_cache_path = f"{self.config.volume}/{model_cache_folder}"

        snapshot_ref = (
            open(
                f"{model_cache_path}/refs/{self.config.hub_kwargs.get('revision', 'main')}",
                "r",
            )
            .read()
            .strip()
        )

        model_snapshot_path = f"{model_cache_path}/snapshots/{snapshot_ref}"
        self.pretrained_generation_config.save_pretrained(
            save_directory=model_snapshot_path
        )

        self.start_tgi_server()

    def create_no_weights_model(self) -> None:

        LOGGER.info("\t+ Creating no weights model directory")
        self.no_weights_model = os.path.join(self.tmp_dir.name, "no_weights")
        os.makedirs(self.no_weights_model, exist_ok=True)

        LOGGER.info("\t+ Saving pretrained config")
        self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

        LOGGER.info("\t+ Saving pretrained tokenizer")
        self.pretrained_processor.save_pretrained(save_directory=self.no_weights_model)

        LOGGER.info("\t+ Saving no weights model")
        save_model(
            filename=os.path.join(self.no_weights_model, "model.safetensors"),
            model=torch.nn.Linear(1, 1),
            metadata={"format": "pt"},
        )
        # unlike transformers api, TGI won't accept an empty model.safetensors
        # so we need to materialize the model and resave it
        LOGGER.info(f"\t+ Loading no weights model from {self.no_weights_model}")
        self.pretrained_model = self.automodel_class.from_pretrained(
            self.no_weights_model,
            **self.config.hub_kwargs,
            device_map="auto",
        )

        LOGGER.info("\t+ Randomizing weights")
        randomize_weights(self.pretrained_model)

        LOGGER.info("\t+ Saving no weights model")
        self.pretrained_model.save_pretrained(save_directory=self.no_weights_model)
        self.delete_pretrained_model()

        LOGGER.info(f"\t+ Saving generation config")
        self.pretrained_generation_config.eos_token_id = -100
        self.pretrained_generation_config.pad_token_id = -101
        self.pretrained_generation_config.save_pretrained(
            save_directory=self.no_weights_model
        )

    def load_model_with_no_weights(self) -> None:
        self.create_no_weights_model()
        original_model = self.config.model
        self.config.model = self.no_weights_model
        self.start_tgi_server()
        self.config.model = original_model

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
            self.config.model,
            "--revision",
            self.config.hub_kwargs.get("revision", "main"),
        ]

        if self.config.sharded is not None:
            self.command.extend(["--sharded", str(self.config.sharded).lower()])
        if self.config.num_shard is not None:
            self.command.extend(["--num-shard", str(self.config.num_shard)])
        if self.config.quantization_scheme is not None:
            self.command.extend(["--quantize", self.config.quantization_scheme])
        if self.config.torch_dtype is not None:
            self.command.extend(["--dtype", self.config.torch_dtype])

        if self.config.hub_kwargs.get("trust_remote_code", False):
            self.command.append("--trust-remote-code")
        if self.config.disable_custom_kernels:
            self.command.append("--disable-custom-kernels")

        if self.config.device == "cuda":
            device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            LOGGER.info(f"\t+ Starting TGI container on CUDA device(s): {device_ids}")
            device_requests = [
                docker.types.DeviceRequest(
                    device_ids=[device_ids], capabilities=[["gpu"]]
                )
            ]
        else:
            LOGGER.info("\t+ Starting TGI container on CPU device")
            device_requests = None

        if self.config.no_weights:
            self.volumes = {
                self.tmp_dir.name: {"bind": self.tmp_dir.name, "mode": "rw"}
            }
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
        self.client = InferenceClient(
            model=f"http://{self.config.address}:{self.config.port}"
        )

        while True:
            try:
                LOGGER.info("\t+ Checking if TGI client is ready")
                self.client.text_generation(prompt="test", max_new_tokens=1)
                LOGGER.info("\t+ TGI client is ready")
                break
            except Exception as e:
                LOGGER.info(f"\t+ TGI client is not ready yet: {e}")
                time.sleep(0.5)

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "input_ids" in inputs:
            return {
                "prompt": self.pretrained_processor.batch_decode(
                    inputs["input_ids"].tolist()
                )
            }
        elif "inputs" in inputs:
            return {
                "prompt": self.pretrained_processor.batch_decode(
                    inputs["inputs"].tolist()
                )
            }
        else:
            raise ValueError("inputs must contain either input_ids or inputs")

    def forward(
        self, inputs: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> List[TextGenerationResponse]:
        output = []
        with ThreadPoolExecutor(max_workers=len(inputs["prompt"])) as executor:
            futures = [
                executor.submit(
                    self.client.text_generation,
                    decoder_input_details=True,
                    prompt=inputs["prompt"][i],
                    max_new_tokens=1,
                    details=True,
                )
                for i in range(len(inputs["prompt"]))
            ]
        for future in futures:
            output.append(future.result())

        return output

    def generate(
        self, inputs: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> List[TextGenerationResponse]:
        output = []
        with ThreadPoolExecutor(max_workers=len(inputs["prompt"])) as executor:
            futures = [
                executor.submit(
                    self.client.text_generation,
                    max_new_tokens=kwargs["max_new_tokens"],
                    do_sample=kwargs["do_sample"],
                    prompt=inputs["prompt"][i],
                    details=True,
                )
                for i in range(len(inputs["prompt"]))
            ]
        for i in range(len(inputs["prompt"])):
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
