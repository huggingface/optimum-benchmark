import gc
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import torch
from huggingface_hub import snapshot_download
from py_tgi import TGI
from safetensors.torch import save_model
from transformers import logging as transformers_logging

from ...system_utils import is_nvidia_system, is_rocm_system
from ...task_utils import TEXT_GENERATION_TASKS
from ..base import Backend
from ..transformers_utils import randomize_weights
from .config import TGIConfig

# bachend logger
LOGGER = getLogger("text-generation-inference")

# disable other loggers
transformers_logging.set_verbosity_error()


class TGIBackend(Backend[TGIConfig]):
    NAME: str = "text-generation-inference"

    def __init__(self, config: TGIConfig) -> None:
        super().__init__(config)
        self.validate_task()

        if self.config.device == "cuda" and is_nvidia_system():
            self.devices = None
            self.gpus = self.config.device_ids
            LOGGER.info(f"\t+ CUDA devices: {self.gpus}")
        if self.config.device == "cuda" and is_rocm_system():
            self.gpus = None
            device_ids = list(map(int, self.config.device_ids.split(",")))
            renderDs = [file for file in os.listdir("/dev/dri") if file.startswith("renderD")]
            self.devices = ["/dev/kfd"] + [f"/dev/dri/{renderDs[i]}" for i in device_ids]
            LOGGER.info(f"\t+ ROCm devices: {self.devices}")
        else:
            self.gpus = None
            self.devices = None
            LOGGER.info("\t+ CPU device")

        LOGGER.info("\t+ Creating backend temporary directory")
        self.tmp_dir = TemporaryDirectory()

        if self.config.no_weights:
            self.load_model_with_no_weights()
        else:
            self.download_pretrained_model()
            self.load_model_from_pretrained()

    def validate_task(self) -> None:
        if self.config.task not in TEXT_GENERATION_TASKS:
            raise NotImplementedError(f"TGI does not support task {self.config.task}")

    def download_pretrained_model(self) -> None:
        LOGGER.info("\t+ Downloading pretrained model")
        snapshot_download(self.config.model, **self.config.hub_kwargs)

    def prepare_pretrained_model(self) -> None:
        LOGGER.info("\t+ Modifying pretrained generation config")
        self.generation_config.eos_token_id = -100
        self.generation_config.pad_token_id = -101

        LOGGER.info("\t+ Saving new pretrained generation config")
        model_cache_folder = f"models/{self.config.model}".replace("/", "--")
        model_cache_path = f"{self.config.volume}/{model_cache_folder}"

        snapshot_file = f"{model_cache_path}/refs/{self.config.hub_kwargs.get('revision', 'main')}"
        snapshot_ref = open(snapshot_file, "r").read().strip()

        model_snapshot_path = f"{model_cache_path}/snapshots/{snapshot_ref}"
        self.generation_config.save_pretrained(save_directory=model_snapshot_path)

    def load_model_from_pretrained(self) -> None:
        self.prepare_pretrained_model()
        self.start_tgi_server()

    def create_no_weights_model(self) -> None:
        LOGGER.info("\t+ Creating no weights model directory")
        self.no_weights_model = os.path.join(self.config.volume, "no_weights_model")
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
            device_map="auto",  # for faster/safer loading
        )

        LOGGER.info("\t+ Randomizing weights")
        randomize_weights(self.pretrained_model)

        LOGGER.info("\t+ Saving no weights model")
        self.pretrained_model.save_pretrained(save_directory=self.no_weights_model)
        self.delete_pretrained_model()

        LOGGER.info("\t+ Saving generation config")
        self.generation_config.eos_token_id = -100
        self.generation_config.pad_token_id = -101
        self.generation_config.save_pretrained(save_directory=self.no_weights_model)

    def load_model_with_no_weights(self) -> None:
        self.create_no_weights_model()
        original_model = self.config.model
        self.config.model = "data/no_weights_model"
        self.start_tgi_server()
        self.config.model = original_model

    def start_tgi_server(self) -> None:
        self.pretrained_model = TGI(
            model=self.config.model,
            dtype=self.config.dtype,
            image=self.config.image,
            quantize=self.config.quantize,
            port=self.config.port,
            volume=self.config.volume,
            address=self.config.address,
            shm_size=self.config.shm_size,
            gpus=self.gpus,
            devices=self.devices,
            sharded=self.config.sharded,
            num_shard=self.config.num_shard,
            disable_custom_kernels=self.config.disable_custom_kernels,
            revision=self.config.hub_kwargs.get("revision", "main"),
            trust_remote_code=self.config.hub_kwargs.get("trust_remote_code", False),
        )

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "input_ids" in inputs:
            return {"prompt": self.pretrained_processor.batch_decode(inputs["input_ids"].tolist())}
        elif "inputs" in inputs:
            return {"prompt": self.pretrained_processor.batch_decode(inputs["inputs"].tolist())}
        else:
            raise ValueError("inputs must contain either input_ids or inputs")

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return self.pretrained_model.generate(**inputs, **kwargs, max_new_tokens=1)

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return self.pretrained_model.generate(
            **inputs,
            do_sample=kwargs.get("do_sample", False),
            max_new_tokens=kwargs.get("max_new_tokens", 1),
        )

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmp_dir"):
            LOGGER.info("\t+ Cleaning temporary directory")
            self.tmp_dir.cleanup()

        gc.collect()
