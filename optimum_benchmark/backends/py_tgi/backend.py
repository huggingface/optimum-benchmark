import gc
import os
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import torch
from py_tgi import TGI
from safetensors.torch import save_file
from transformers import GenerationConfig

from ...task_utils import TEXT_GENERATION_TASKS
from ..base import Backend
from ..transformers_utils import random_init_weights
from .config import PyTGIConfig

# bachend logger
LOGGER = getLogger("text-generation-inference")


class PyTGIBackend(Backend[PyTGIConfig]):
    NAME: str = "py-tgi"

    def __init__(self, config: PyTGIConfig) -> None:
        super().__init__(config)
        self.validate_task()

        if self.generation_config is None:
            self.generation_config = GenerationConfig()

        LOGGER.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            LOGGER.info("\t+ Loading no weights model")
            self.load_model_with_no_weights()
        else:
            LOGGER.info("\t+ Downloading pretrained model")
            self.download_pretrained_model()
            LOGGER.info("\t+ Preparing generation config")
            self.prepare_generation_config()
            LOGGER.info("\t+ Loading pretrained model")
            self.load_model_from_pretrained()

        self.tmpdir.cleanup()

    def validate_task(self) -> None:
        if self.config.task not in TEXT_GENERATION_TASKS:
            raise NotImplementedError(f"TGI does not support task {self.config.task}")

    def download_pretrained_model(self) -> None:
        LOGGER.info("\t+ Downloading pretrained model")
        with torch.device("meta"):
            self.automodel_class.from_pretrained(self.config.model, **self.config.hub_kwargs)

    def prepare_generation_config(self) -> None:
        LOGGER.info("\t+ Modifying generation config for fixed length generation")
        self.generation_config.eos_token_id = None
        self.generation_config.pad_token_id = None
        model_cache_folder = f"models/{self.config.model}".replace("/", "--")
        model_cache_path = f"{self.config.volume}/{model_cache_folder}"
        snapshot_file = f"{model_cache_path}/refs/{self.config.hub_kwargs.get('revision', 'main')}"
        snapshot_ref = open(snapshot_file, "r").read().strip()
        model_snapshot_path = f"{model_cache_path}/snapshots/{snapshot_ref}"
        LOGGER.info("\t+ Saving new pretrained generation config")
        self.generation_config.save_pretrained(save_directory=model_snapshot_path)

    def create_no_weights_model(self) -> None:
        self.no_weights_model = os.path.join(self.tmp_dir.name, "no_weights_model")
        LOGGER.info("\t+ Creating no weights model directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        LOGGER.info("\t+ Creating no weights model state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()
        LOGGER.info("\t+ Saving no weights model safetensors")
        safetensor = os.path.join(self.no_weights_model, "model.safetensors")
        save_file(tensors=state_dict, filename=safetensor, metadata={"format": "pt"})
        # unlike Transformers api, TGI won't accept any missing tensors
        # so we need to materialize the model and resave it
        LOGGER.info(f"\t+ Loading no weights model from {self.no_weights_model}")
        with random_init_weights():
            self.pretrained_model = self.automodel_class.from_pretrained(
                self.no_weights_model, **self.config.hub_kwargs, device_map="auto", _fast_init=False
            )
        LOGGER.info("\t+ Saving no weights model")
        self.pretrained_model.save_pretrained(save_directory=self.no_weights_model)
        LOGGER.info("\t+ Saving no weights model pretrained config")
        self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)
        LOGGER.info("\t+ Saving no weights model pretrained processor")
        self.pretrained_processor.save_pretrained(save_directory=self.no_weights_model)
        LOGGER.info("\t+ Modifying generation config for fixed length generation")
        self.generation_config.eos_token_id = None
        self.generation_config.pad_token_id = None
        LOGGER.info("\t+ Saving new pretrained generation config")
        self.generation_config.save_pretrained(save_directory=self.no_weights_model)

    def load_model_with_no_weights(self) -> None:
        LOGGER.info("\t+ Creating no weights model")
        self.create_no_weights_model()

        original_volume, self.config.volume = self.config.volume, self.tmp_dir.name
        original_model, self.config.model = self.config.model, "/data/no_weights_model"
        LOGGER.info("\t+ Loading no weights model")
        self.load_model_from_pretrained()
        self.config.model, self.config.volume = original_model, original_volume

    def load_model_from_pretrained(self) -> None:
        self.pretrained_model = TGI(
            # model
            model=self.config.model,
            dtype=self.config.dtype,
            quantize=self.config.quantize,
            # docker
            image=self.config.image,
            shm_size=self.config.shm_size,
            address=self.config.address,
            volume=self.config.volume,
            port=self.config.port,
            # device
            gpus=self.config.gpus,
            devices=self.config.devices,
            # sharding
            sharded=self.config.sharded,
            num_shard=self.config.num_shard,
            # other
            disable_custom_kernels=self.config.disable_custom_kernels,
            trust_remote_code=self.config.hub_kwargs.get("trust_remote_code", False),
            revision=self.config.hub_kwargs.get("revision", "main"),
        )

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "inputs" in inputs:
            return {"prompt": self.pretrained_processor.batch_decode(inputs["inputs"].tolist())}
        elif "input_ids" in inputs:
            return {"prompt": self.pretrained_processor.batch_decode(inputs["input_ids"].tolist())}
        else:
            raise ValueError("inputs must contain either input_ids or inputs")

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return self.pretrained_model.generate(**inputs, **kwargs, max_new_tokens=1)

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return self.pretrained_model.generate(
            **inputs, do_sample=kwargs.get("do_sample", False), max_new_tokens=kwargs.get("max_new_tokens", 1)
        )

    def clean(self) -> None:
        super().clean()

        if hasattr(self, "tmpdir"):
            LOGGER.info("\t+ Cleaning temporary directory")
            self.tmpdir.cleanup()

        gc.collect()
