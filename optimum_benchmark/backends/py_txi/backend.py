import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

import torch
from py_txi import TEI, TGI, TEIConfig, TGIConfig
from safetensors.torch import save_file

from ...task_utils import TEXT_EMBEDDING_TASKS, TEXT_GENERATION_TASKS
from ..base import Backend
from ..transformers_utils import random_init_weights
from .config import PyTXIConfig


class PyTXIBackend(Backend[PyTXIConfig]):
    NAME: str = "py-txi"

    def __init__(self, config: PyTXIConfig) -> None:
        super().__init__(config)

        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
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

        self.tmpdir.cleanup()

    @property
    def volume(self) -> str:
        return list(self.config.volumes.keys())[0]

    def download_pretrained_model(self) -> None:
        # directly downloads pretrained model in volume (/data) to change generation config before loading model
        with torch.device("meta"):
            self.automodel_class.from_pretrained(self.config.model, **self.config.model_kwargs, cache_dir=self.volume)

    def prepare_generation_config(self) -> None:
        self.generation_config.eos_token_id = None
        self.generation_config.pad_token_id = None

        model_cache_folder = f"models/{self.config.model}".replace("/", "--")
        model_cache_path = f"{self.volume}/{model_cache_folder}"
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
        # unlike Transformers, TXI won't accept any missing tensors so we need to materialize the model
        self.logger.info(f"\t+ Loading no weights model from {self.no_weights_model}")
        with random_init_weights():
            self.pretrained_model = self.automodel_class.from_pretrained(
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
        self.logger.info("\t+ Creating no weights model")
        self.create_no_weights_model()

        original_volumes, self.config.volumes = self.config.volumes, {self.tmpdir.name: {"bind": "/data", "mode": "rw"}}
        original_model, self.config.model = self.config.model, "/data/no_weights_model"
        self.logger.info("\t+ Loading no weights model")
        self.load_model_from_pretrained()
        self.config.model, self.config.volumes = original_model, original_volumes

    def load_model_from_pretrained(self) -> None:
        if self.config.task in TEXT_GENERATION_TASKS:
            self.pretrained_model = TGI(
                config=TGIConfig(
                    model_id=self.config.model,
                    gpus=self.config.gpus,
                    devices=self.config.devices,
                    volumes=self.config.volumes,
                    environment=self.config.environment,
                    ports=self.config.ports,
                    dtype=self.config.dtype,
                    sharded=self.config.sharded,
                    quantize=self.config.quantize,
                    num_shard=self.config.num_shard,
                    speculate=self.config.speculate,
                    cuda_graphs=self.config.cuda_graphs,
                    disable_custom_kernels=self.config.disable_custom_kernels,
                    trust_remote_code=self.config.trust_remote_code,
                    max_concurrent_requests=self.config.max_concurrent_requests,
                ),
            )

        elif self.config.task in TEXT_EMBEDDING_TASKS:
            self.pretrained_model = TEI(
                config=TEIConfig(
                    model_id=self.config.model,
                    gpus=self.config.gpus,
                    devices=self.config.devices,
                    volumes=self.config.volumes,
                    environment=self.config.environment,
                    ports=self.config.ports,
                    dtype=self.config.dtype,
                    pooling=self.config.pooling,
                    max_concurrent_requests=self.config.max_concurrent_requests,
                ),
            )
        else:
            raise NotImplementedError(f"TXI does not support task {self.config.task}")

    def prepare_inputs(
        self, inputs: Dict[str, Any], input_shapes: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, input_shapes = super().prepare_inputs(inputs, input_shapes)

        if self.config.task in TEXT_GENERATION_TASKS:
            inputs = {"prompt": self.pretrained_processor.batch_decode(inputs["input_ids"].tolist())}
        elif self.config.task in TEXT_EMBEDDING_TASKS:
            inputs = {"text": self.pretrained_processor.batch_decode(inputs["input_ids"].tolist())}
        else:
            raise NotImplementedError(f"TXI does not support task {self.config.task}")

        return inputs, input_shapes

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return self.pretrained_model.encode(**inputs, **kwargs)

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return self.pretrained_model.generate(
            **inputs,
            do_sample=kwargs.get("do_sample", False),
            max_new_tokens=kwargs.get("max_new_tokens"),
        )

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> List[str]:
        return self.pretrained_model.generate(
            **inputs,
            do_sample=kwargs.get("do_sample", False),
            max_new_tokens=kwargs.get("max_new_tokens"),
        )
