import shutil
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Union

from py_txi import TEI, TGI, TEIConfig, TGIConfig

from ...task_utils import TEXT_EMBEDDING_TASKS, TEXT_GENERATION_TASKS
from ..base import Backend
from .config import PyTXIConfig


class PyTXIBackend(Backend[PyTXIConfig]):
    NAME: str = "py-txi"
    pretrained_model: Union[TEI, TGI]

    def __init__(self, config: PyTXIConfig) -> None:
        super().__init__(config)

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            self.logger.info("\t+ Creating no weights model")
            self.create_no_weights_model_slow()
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

    def load_model_with_no_weights(self) -> None:
        original_volumes, self.config.volumes = self.config.volumes, {self.tmpdir.name: {"bind": "/data", "mode": "rw"}}
        self.load_model_from_pretrained()
        self.config.volumes = original_volumes

    def load_model_from_pretrained(self) -> None:
        if self.config.task in TEXT_GENERATION_TASKS:
            self.pretrained_model = TGI(
                config=TGIConfig(model_id=self.config.model, **self.txi_kwargs, **self.tgi_kwargs),
            )
        elif self.config.task in TEXT_EMBEDDING_TASKS:
            self.pretrained_model = TEI(
                config=TEIConfig(model_id=self.config.model, **self.txi_kwargs, **self.tei_kwargs),
            )
        else:
            raise NotImplementedError(f"TXI does not support task {self.config.task}")

    @property
    def txi_kwargs(self):
        kwargs = {}

        if self.config.gpus is not None:
            kwargs["gpus"] = self.config.gpus

        if self.config.image is not None:
            kwargs["image"] = self.config.image

        if self.config.ports is not None:
            kwargs["ports"] = self.config.ports

        if self.config.volumes is not None:
            kwargs["volumes"] = self.config.volumes

        if self.config.devices is not None:
            kwargs["devices"] = self.config.devices

        if self.config.shm_size is not None:
            kwargs["shm_size"] = self.config.shm_size

        if self.config.environment is not None:
            kwargs["environment"] = self.config.environment

        if self.config.connection_timeout is not None:
            kwargs["connection_timeout"] = self.config.connection_timeout

        if self.config.first_request_timeout is not None:
            kwargs["first_request_timeout"] = self.config.first_request_timeout

        if self.config.max_concurrent_requests is not None:
            kwargs["max_concurrent_requests"] = self.config.max_concurrent_requests

        return kwargs

    @property
    def tei_kwargs(self):
        kwargs = {}

        if self.config.dtype is not None:
            kwargs["dtype"] = self.config.dtype

        if self.config.pooling is not None:
            kwargs["pooling"] = self.config.pooling

        return kwargs

    @property
    def tgi_kwargs(self):
        kwargs = {}

        if self.config.dtype is not None:
            kwargs["dtype"] = self.config.dtype

        if self.config.sharded is not None:
            kwargs["sharded"] = self.config.sharded

        if self.config.quantize is not None:
            kwargs["quantize"] = self.config.quantize

        if self.config.num_shard is not None:
            kwargs["num_shard"] = self.config.num_shard

        if self.config.speculate is not None:
            kwargs["speculate"] = self.config.speculate

        if self.config.cuda_graphs is not None:
            kwargs["cuda_graphs"] = self.config.cuda_graphs

        if self.config.trust_remote_code is not None:
            kwargs["trust_remote_code"] = self.config.trust_remote_code

        if self.config.disable_custom_kernels is not None:
            kwargs["disable_custom_kernels"] = self.config.disable_custom_kernels

        return kwargs

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.task in TEXT_GENERATION_TASKS:
            inputs = {"prompt": self.pretrained_processor.batch_decode(inputs["input_ids"].tolist())}
        elif self.config.task in TEXT_EMBEDDING_TASKS:
            inputs = {"text": self.pretrained_processor.batch_decode(inputs["input_ids"].tolist())}
        else:
            raise NotImplementedError(f"TXI does not support task {self.config.task}")

        return inputs

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
