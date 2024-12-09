from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Any, Dict

import torch
from hydra.utils import get_class

from ...import_utils import is_accelerate_available, is_torch_distributed_available
from ..base import Backend
from ..transformers_utils import fast_weights_init
from .config import IPEXConfig
from .utils import TASKS_TO_IPEXMODEL

if is_accelerate_available():
    from accelerate import Accelerator

if is_torch_distributed_available():
    import torch.distributed


class IPEXBackend(Backend[IPEXConfig]):
    NAME: str = "ipex"

    def __init__(self, config: IPEXConfig) -> None:
        super().__init__(config)

        if self.config.task in TASKS_TO_IPEXMODEL:
            self.ipexmodel_class = get_class(TASKS_TO_IPEXMODEL[self.config.task])
            self.logger.info(f"\t+ Using IPEXModel class {self.ipexmodel_class.__name__}")
        else:
            raise NotImplementedError(f"IPEXBackend does not support task {self.config.task}")

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()

        if self.config.no_weights:
            self.logger.info("\t+ Creating no weights IPEXModel")
            self.create_no_weights_model()
            self.logger.info("\t+ Loading no weights IPEXModel")
            self._load_ipexmodel_with_no_weights()
        else:
            self.logger.info("\t+ Loading pretrained IPEXModel")
            self._load_ipexmodel_from_pretrained()

        self.tmpdir.cleanup()

    def _load_automodel_from_pretrained(self) -> None:
        self.pretrained_model = self.automodel_loader.from_pretrained(self.config.model, **self.config.model_kwargs)

    def _load_automodel_with_no_weights(self) -> None:
        original_model, self.config.model = self.config.model, self.no_weights_model

        with fast_weights_init():
            self._load_automodel_from_pretrained()

        self.logger.info("\t+ Tying model weights")
        self.pretrained_model.tie_weights()

        self.config.model = original_model

    def _load_ipexmodel_from_pretrained(self) -> None:
        self.pretrained_model = self.ipexmodel_class.from_pretrained(
            self.config.model,
            export=self.config.export,
            **self.config.model_kwargs,
            **self.automodel_kwargs,
        )

    def _load_ipexmodel_with_no_weights(self) -> None:
        with fast_weights_init():
            original_model, self.config.model = self.config.model, self.no_weights_model
            original_export, self.config.export = self.config.export, True
            self.logger.info("\t+ Loading no weights IPEXModel")
            self._load_ipexmodel_from_pretrained()
            self.config.export = original_export
            self.config.model = original_model

    @property
    def automodel_kwargs(self) -> Dict[str, Any]:
        kwargs = {}

        if self.config.torch_dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)

        return kwargs

    @property
    def split_between_processes(self) -> bool:
        return is_torch_distributed_available() and torch.distributed.is_initialized()

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.split_between_processes:
            with Accelerator().split_between_processes(inputs=inputs, apply_padding=False) as process_inputs:
                inputs = process_inputs

        return inputs

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.forward(**inputs, **kwargs)

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(**inputs, **kwargs)

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model.generate(**inputs, **kwargs)

    def call(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        return self.pretrained_model(**inputs, **kwargs)
