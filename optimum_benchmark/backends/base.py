import os
from abc import ABC
from collections import OrderedDict
from logging import getLogger
from typing import Any, ClassVar, Dict, Generic, Optional

import datasets.utils.logging as datasets_logging
import transformers.utils.logging as transformers_logging
from safetensors.torch import save_file
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, TrainerState, set_seed

from ..import_utils import is_torch_available
from .config import BackendConfigT
from .diffusers_utils import (
    extract_diffusers_shapes_from_model,
    get_diffusers_automodel_loader_for_task,
    get_diffusers_pretrained_config,
)
from .timm_utils import extract_timm_shapes_from_config, get_timm_automodel_loader, get_timm_pretrained_config
from .transformers_utils import (
    PretrainedProcessor,
    extract_transformers_shapes_from_artifacts,
    get_transformers_automodel_loader_for_task,
    get_transformers_generation_config,
    get_transformers_pretrained_config,
    get_transformers_pretrained_processor,
)

if is_torch_available():
    import torch

datasets_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    model_type: str
    model_shapes: Dict[str, int]

    pretrained_model: PreTrainedModel
    pretrained_config: Optional[PretrainedConfig]
    generation_config: Optional[GenerationConfig]
    pretrained_processor: Optional[PretrainedProcessor]

    def __init__(self, config: BackendConfigT):
        self.config = config

        self.logger = getLogger(self.NAME)
        self.logger.info(f"Allocating {self.NAME} backend")

        self.logger.info(f"\t+ Seeding backend with {self.config.seed}")
        self.seed()

        if self.config.library == "diffusers":
            self.logger.info("\t+ Benchmarking a Diffusers pipeline")
            self.pretrained_config = get_diffusers_pretrained_config(self.config.model, **self.config.model_kwargs)
            self.model_shapes = extract_diffusers_shapes_from_model(self.config.model, **self.config.model_kwargs)
            self.automodel_loader = get_diffusers_automodel_loader_for_task(self.config.task)
            self.pretrained_processor = None
            self.generation_config = None

        elif self.config.library == "timm":
            self.logger.info("\t+ Benchmarking a Timm model")
            self.pretrained_config = get_timm_pretrained_config(self.config.model)
            self.model_shapes = extract_timm_shapes_from_config(self.pretrained_config)
            self.automodel_loader = get_timm_automodel_loader()
            self.pretrained_processor = None
            self.generation_config = None

        elif self.config.library == "llama_cpp":
            self.logger.info("\t+ Benchmarking a LlamaCpp model")
            self.pretrained_processor = None
            self.generation_config = None
            self.pretrained_config = None
            self.automodel_loader = None
            # TOD: need a custom method to extract shapes from gguf
            self.model_shapes = extract_transformers_shapes_from_artifacts(
                self.pretrained_config, self.pretrained_processor
            )

        else:
            self.logger.info("\t+ Benchmarking a Transformers model")
            self.generation_config = get_transformers_generation_config(self.config.model, **self.config.model_kwargs)
            self.pretrained_config = get_transformers_pretrained_config(self.config.model, **self.config.model_kwargs)
            self.automodel_loader = get_transformers_automodel_loader_for_task(self.config.task)
            self.pretrained_processor = get_transformers_pretrained_processor(
                self.config.processor, **self.config.processor_kwargs
            )
            self.model_shapes = extract_transformers_shapes_from_artifacts(
                self.pretrained_config, self.pretrained_processor
            )

    def seed(self) -> None:
        set_seed(self.config.seed)

    def create_no_weights_model(self) -> None:
        if self.pretrained_config is None:
            raise ValueError("Can't create no weights model without a pretrained config")

        self.no_weights_model = os.path.join(self.tmpdir.name, "no_weights_model")
        self.logger.info("\t+ Creating no weights model's directory")
        os.makedirs(self.no_weights_model, exist_ok=True)
        self.logger.info("\t+ Creating no weights model's state dict")
        state_dict = torch.nn.Linear(1, 1).state_dict()
        self.logger.info("\t+ Saving no weights model's safetensors")
        safetensors = os.path.join(self.no_weights_model, "model.safetensors")
        save_file(tensors=state_dict, filename=safetensors, metadata={"format": "pt"})
        self.logger.info("\t+ Saving no weights model's config")
        self.pretrained_config.save_pretrained(save_directory=self.no_weights_model)

    def prepare_input_shapes(self, input_shapes: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method is used to prepare and register the input shapes before using them by the model.
        It can be used to pad the inputs to the correct shape, or compile it to the correct format.
        """
        return input_shapes

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method is used to prepare and register the inputs before passing them to the model.
        It can be used to move the inputs to the correct device, or rename their keys.
        """
        return inputs

    def load(self) -> None:
        raise NotImplementedError("Backend must implement load method")

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        """
        This method is used to perform the forward pass of the model.
        """
        raise NotImplementedError("Backend must implement forward method")

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        """
        This method is used to perform the prefill pass of the model.
        """
        raise NotImplementedError("Backend must implement prefill method")

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        """
        This method is used to perform the generation pass of the model.
        """
        raise NotImplementedError("Backend must implement generate method")

    def call(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> OrderedDict:
        """
        This method is used to call a whole pipeline.
        """
        raise NotImplementedError("Backend must implement call method")

    def train(self, **kwargs) -> TrainerState:
        """
        This method is used to train the model.
        """
        raise NotImplementedError("Backend must implement train method")
