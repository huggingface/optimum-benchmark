import gc
import os
import random
import shutil
from abc import ABC
from logging import getLogger
from typing import (
    Optional,
    ClassVar,
    Generic,
    Dict,
    Any,
)

import numpy as np
from transformers.utils import ModelOutput
from transformers import (
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    TrainerState,
    AutoModel,
)

from .config import BackendConfigT, BackendConfig
from ..task_utils import get_automodel_class_for_task
from .diffusers_utils import (
    extract_diffusers_shapes_from_config,
    get_diffusers_pretrained_config,
)
from .transformers_utils import (
    extract_transformers_shapes_from_artifacts,
    get_transformers_pretrained_processor,
    get_transformers_generation_config,
    get_transformers_pretrained_config,
    get_transformers_cache_dir,
    PretrainedProcessor,
)
from .timm_utils import (
    extract_timm_shapes_from_config,
    get_timm_pretrained_processor,
    get_timm_pretrained_config,
)

LOGGER = getLogger("backend")


class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    config: BackendConfigT
    automodel_class: AutoModel
    pretrained_model: PreTrainedModel
    model_shapes: Dict[str, int]

    pretrained_config: Optional[PretrainedConfig]
    pretrained_processor: Optional[PretrainedProcessor]
    pretrained_generation_config: Optional[GenerationConfig]

    def __init__(self, config: BackendConfigT):
        LOGGER.info(f"َAllocating {self.NAME} backend")
        self.config = config

        if self.config.library == "diffusers":
            self.pretrained_processor = None
            self.pretrained_generation_config = None
            self.pretrained_config = get_diffusers_pretrained_config(
                model=self.config.model, **self.config.hub_kwargs
            )
            self.model_shapes = extract_diffusers_shapes_from_config(
                model=self.config.model, **self.config.hub_kwargs
            )
            self.model_type = self.config.task
        elif self.config.library == "timm":
            self.pretrained_processor = get_timm_pretrained_processor(self.config.model)
            self.pretrained_config = get_timm_pretrained_config(self.config.model)
            self.model_shapes = extract_timm_shapes_from_config(
                config=self.pretrained_config
            )
            self.model_type = self.pretrained_config.architecture
            self.pretrained_generation_config = None
        else:
            self.pretrained_config = get_transformers_pretrained_config(
                self.config.model, **self.config.hub_kwargs
            )
            self.pretrained_generation_config = get_transformers_generation_config(
                self.config.model, **self.config.hub_kwargs
            )
            self.pretrained_processor = get_transformers_pretrained_processor(
                self.config.model, **self.config.hub_kwargs
            )
            self.model_shapes = extract_transformers_shapes_from_artifacts(
                config=self.pretrained_config,
                processor=self.pretrained_processor,
            )
            self.model_type = self.pretrained_config.model_type

        self.automodel_class = get_automodel_class_for_task(
            model_type=self.model_type,
            library=self.config.library,
            task=self.config.task,
            framework="pt",
        )

    def seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def prepare_for_inference(self, **kwargs) -> None:
        """
        This method is used to prepare the model for inference.
        It can be used to compile the model with certain input/output shapes, for example.
        """
        pass

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method is used to prepare the inputs before passing them to the model.
        It can be used to move the inputs to the correct device, for example.
        """
        return inputs

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        """
        This method is used to perform the forward pass of the model.
        """
        raise NotImplementedError("Backend must implement forward method")

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        """
        This method is used to perform the generation pass of the model.
        """
        raise NotImplementedError("Backend must implement generate method")

    def train(self, **kwargs) -> TrainerState:
        """
        This method is used to train the model.
        """
        raise NotImplementedError("Backend must implement train method")

    def delete_hf_model_cache(self) -> None:
        LOGGER.info("\t+ Deleting model cache")
        transformers_cache_path = get_transformers_cache_dir()
        model_cache_folder = f"models/{self.config.model}".replace("/", "--")
        model_cache_path = os.path.join(transformers_cache_path, model_cache_folder)
        shutil.rmtree(model_cache_path, ignore_errors=True)

    def delete_pretrained_model(self) -> None:
        LOGGER.info("\t+ Deleting pretrained model")
        del self.pretrained_model
        gc.collect()

    def clean(self) -> None:
        LOGGER.info(f"Cleaning {self.NAME} backend")

        if hasattr(self, "pretrained_model"):
            self.delete_pretrained_model()

        gc.collect()
