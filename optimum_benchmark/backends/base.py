import gc
import os
import random
import shutil
from abc import ABC
from logging import getLogger
from typing import Any, Callable, ClassVar, Dict, Generic, Optional, Union

import numpy as np
from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
    Pipeline,
    PretrainedConfig,
    PreTrainedModel,
    TrainerState,
)
from transformers.utils import ModelOutput

from ..task_utils import (
    DIFFUSION_TASKS,
    TEXT_GENERATION_TASKS,
    get_model_class_for_task,
)
from .config import BackendConfigT
from .utils import (
    PreTrainedProcessor,
    extract_shapes_from_diffusion_pipeline,
    extract_shapes_from_model_artifacts,
)

LOGGER = getLogger("backend")


class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    library: str
    model_type: str
    config: BackendConfigT
    pretrained_model: Union[PreTrainedModel, Pipeline]
    pretrained_config: Optional[PretrainedConfig]
    pretrained_processor: Optional[PreTrainedProcessor]
    pretrained_generation_config: Optional[GenerationConfig]
    automodel_class: Callable[..., PreTrainedModel]

    def __init__(self, model: str, task: str, library: str, device: str, hub_kwargs: Dict[str, Any]):
        self.task = task
        self.model = model
        self.device = device
        self.library = library
        self.hub_kwargs = hub_kwargs

        if self.library == "diffusers":
            self.model_type = self.task
            self.pretrained_config = None
            self.pretrained_processor = None
        elif self.library == "timm":
            from .timm_utils import get_pretrained_config

            self.pretrained_config = get_pretrained_config(self.model)
            self.model_type = self.pretrained_config.architecture
            self.pretrained_processor = None
        else:
            self.pretrained_config = AutoConfig.from_pretrained(self.model, **self.hub_kwargs)
            self.model_type = self.pretrained_config.model_type

            try:
                # sometimes contains information about the model's input shapes that are not available in the config
                self.pretrained_processor = AutoProcessor.from_pretrained(self.model, **self.hub_kwargs)
            except ValueError:
                # sometimes the processor is not available or can't be determined/detected
                LOGGER.warning("Could not find the model's preprocessor")
                self.pretrained_processor = None

        try:
            self.pretrained_generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name=self.model, **self.hub_kwargs
            )
        except Exception:
            self.pretrained_generation_config = None

        self.automodel_class = get_model_class_for_task(
            model_type=self.model_type,
            library=self.library,
            task=self.task,
            framework="pt",
        )

    def is_text_generation_model(self) -> bool:
        return self.task in TEXT_GENERATION_TASKS

    def is_diffusion_pipeline(self) -> bool:
        return self.task in DIFFUSION_TASKS

    def configure(self, config: BackendConfigT) -> None:
        LOGGER.info(f"Configuring {self.NAME} backend")
        self.config = config

        # clean up options
        if self.config.delete_cache:
            LOGGER.info("\t+ Model cache will be deleted after benchmark")

    def seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def prepare_for_inference(self, **kwargs) -> None:
        pass

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model(**input, **kwargs)

    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model.generate(**input, **kwargs)

    def train(self, **kwargs) -> TrainerState:
        raise NotImplementedError("Backend must implement train method")

    @property
    def model_shapes(self) -> Dict[str, int]:
        if self.library == "diffusers":
            model_shapes = extract_shapes_from_diffusion_pipeline(
                pipeline=self.pretrained_model,
            )
        else:
            model_shapes = extract_shapes_from_model_artifacts(
                config=self.pretrained_config,
                processor=self.pretrained_processor,
            )

        return model_shapes

    def delete_pretrained_model(self) -> None:
        LOGGER.info("\t+ Deleting pretrained model")
        del self.pretrained_model
        gc.collect()

    def delete_hf_model_cache(self) -> None:
        LOGGER.info("\t+ Deleting model cache")
        model_cache_folder = f"models/{self.model}".replace("/", "--")
        model_cache_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), model_cache_folder)
        shutil.rmtree(model_cache_path, ignore_errors=True)

    def clean(self) -> None:
        LOGGER.info(f"Cleaning {self.NAME} backend")

        if hasattr(self, "pretrained_model"):
            self.delete_pretrained_model()

        if self.config.delete_cache:
            self.delete_hf_model_cache()

        gc.collect()
