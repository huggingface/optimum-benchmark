from abc import ABC
from collections import OrderedDict
from logging import getLogger
from typing import Any, ClassVar, Dict, Generic, Optional, Tuple

import datasets.utils.logging as datasets_logging
import transformers.utils.logging as transformers_logging
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, TrainerState, set_seed

from ..task_utils import get_automodel_class_for_task
from .config import BackendConfigT
from .diffusers_utils import extract_diffusers_shapes_from_model, get_diffusers_pretrained_config
from .timm_utils import extract_timm_shapes_from_config, get_timm_pretrained_config
from .transformers_utils import (
    PretrainedProcessor,
    extract_transformers_shapes_from_artifacts,
    get_transformers_generation_config,
    get_transformers_pretrained_config,
    get_transformers_pretrained_processor,
)

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
            self.model_type = self.config.task
            self.pretrained_processor = None
            self.generation_config = None

        elif self.config.library == "timm":
            self.logger.info("\t+ Benchmarking a Timm model")
            self.pretrained_config = get_timm_pretrained_config(self.config.model)
            self.model_shapes = extract_timm_shapes_from_config(self.pretrained_config)
            self.model_type = self.pretrained_config.architecture
            self.pretrained_processor = None
            self.generation_config = None

        else:
            self.logger.info("\t+ Benchmarking a Transformers model")
            self.generation_config = get_transformers_generation_config(self.config.model, **self.config.model_kwargs)
            self.pretrained_config = get_transformers_pretrained_config(self.config.model, **self.config.model_kwargs)
            self.pretrained_processor = get_transformers_pretrained_processor(
                self.config.processor, **self.config.processor_kwargs
            )
            self.model_shapes = extract_transformers_shapes_from_artifacts(
                self.pretrained_config, self.pretrained_processor
            )
            self.model_type = self.pretrained_config.model_type

        self.automodel_class = get_automodel_class_for_task(
            model_type=self.model_type, library=self.config.library, task=self.config.task, framework="pt"
        )
        self.logger.info(f"\t+ Using automodel class {self.automodel_class.__name__}")

    def seed(self) -> None:
        set_seed(self.config.seed)

    def prepare_for_inference(self, **kwargs) -> None:
        """
        This method is used to prepare the model for inference.
        It can be used to compile the model with certain input/output shapes, for example.
        """
        pass

    def prepare_inputs(
        self, inputs: Dict[str, Any], input_shapes: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        This method is used to prepare the inputs before passing them to the model.
        It can be used to move the inputs to the correct device, for example.
        """
        return inputs, input_shapes

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
