from dataclasses import dataclass, MISSING
from typing import Dict, List, Optional
from abc import abstractmethod, ABC
from logging import getLogger
import inspect

import torch
from torch import Tensor
from psutil import cpu_count
from omegaconf import DictConfig
from optimum.exporters import TasksManager
from transformers.onnx.utils import get_preprocessor
from transformers import AutoConfig, PreTrainedModel  # type: ignore

from src.utils import LLM_MODEL_TYPES

LOGGER = getLogger("backend")


@dataclass
class BackendConfig(ABC):
    name: str = MISSING  # type: ignore
    version: str = MISSING  # type: ignore
    _target_: str = MISSING  # type: ignore

    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None


class Backend(ABC):
    pretrained_model: PreTrainedModel

    def __init__(self, model: str, device: str, cache_kwargs: DictConfig):
        self.model = model
        self.cache_kwargs = cache_kwargs
        self.device = torch.device(device)

        # get pretrained config
        try:
            self.pretrained_config = AutoConfig.from_pretrained(
                self.model, **self.cache_kwargs
            )
        except OSError:
            LOGGER.error(
                f"Model {self.model} is not a transformers model. Will try to load it with diffusers"
            )

        self.task = TasksManager.infer_task_from_model(
            model=model,
            subfolder=cache_kwargs.subfolder,
            revision=cache_kwargs.revision
        )

    @abstractmethod
    def configure(self, config: BackendConfig) -> None:
        LOGGER.info(f"Configuring {config.name} backend")
        if config.inter_op_num_threads is not None:
            if config.inter_op_num_threads == -1:
                config.inter_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.inter_op_num_threads to cpu_count({config.inter_op_num_threads})"
                )

        if config.intra_op_num_threads is not None:
            if config.intra_op_num_threads == -1:
                config.intra_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.intra_op_num_threads to cpu_count({config.intra_op_num_threads})"
                )

    @abstractmethod
    def forward(self, input: Dict[str, Tensor]):
        raise NotImplementedError("Backend must implement forward method")

    @abstractmethod
    def generate(self, input: Dict[str, Tensor], **kwargs) -> str:
        raise NotImplementedError("Backend must implement generate method")

    @abstractmethod
    def prepare_for_profiling(self, input_names: List[str]) -> None:
        raise NotImplementedError(
            "Backend must implement prepare_for_profiling method")

    @abstractmethod
    def clean(self) -> None:
        raise NotImplementedError("Backend must implement clean method")

    def can_generate(self) -> bool:
        return hasattr(self.pretrained_model, "can_generate") and self.pretrained_model.can_generate()

    def generate_dummy_inputs(self, mode, **input_shapes) -> Dict[str, Tensor]:
        assert mode in ["forward", "generate"], f"mode {mode} not supported"
        assert "batch_size" in input_shapes, "batch_size must be provided in input_shapes"

        # patch for some LLM model types not recognized by TasksManager
        if self.pretrained_config.model_type in TasksManager._SUPPORTED_MODEL_TYPE:
            model_type = self.pretrained_config.model_type
        elif self.pretrained_config.model_type in LLM_MODEL_TYPES:
            model_type = "gpt2"
        else:
            raise ValueError(
                f"Unknown model type {self.pretrained_config.model_type}")

        onnx_config = TasksManager._SUPPORTED_MODEL_TYPE[model_type]["onnx"][self.task](
            self.pretrained_config
        )

        normalized_config = onnx_config.NORMALIZED_CONFIG_CLASS(
            self.pretrained_config)
        generator_classes = onnx_config.DUMMY_INPUT_GENERATOR_CLASSES

        if mode == "forward":
            input_names = list(onnx_config.inputs.keys())
        elif mode == "generate":
            if model_type in LLM_MODEL_TYPES:
                input_names = ["input_ids"]
            else:
                # sometimes it fails because of recursive calls
                input_names = get_preprocessor(self.model).model_input_names
        else:
            raise ValueError(f"Unknown mode {mode}")

        LOGGER.info(f"Generating dummy inputs for {input_names}")

        dummy_inputs = dict()
        for input_name in input_names:
            generator = None
            for generator_class in generator_classes:
                supported_generator_params = inspect.signature(
                    generator_class.__init__
                ).parameters.keys()
                supported_generator_inputs = generator_class.SUPPORTED_INPUT_NAMES

                if input_name in supported_generator_inputs:
                    generator = generator_class(
                        task=self.task,
                        normalized_config=normalized_config,
                        **{
                            param: input_shapes[param]
                            for param in supported_generator_params
                            if param in input_shapes
                        },
                    )

            if generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            dummy_inputs[input_name] = generator.generate(
                input_name).to(self.device)

            if input_name == "attention_mask":
                # patch for until sparse attention is supported
                dummy_inputs["attention_mask"] = torch.ones_like(
                    dummy_inputs["attention_mask"]
                )

        return dummy_inputs
