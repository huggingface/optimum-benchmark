import gc
import os
import random
import shutil
from abc import ABC
from logging import getLogger
from multiprocessing import Process
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Union,
)

import numpy as np
import torch
from optimum.exporters import TasksManager
from transformers import AutoConfig, AutoProcessor

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        Pipeline,
        PretrainedConfig,
        PreTrainedModel,
        TrainerCallback,
        TrainerState,
    )
    from transformers.utils import ModelOutput

    from .utils import PreTrainedProcessor

from ..task_utils import DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from .config import BackendConfigT
from .utils import (
    check_no_process_is_running_on_cuda_device,
    check_only_this_process_is_running_on_cuda_device,
    extract_shapes_from_diffusion_pipeline,
    extract_shapes_from_model_artifacts,
)

LOGGER = getLogger("backend")


class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    # instance variables withouth default values https://stackoverflow.com/a/44962662
    config: BackendConfigT
    pretrained_model: Union["PreTrainedModel", "Pipeline"]
    pretrained_processor: Optional["PreTrainedProcessor"]
    pretrained_config: Optional["PretrainedConfig"]

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]):
        self.task = task
        self.model = model
        self.hub_kwargs = hub_kwargs
        self.device = torch.device(device)

        if self.is_diffusion_pipeline():
            # for pipelines
            self.pretrained_config = None
            self.pretrained_processor = None
            self.model_type = self.task
        else:
            # for models
            self.pretrained_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model, **self.hub_kwargs
            )
            self.model_type = self.pretrained_config.model_type

            try:
                # the processor sometimes contains information about the model's
                # input shapes that's not available in the config
                self.pretrained_processor = AutoProcessor.from_pretrained(
                    pretrained_model_name_or_path=self.model, **self.hub_kwargs
                )
            except ValueError:
                LOGGER.warning("Could not find the model's preprocessor")
                self.pretrained_processor = None

        self.automodel_class = TasksManager.get_model_class_for_task(
            task=self.task,
            framework="pt",
            model_type=self.model_type,
        )

    def is_text_generation_model(self) -> bool:
        return self.task in TEXT_GENERATION_TASKS

    def is_diffusion_pipeline(self) -> bool:
        return self.task in DIFFUSION_TASKS

    def check_initial_isolation(self) -> None:
        if self.device.type == "cuda":
            # at this point we are sure that CUDA_VISIBLE_DEVICES is set if there are multiple GPUs available on the machine
            CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if CUDA_VISIBLE_DEVICES is None:
                device_ids = [self.device.index if self.device.index is not None else 0]
            else:
                device_ids = list(map(int, CUDA_VISIBLE_DEVICES.split(",")))

            LOGGER.info(f"\t+ Checking initial device(s) isolation of CUDA device(s): {device_ids}")
            check_no_process_is_running_on_cuda_device(device_ids)

    def check_continuous_isolation(self) -> None:
        if self.device.type == "cuda":
            # at this point we are sure that CUDA_VISIBLE_DEVICES is set if there are multiple GPUs available on the machine
            CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if CUDA_VISIBLE_DEVICES is None:
                device_ids = [self.device.index if self.device.index is not None else 0]
            else:
                device_ids = list(map(int, CUDA_VISIBLE_DEVICES.split(",")))

            LOGGER.info(f"\t+ Checking contineous device(s) isolation of CUDA device(s): {device_ids}")
            self.isolation_thread = Process(
                target=check_only_this_process_is_running_on_cuda_device,
                args=(device_ids, os.getpid()),
                daemon=True,
            )
            self.isolation_thread.start()

    def configure(self, config: BackendConfigT) -> None:
        LOGGER.info(f"Configuring {self.NAME} backend")
        self.config = config

        # seeding backend
        self.seed()

        # isolation options
        if self.config.initial_isolation_check:
            self.check_initial_isolation()
        if self.config.continous_isolation_check:
            self.check_continuous_isolation()

        # clean up options
        if self.config.delete_cache:
            LOGGER.info("\t+ Model cache will be deleted after benchmark")

    def seed(self) -> None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)  # safe to call
            # torch.use_deterministic_algorithms()  # might throw an error
            # torch.backends.cudnn.deterministic = True # same as above
            # torch.backends.cudnn.benchmark = False  # might reduce performance

    def prepare_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_diffusion_pipeline():
            # diffusion pipelines expect a list of strings as input
            return input
        else:
            # models expect tensors on the target device as input
            for key, value in input.items():
                input[key] = value.to(self.device)

        return input

    # compiling in openvino requires input shapes
    def prepare_for_inference(self, input_shapes: Dict[str, int]) -> Dict[str, Any]:
        pass

    # symbolic tracing in transformers requires input names
    def prepare_for_profiling(self, input_names: List[str]) -> Dict[str, Any]:
        pass

    def forward(self, input: Dict[str, Any], kwargs) -> "ModelOutput":
        return self.pretrained_model(**input, **kwargs)

    def generate(self, input: Dict[str, Any], kwargs) -> "ModelOutput":
        return self.pretrained_model.generate(**input, **kwargs)

    def train(
        self,
        training_dataset: "Dataset",
        training_arguments: Dict[str, Any],
        training_callbacks: List["TrainerCallback"],
        training_data_collator: Callable,
    ) -> "TrainerState":
        raise NotImplementedError("Backend must implement train method")

    @property
    def model_shapes(self) -> Dict[str, int]:
        if self.is_diffusion_pipeline():
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
        if hasattr(self, "pretrained_model"):
            del self.pretrained_model

        gc.collect()

    def delete_model_cache(self) -> None:
        LOGGER.info("\t+ Deleting model cache")
        model_cache_path = f"models/{self.model}".replace("/", "--")
        model_cache_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), model_cache_path)
        shutil.rmtree(model_cache_path, ignore_errors=True)

    def clean(self) -> None:
        LOGGER.info(f"Cleaning {self.NAME} backend")
        self.delete_pretrained_model()

        if self.config.delete_cache:
            self.delete_model_cache()
