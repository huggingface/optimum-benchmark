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
from .isolation_utils import (
    only_this_process_is_running_on_cuda_devices,
    only_this_process_will_run_on_cuda_devices,
)
from .utils import (
    extract_shapes_from_diffusion_pipeline,
    extract_shapes_from_model_artifacts,
)

LOGGER = getLogger("backend")

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if CUDA_VISIBLE_DEVICES is not None:
    CUDA_DEVICES = list(map(int, CUDA_VISIBLE_DEVICES.split(",")))
elif torch.cuda.is_available():
    CUDA_DEVICES = list(range(torch.cuda.device_count()))
else:
    CUDA_DEVICES = []


class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    # instance variables without default values https://stackoverflow.com/a/44962662
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
            self.library = "diffusers"
            self.model_type = self.task
            self.pretrained_config = None
            self.pretrained_processor = None
        else:
            # for models
            self.library = "transformers"
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
                # sometimes the processor is not available or can't be determined/detected
                LOGGER.warning("Could not find the model's preprocessor")
                self.pretrained_processor = None

        self.automodel_class = TasksManager.get_model_class_for_task(
            task=self.task, library=self.library, model_type=self.model_type
        )

    def is_text_generation_model(self) -> bool:
        return self.task in TEXT_GENERATION_TASKS

    def is_diffusion_pipeline(self) -> bool:
        return self.task in DIFFUSION_TASKS

    def configure(self, config: BackendConfigT) -> None:
        LOGGER.info(f"Configuring {self.NAME} backend")
        self.config = config

        # isolation options
        if self.config.initial_isolation_check:
            self.check_initial_isolation()
        if self.config.continous_isolation_check:
            self.check_continuous_isolation()

        # seeding backend
        LOGGER.info(f"\t+ Seeding backend with seed {self.config.seed}")
        self.seed()

        # clean up options
        if self.config.delete_cache:
            LOGGER.info("\t+ Model cache will be deleted after benchmark")

    def check_initial_isolation(self) -> None:
        if self.device.type == "cuda":
            LOGGER.info(f"\t+ Checking initial device(s) isolation of CUDA device(s): {CUDA_DEVICES}")
            only_this_process_is_running_on_cuda_devices(cuda_devices=CUDA_DEVICES, benchmark_pid=os.getpid())

    def check_continuous_isolation(self) -> None:
        if self.device.type == "cuda":
            LOGGER.info(f"\t+ Checking continuous device(s) isolation of CUDA device(s): {CUDA_DEVICES}")
            self.isolation_thread = Process(
                target=only_this_process_will_run_on_cuda_devices,
                kwargs={"cuda_devices": CUDA_DEVICES, "benchmark_pid": os.getpid()},
                daemon=True,
            )
            self.isolation_thread.start()

    def seed(self) -> None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def prepare_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_diffusion_pipeline():
            # diffusion pipelines takes a list of strings
            return input
        else:
            # models expect tensors on the target device
            for key, value in input.items():
                input[key] = value.to(self.device)

        return input

    def prepare_for_inference(self, **kwargs) -> None:
        pass

    # # symbolic tracing in transformers requires input names
    # def prepare_for_profiling(self, input_names: List[str]) -> Dict[str, Any]:
    #     pass

    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> "ModelOutput":
        return self.pretrained_model(**input, **kwargs)

    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> "ModelOutput":
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
            LOGGER.info("\t+ Deleting pretrained model")
            del self.pretrained_model
        gc.collect()

    def delete_model_cache(self) -> None:
        LOGGER.info("\t+ Deleting model cache")
        model_cache_folder = f"models/{self.model}".replace("/", "--")
        model_cache_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), model_cache_folder)
        shutil.rmtree(model_cache_path, ignore_errors=True)

    def clean(self) -> None:
        LOGGER.info(f"Cleaning {self.NAME} backend")
        self.delete_pretrained_model()

        if self.config.delete_cache:
            self.delete_model_cache()
