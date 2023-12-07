import gc
import os
import random
import shutil
from abc import ABC
from logging import getLogger
from multiprocessing import Process
from typing import Any, Callable, ClassVar, Dict, Generic, Optional, Union

import numpy as np
from optimum.exporters import TasksManager
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

from ..task_utils import DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from .config import BackendConfigT
from .isolation_utils import check_cuda_continuous_isolation
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
    isolation_thread: Optional[Process]
    pretrained_model: Union[PreTrainedModel, Pipeline]
    pretrained_config: Optional[PretrainedConfig]
    pretrained_processor: Optional[PreTrainedProcessor]
    pretrained_generation_config: Optional[GenerationConfig]
    automodel_class: Callable[..., PreTrainedModel]

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]):
        self.task = task
        self.model = model
        self.device = device
        self.hub_kwargs = hub_kwargs

        if self.is_diffusion_pipeline():
            self.library = "diffusers"
            self.model_type = self.task
            self.pretrained_config = None
            self.pretrained_processor = None
        else:
            self.library = "transformers"
            self.pretrained_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model, **self.hub_kwargs
            )
            self.model_type = self.pretrained_config.model_type

            try:
                # sometimes contains information about the model's
                # input shapes that are not available in the config
                self.pretrained_processor = AutoProcessor.from_pretrained(
                    pretrained_model_name_or_path=self.model, **self.hub_kwargs
                )
            except ValueError:
                # sometimes the processor is not available or can't be determined/detected
                LOGGER.warning("Could not find the model's preprocessor")
                self.pretrained_processor = None

        if self.is_text_generation_model():
            try:
                self.pretrained_generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name=self.model, **self.hub_kwargs
                )
            except Exception:
                LOGGER.warning("Could not find the model's generation config")
                self.pretrained_generation_config = None
        else:
            self.pretrained_generation_config = None

        self.automodel_class = TasksManager.get_model_class_for_task(
            framework="pt",  # TODO: make this configurable to add support for other frameworks
            task=self.task,
            library=self.library,
            model_type=self.model_type,
        )

    def is_text_generation_model(self) -> bool:
        return self.task in TEXT_GENERATION_TASKS

    def is_diffusion_pipeline(self) -> bool:
        return self.task in DIFFUSION_TASKS

    def configure(self, config: BackendConfigT) -> None:
        LOGGER.info(f"Configuring {self.NAME} backend")
        self.config = config

        # isolation options
        if self.config.continuous_isolation:
            LOGGER.info("\t+ Running continuous isolation check")
            self.check_continuous_isolation()

        # clean up options
        if self.config.delete_cache:
            LOGGER.info("\t+ Model cache will be deleted after benchmark")

    def check_continuous_isolation(self) -> None:
        if self.device == "cuda":
            self.isolation_process = Process(
                target=check_cuda_continuous_isolation,
                kwargs={
                    "isolated_pid": os.getpid(),
                    "isolation_check_interval": self.config.isolation_check_interval,
                },
                daemon=True,
            )
            self.isolation_process.start()
            LOGGER.info(f"\t+ Started isolation process with PID {self.isolation_process.pid}")
        else:
            raise ValueError("Continuous isolation is only supported for CUDA devices")

    def seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: move this to only backends that need it (non cpu backends)
        if self.is_diffusion_pipeline():
            return inputs  # diffusion pipelines takes a list of strings
        else:
            LOGGER.info(f"\t+ Moving inputs tensors to device {self.device}")
            for key, value in inputs.items():
                inputs[key] = value.to(self.device)

        return inputs

    def prepare_for_inference(self, **kwargs) -> None:
        pass

    def forward(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model(**input, **kwargs)

    def generate(self, input: Dict[str, Any], kwargs: Dict[str, Any]) -> ModelOutput:
        return self.pretrained_model.generate(**input, **kwargs)

    def train(self, **kwargs) -> TrainerState:
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
        LOGGER.info("\t+ Deleting pretrained model")
        del self.pretrained_model
        gc.collect()

    def delete_hf_model_cache(self) -> None:
        LOGGER.info("\t+ Deleting model cache")
        model_cache_folder = f"models/{self.model}".replace("/", "--")
        model_cache_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), model_cache_folder)
        shutil.rmtree(model_cache_path, ignore_errors=True)

    def terminate_isolation_process(self) -> None:
        LOGGER.info("\t+ Terminating isolation process")
        self.isolation_process.kill()
        self.isolation_process.join()
        self.isolation_process.close()

    def clean(self) -> None:
        LOGGER.info(f"Cleaning {self.NAME} backend")

        if self.config.continuous_isolation:
            self.terminate_isolation_process()

        if hasattr(self, "pretrained_model"):
            self.delete_pretrained_model()

        if self.config.delete_cache:
            self.delete_hf_model_cache()

        gc.collect()
