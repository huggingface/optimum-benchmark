from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, MISSING
from multiprocessing import Process
from abc import abstractmethod, ABC
from logging import getLogger
import shutil
import os
import gc

import torch
from torch import Tensor
from datasets import Dataset
from psutil import cpu_count
from omegaconf import DictConfig
from optimum.exporters import TasksManager
from transformers import (
    AutoConfig,
    AutoProcessor,
    PreTrainedModel,
    Pipeline,
)


from optimum_benchmark.utils import (
    check_no_process_is_running_on_cuda_device,
    check_only_this_process_is_running_on_cuda_device,
)

LOGGER = getLogger("backend")


@dataclass
class BackendConfig(ABC):
    name: str = MISSING
    version: str = MISSING
    _target_: str = MISSING

    # backend options
    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    # isolation options
    initial_isolation_check: bool = True
    continous_isolation_check: bool = True

    # clean up options
    delete_cache: bool = False


class Backend(ABC):
    pretrained_model: Union[PreTrainedModel, Pipeline]
    pretrained_config: Optional[PretrainedConfig]
    pretrained_preprocessor: Optional[
        Union[
            PreTrainedTokenizer,
            ImageProcessingMixin,
            FeatureExtractionMixin,
            ProcessorMixin,
        ]
    ]

    def __init__(self, model: str, task: str, device: str, hub_kwargs: DictConfig):
        self.model = model
        self.task = task
        self.device = torch.device(device)
        self.hub_kwargs = hub_kwargs

        if self.task in ["stable-diffusion", "stable-diffusion-xl"]:
            # for pipelines
            self.pretrained_config = None
            self.pretrained_preprocessor = None
            self.model_type = self.task
        else:
            # for models
            self.pretrained_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model,
                **self.hub_kwargs,
            )
            self.model_type = self.pretrained_config.model_type

            try:
                # the processor someyimes contain information about the model's
                # input and output shapes that are not available in the config
                self.pretrained_preprocessor = AutoProcessor.from_pretrained(
                    pretrained_model_name_or_path=self.model,
                    **self.hub_kwargs,
                )
            except ValueError:
                LOGGER.warning(f"Could not find the model's preprocessor")
                self.pretrained_preprocessor = None

        # we're using this one as the default model_class which is used
        # for exporting the model to onnx for example. Although does suppose that
        # the model weights are pytorch weights
        self.automodel_class = TasksManager.get_model_class_for_task(
            model_type=self.model_type,
            task=self.task,
        )

    def can_generate(self) -> bool:
        return self.task in [
            "text-generation",
            "text2text-generation",
            "image-to-text",
            "automatic-speech-recognition",
        ]

    def check_initial_isolation(self) -> None:
        if self.device.type == "cuda":
            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_devices is None:
                LOGGER.warning(
                    "Asked to check the initial device isolation, but the variable CUDA_VISIBLE_DEVICES was not set. Defaulting to checking on the first device."
                )
                device_ids = {self.device.index if self.device.index is not None else 0}
            else:
                device_ids = {
                    int(device_index) for device_index in cuda_devices.split(",")
                }
            check_no_process_is_running_on_cuda_device(device_ids)

    def check_continuous_isolation(self) -> None:
        if self.device.type == "cuda":
            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_devices is None:
                LOGGER.warning(
                    "Asked to check the continuous device isolation, but the variable CUDA_VISIBLE_DEVICES was not set. Defaulting to checking on the first device."
                )
                device_ids = {self.device.index if self.device.index is not None else 0}
            else:
                device_ids = {
                    int(device_index) for device_index in cuda_devices.split(",")
                }

            self.isolation_thread = Process(
                target=check_only_this_process_is_running_on_cuda_device,
                args=(device_ids, os.getpid()),
                daemon=True,
            )
            self.isolation_thread.start()

    @abstractmethod
    def configure(self, config: BackendConfig) -> None:
        self.config = config

        LOGGER.info(f"Configuring {config.name} backend")

        self.config = config
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

        # clean up options
        if config.delete_cache:
            LOGGER.info("\t+ Will delete model cache after benchmarking")
        self.delete_cache = config.delete_cache

        # isolation options
        if config.initial_isolation_check:
            LOGGER.info("\t+ Checking initial device isolation")
            self.check_initial_isolation()
        if config.continous_isolation_check:
            LOGGER.info("\t+ Checking contineous device isolation")
            self.check_continuous_isolation()

    # compiling in openvino requires static shapes
    def prepare_for_inference(self, static_shapes: Dict[str, int]) -> None:
        pass

    # symbolic tracing intransformers requires input names
    def prepare_for_profiling(self, input_names: List[str]) -> None:
        pass

    # depending on the backend, we might need to prepare the model for training
    # although I prefer to pass these in the train method
    def prepare_for_training(
        self,
        training_dataset: Dataset,
        training_data_collator: Callable,
        training_arguments: Dict[str, Any],
    ) -> None:
        pass

    def forward(self, input: Dict[str, Tensor], **kwargs):
        raise NotImplementedError("Backend must implement forward method")

    def generate(self, input: Dict[str, Tensor], **kwargs) -> str:
        raise NotImplementedError("Backend must implement generate method")

    def train(self):
        raise NotImplementedError("Backend must implement train method")

    def delete_pretrained_model(self) -> None:
        if hasattr(self, "pretrained_model"):
            del self.pretrained_model
            gc.collect()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def delete_model_hub_cache(self) -> None:
        model_cache_path = "models--" + self.model.replace("/", "--")
        model_cache_path = os.path.join(
            os.path.expanduser("~/.cache/huggingface/hub"), model_cache_path
        )
        shutil.rmtree(model_cache_path, ignore_errors=True)

    def clean(self) -> None:
        LOGGER.info(f"Cleaning backend")
        self.delete_pretrained_model()

        if self.delete_cache:
            self.delete_model_hub_cache()
