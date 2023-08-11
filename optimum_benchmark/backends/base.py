from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, MISSING
from abc import abstractmethod, ABC
from omegaconf import DictConfig
from logging import getLogger
from psutil import cpu_count
from torch import Tensor
import shutil
import torch
import os
import gc


from datasets import Dataset
from optimum.exporters import TasksManager
from transformers.tokenization_utils import PreTrainedTokenizer

from transformers import (
    # configs
    AutoConfig,
    PretrainedConfig,
    # models
    PreTrainedModel,
    # preprocessors
    AutoProcessor,
    PreTrainedTokenizer,
    ImageProcessingMixin,
    FeatureExtractionMixin,
    ProcessorMixin,
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
    pretrained_model: PreTrainedModel
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
            # diffusers
            self.pretrained_config = None
            self.pretrained_preprocessor = None
            self.model_type = self.task
        else:
            # transformers autoconfig and automodel
            self.pretrained_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model,
                **self.hub_kwargs,
            )
            self.pretrained_preprocessor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=self.model,
                **self.hub_kwargs,
            )
            self.model_type = self.pretrained_config.model_type

        self.automodel_class = TasksManager.get_model_class_for_task(
            model_type=self.model_type,
            task=self.task,
        )

    def can_generate(self) -> bool:
        from accelerate import init_empty_weights

        with init_empty_weights():
            dummy_model = self.automodel_class.from_pretrained(
                pretrained_model_name_or_path=self.model,
                **self.hub_kwargs,
            )

        return hasattr(dummy_model, "can_generate") and dummy_model.can_generate()

    def check_initial_isolation(self) -> None:
        if self.device.type == "cuda":
            check_no_process_is_running_on_cuda_device(self.device)

    def check_continous_isolation(self) -> None:
        if self.device.type == "cuda":
            from multiprocessing import Process

            self.isolation_thread = Process(
                target=check_only_this_process_is_running_on_cuda_device,
                args=(self.device, os.getpid()),
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
            LOGGER.info("\t+ Will delete cache after benchmarking")
            self.delete_cache = True
        else:
            self.delete_cache = False

        # isolation options
        if config.initial_isolation_check:
            LOGGER.info("\t+ Checking initial device isolation")
            self.check_initial_isolation()
        if config.continous_isolation_check:
            LOGGER.info("\t+ Checking contineous device isolation")
            self.check_continous_isolation()

    def prepare_for_inference(
        self,
        input_names: List[str],
        input_shapes: Dict[str, int],
    ) -> None:
        pass

    def prepare_for_profiling(
        self,
        input_names: List[str],
        input_shapes: Dict[str, int],
    ) -> None:
        pass

    def prepare_for_training(
        self,
        training_dataset: Dataset,
        training_data_collator: Callable,
        training_arguments: Dict[str, Any],
    ) -> None:
        pass

    @abstractmethod
    def forward(self, input: Dict[str, Tensor], **kwargs):
        raise NotImplementedError("Backend must implement forward method")

    @abstractmethod
    def generate(self, input: Dict[str, Tensor], **kwargs) -> str:
        raise NotImplementedError("Backend must implement generate method")

    @abstractmethod
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
