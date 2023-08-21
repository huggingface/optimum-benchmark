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
    PreTrainedTokenizer,
    PretrainedConfig,
    ImageProcessingMixin,
    FeatureExtractionMixin,
    ProcessorMixin,
    Pipeline,
)


from optimum_benchmark.utils import (
    DIFFUSION_TASKS,
    TEXT_GENERATION_TASKS,
    check_no_process_is_running_on_cuda_device,
    check_only_this_process_is_running_on_cuda_device,
)


PreTrainedProcessor = Union[
    PreTrainedTokenizer,
    ImageProcessingMixin,
    FeatureExtractionMixin,
    ProcessorMixin,
]

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
    # model and pipeline benchmarks
    pretrained_model: Union[PreTrainedModel, Pipeline]
    # only for model benchmarks
    pretrained_config: Optional[PretrainedConfig]
    pretrained_processor: Optional[PreTrainedProcessor]

    def __init__(self, model: str, task: str, device: str, hub_kwargs: DictConfig):
        self.model = model
        self.task = task
        self.device = torch.device(device)
        self.hub_kwargs = hub_kwargs

        if self.is_diffusion_pipeline():
            # for pipelines
            self.pretrained_config = None
            self.pretrained_processor = None
            self.model_type = self.task
        else:
            # for models
            self.pretrained_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model,
                **self.hub_kwargs,
            )
            self.model_type = self.pretrained_config.model_type

            try:
                # the processor sometimes contains information about the model's
                # input shapes that's not available in the config
                self.pretrained_processor = AutoProcessor.from_pretrained(
                    pretrained_model_name_or_path=self.model,
                    **self.hub_kwargs,
                )
            except ValueError:
                LOGGER.warning(f"Could not find the model's preprocessor")
                self.pretrained_processor = None

        # we're using this one as the default model_class which is used
        # for exporting the model to onnx for example. Although does suppose that
        # the model weights are pytorch weights so we might need to change somehow.
        self.automodel_class = TasksManager.get_model_class_for_task(
            task=self.task,
            model_type=self.model_type,
        )

    def is_text_generation_model(self) -> bool:
        return self.task in TEXT_GENERATION_TASKS

    def is_diffusion_pipeline(self) -> bool:
        return self.task in DIFFUSION_TASKS

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

    # compiling in openvino requires input shapes
    def prepare_for_inference(self, input_shapes: Dict[str, int]) -> None:
        pass

    # symbolic tracing in transformers requires input names
    def prepare_for_profiling(self, input_names: List[str]) -> None:
        pass

    # depending on the backend, we might need to prepare the model for training
    # in different ways although I prefer to pass these in the train method
    def prepare_for_training(
        self,
        training_dataset: Dataset,
        training_data_collator: Callable,
        training_arguments: Dict[str, Any],
    ) -> None:
        pass

    def forward(self, input: Dict[str, Tensor], **kwargs):
        raise NotImplementedError("Backend must implement forward method")

    def generate(self, input: Dict[str, Tensor], **kwargs):
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


def extract_shapes_from_diffusion_pipeline(
    pipeline: Pipeline,
) -> Dict[str, Any]:
    # this is the only way I found to extract a
    # diffusion pipeline's "output" shapes
    shapes = {}
    try:
        shapes["num_channels"] = pipeline.vae_encoder.config.out_channels
        shapes["height"] = pipeline.vae_encoder.config.sample_size
        shapes["width"] = pipeline.vae_encoder.config.sample_size
    except AttributeError:
        LOGGER.warning("Could not find the diffusion pipeline's output shapes")
        shapes["num_channels"] = -1
        shapes["height"] = -1
        shapes["width"] = -1

    return shapes


def extract_shapes_from_model_artifacts(
    config: PretrainedConfig,
    processor: Optional[PreTrainedProcessor] = None,
) -> Dict[str, Any]:
    shapes = {}
    artifacts_dict = {}

    config_dict = {k: v for k, v in config.to_dict().items() if v is not None}
    artifacts_dict.update(config_dict)

    if processor is not None and hasattr(processor, "to_dict"):
        processor_dict = {k: v for k, v in processor.to_dict().items() if v is not None}
        artifacts_dict.update(processor_dict)

    # text input
    shapes["vocab_size"] = artifacts_dict.get("vocab_size", 2)
    shapes["type_vocab_size"] = artifacts_dict.get("type_vocab_size", 2)

    # image input
    shapes["num_channels"] = artifacts_dict.get("num_channels", None)

    image_size = artifacts_dict.get("image_size", None)
    if image_size is None:
        # processors have different names for the image size
        image_size = artifacts_dict.get("size", None)

    if isinstance(image_size, (int, float)):
        shapes["height"] = image_size
        shapes["width"] = image_size
    elif isinstance(image_size, (list, tuple)):
        shapes["height"] = image_size[0]
        shapes["width"] = image_size[0]
    elif type(image_size) == dict and len(image_size) == 2:
        shapes["height"] = list(image_size.values())[0]
        shapes["width"] = list(image_size.values())[1]
    elif type(image_size) == dict and len(image_size) == 1:
        shapes["height"] = list(image_size.values())[0]
        shapes["width"] = list(image_size.values())[0]
    else:
        shapes["height"] = None
        shapes["width"] = None

    # classification labels (default to 2)
    shapes["num_labels"] = len(
        artifacts_dict.get("id2label", {"0": "LABEL_0", "1": "LABEL_1"})
    )

    # object detection labels (default to 2)
    shapes["num_queries"] = artifacts_dict.get("num_queries", 2)

    return shapes
