from typing import Any, ClassVar, Dict, List, Optional, Union, TYPE_CHECKING
from multiprocessing import Process
from abc import abstractmethod, ABC
from dataclasses import dataclass
from logging import getLogger
import os
import gc


import shutil
from psutil import cpu_count
from diffusers import DiffusionPipeline
from optimum.exporters import TasksManager
from transformers import (
    AutoConfig,
    AutoProcessor,
    ProcessorMixin,
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    ImageProcessingMixin,
    FeatureExtractionMixin,
)


if TYPE_CHECKING:
    from transformers.utils import ModelOutput
    from transformers import TrainerState


from .utils.base_utils import (
    extract_shapes_from_diffusion_pipeline,
    extract_shapes_from_model_artifacts,
)
from ..utils import (
    DIFFUSION_TASKS,
    TEXT_GENERATION_TASKS,
    check_no_process_is_running_on_cuda_device,
    check_only_this_process_is_running_on_cuda_device,
)


LOGGER = getLogger("backend")

PreTrainedProcessor = Union[
    PreTrainedTokenizer,
    ImageProcessingMixin,
    FeatureExtractionMixin,
    ProcessorMixin,
]


@dataclass
class BackendConfig(ABC):
    name: str
    version: str
    _target_: str

    # backend options
    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    # isolation options
    initial_isolation_check: bool = True
    continous_isolation_check: bool = True

    # clean up options
    delete_cache: bool = False

    def __post_init__(self):
        if self.inter_op_num_threads is not None:
            if self.inter_op_num_threads == -1:
                self.inter_op_num_threads = cpu_count()

        if self.intra_op_num_threads is not None:
            if self.intra_op_num_threads == -1:
                self.intra_op_num_threads = cpu_count()


class Backend(ABC):
    name: str
    config: ClassVar[BackendConfig]

    pretrained_model: Union[PreTrainedModel, DiffusionPipeline]
    pretrained_processor: Optional[PreTrainedProcessor]
    pretrained_config: Optional[PretrainedConfig]

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]):
        self.model = model
        self.task = task
        self.device = device
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
                LOGGER.warning("Could not find the model's preprocessor")
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
                    "Asked to check the initial device isolation, "
                    "but the variable CUDA_VISIBLE_DEVICES was not set. "
                    "Defaulting to checking on the first device."
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
                    "Asked to check the continuous device isolation, "
                    "but the variable CUDA_VISIBLE_DEVICES was not set. "
                    "Defaulting to checking on the first device."
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
        LOGGER.info(f"Configuring {config.name} backend")
        self.config = config

        # isolation options
        if self.config.initial_isolation_check:
            LOGGER.info("\t+ Checking initial device isolation")
            self.check_initial_isolation()
        if self.config.continous_isolation_check:
            LOGGER.info("\t+ Checking contineous device isolation")
            self.check_continuous_isolation()

        # clean up options
        if self.config.delete_cache:
            LOGGER.info("\t+ Model cache will be deleted after benchmark")

    # compiling in openvino requires input shapes
    def prepare_for_inference(self, input_shapes: Dict[str, int]) -> Dict[str, Any]:
        pass

    # symbolic tracing in transformers requires input names
    def prepare_for_profiling(self, input_names: List[str]) -> Dict[str, Any]:
        pass

    def forward(self, input: Dict[str, Any], **kwargs) -> "ModelOutput":
        raise NotImplementedError("Backend must implement forward method")

    def generate(self, input: Dict[str, Any], **kwargs) -> "ModelOutput":
        raise NotImplementedError("Backend must implement generate method")

    def train(self) -> "TrainerState":
        raise NotImplementedError("Backend must implement train method")

    def delete_pretrained_model(self) -> None:
        try:
            del self.pretrained_model
        except AttributeError:
            # benchmark might fail before the model is loaded
            pass

        gc.collect()

    def delete_model_cache(self) -> None:
        model_cache_path = "models--" + self.model.replace("/", "--")
        model_cache_path = os.path.join(
            os.path.expanduser("~/.cache/huggingface/hub"), model_cache_path
        )
        shutil.rmtree(model_cache_path, ignore_errors=True)

    def clean(self) -> None:
        LOGGER.info(f"Cleaning {self.config.name} backend")
        self.delete_pretrained_model()

        if self.config.delete_cache:
            self.delete_model_cache()

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
