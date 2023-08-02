from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, MISSING
from abc import abstractmethod, ABC
from omegaconf import DictConfig
from logging import getLogger
from psutil import cpu_count
from torch import Tensor
import inspect
import shutil
import torch
import os
import gc


from optimum.exporters import TasksManager
from transformers import AutoConfig, PreTrainedModel
from transformers.onnx.utils import get_preprocessor


from optimum_benchmark.utils import (
    LLM_MODEL_TYPES,
    check_no_process_is_running_on_cuda_device,
    check_only_this_process_is_running_on_cuda_device,
)

LOGGER = getLogger("backend")


@dataclass
class BackendConfig(ABC):
    name: str = MISSING
    version: str = MISSING
    _target_: str = MISSING

    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    # clean up options
    delete_cache: bool = False

    # isolation options
    initial_isolation_check: bool = True
    continous_isolation_check: bool = True


class Backend(ABC):
    pretrained_model: PreTrainedModel

    def __init__(self, model: str, task: str, device: str, hub_kwargs: DictConfig):
        self.model = model
        self.task = task
        self.device = torch.device(device)
        self.hub_kwargs = hub_kwargs

        # transformers autoconfig and automodel
        if self.task == "stable-diffusion":
            self.pretrained_config = None
            self.model_type = "stable-diffusion"
        else:
            self.pretrained_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model,
                **self.hub_kwargs,
            )
            self.model_type = self.pretrained_config.model_type

        self.automodel_class = TasksManager.get_model_class_for_task(
            model_type=self.model_type,
            task=self.task,
            framework="pt",
        )

    def check_initial_isolation(self) -> None:
        if self.device.type == "cuda":
            LOGGER.info("Checking initial device isolation")
            check_no_process_is_running_on_cuda_device(self.device)
            LOGGER.info(
                f"Initial device isolation check passed: no process is running on device {self.device}"
            )

    def check_continous_isolation(self) -> None:
        if self.device.type == "cuda":
            LOGGER.info("Checking contineous device isolation")
            from multiprocessing import Process

            self.isolation_thread = Process(
                target=check_only_this_process_is_running_on_cuda_device,
                args=(self.device, os.getpid()),
                daemon=True,
            )
            self.isolation_thread.start()

        elif self.device.type == "cpu":
            pass

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

        # clean up options
        if config.delete_cache:
            LOGGER.info("\t+ Will delete cache after benchmarking")
            self.delete_cache = True
        else:
            self.delete_cache = False

        # isolation options
        if config.initial_isolation_check:
            self.check_initial_isolation()
        if config.continous_isolation_check:
            self.check_continous_isolation()

    def prepare_for_forward(self, input_shapes: Dict[str, int]) -> None:
        pass

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        pass

    def prepare_for_training(self, train_dataset: Dict[str, int]) -> None:
        pass

    @abstractmethod
    def forward(self, input: Dict[str, Tensor]):
        raise NotImplementedError("Backend must implement forward method")

    @abstractmethod
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

    def can_generate(self) -> bool:
        return (
            hasattr(self.pretrained_model, "can_generate")
            and self.pretrained_model.can_generate()
        )

    def generate_dummy_input(
        self, mode, **input_shapes
    ) -> Tuple[Dict[str, Tensor], Dict[str, int]]:
        # TODO: should fallback to generating dummy inputs based on the task after it fails for a model type
        assert mode in ["forward", "generate"], f"mode {mode} not supported"

        if self.model_type == "stable-diffusion":
            # patch for stable-diffusion not recognized by TasksManager
            return {
                "prompt": ["This is a sample prompt"] * input_shapes["batch_size"]
            }, {"batch_size": input_shapes["batch_size"]}

        if self.model_type in LLM_MODEL_TYPES:
            # patch for some LLM model types not recognized by TasksManager
            return {
                "input_ids": torch.randint(
                    0,
                    self.pretrained_config.vocab_size,
                    (input_shapes["batch_size"], input_shapes["sequence_length"]),
                ),
                "attention_mask": torch.ones(
                    (input_shapes["batch_size"], input_shapes["sequence_length"])
                ),
            }, {
                "batch_size": input_shapes["batch_size"],
                "sequence_length": input_shapes["sequence_length"],
            }

        onnx_config = TasksManager.get_exporter_config_constructor(
            model_type=self.model_type,
            task=self.task,
            exporter="onnx",
        )(self.pretrained_config)

        generator_classes = onnx_config.DUMMY_INPUT_GENERATOR_CLASSES

        if mode == "forward":
            input_names = list(onnx_config.inputs.keys())
        elif mode == "generate":
            input_names = get_preprocessor(self.model).model_input_names
        else:
            raise ValueError(f"Unknown mode {mode}")

        LOGGER.info(f"Generating dummy inputs for {input_names}")

        dummy_input = dict()
        dummy_input_shapes = dict()
        for input_name in input_names:
            generator = None
            for generator_class in generator_classes:
                supported_generator_input_names = generator_class.SUPPORTED_INPUT_NAMES

                if input_name in supported_generator_input_names:
                    supported_generator_input_shapes = {
                        input_shape: input_shapes[input_shape]
                        for input_shape in input_shapes
                        if input_shape
                        in inspect.signature(generator_class.__init__).parameters
                    }
                    try:
                        generator = generator_class(
                            task=self.task,
                            normalized_config=onnx_config.NORMALIZED_CONFIG_CLASS(
                                self.pretrained_config
                            ),
                            **supported_generator_input_shapes,
                        )
                    except Exception as e:
                        try:
                            generator = generator_class(
                                task=self.task,
                                normalized_config=onnx_config.NORMALIZED_CONFIG_CLASS(
                                    self.pretrained_config.decoder
                                ),
                                **supported_generator_input_shapes,
                            )
                        except Exception as e:
                            generator = generator_class(
                                task=self.task,
                                normalized_config=onnx_config.NORMALIZED_CONFIG_CLASS(
                                    self.pretrained_config
                                ),
                                **supported_generator_input_shapes,
                            )
                    # we found a generator for this input name, let's use it
                    break

            # if no generator was found, raise an error
            if generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            dummy_input[input_name] = generator.generate(input_name).to(self.device)
            dummy_input_shapes.update(supported_generator_input_shapes)

            if input_name == "attention_mask":
                # patch for until sparse attention is supported
                dummy_input["attention_mask"] = torch.ones_like(
                    dummy_input["attention_mask"]
                )

        return dummy_input, dummy_input_shapes
