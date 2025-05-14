import gc
from abc import ABC
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Dict, Generic, Optional

import datasets.utils.logging as datasets_logging
import transformers.utils.logging as transformers_logging
from safetensors.torch import save_model
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, TrainerState, set_seed

from ..hub_utils import HF_API
from ..import_utils import is_torch_available
from ..task_utils import TEXT_GENERATION_TASKS
from .config import BackendConfigT
from .diffusers_utils import (
    extract_diffusers_shapes_from_model,
    get_diffusers_auto_pipeline_class_for_task,
    get_diffusers_pretrained_config,
)
from .timm_utils import extract_timm_shapes_from_config, get_timm_model_creator, get_timm_pretrained_config
from .transformers_utils import (
    PretrainedProcessor,
    extract_transformers_shapes_from_artifacts,
    fast_weights_init,
    get_transformers_auto_model_class_for_task,
    get_transformers_generation_config,
    get_transformers_pretrained_config,
    get_transformers_pretrained_processor,
)

if is_torch_available():
    import torch

datasets_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    tmpdir: TemporaryDirectory
    model_shapes: Dict[str, int]
    no_weights_model_path: Optional[Path]

    config: BackendConfigT
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
            self.automodel_loader = get_diffusers_auto_pipeline_class_for_task(self.config.task)
            self.model_shapes = extract_diffusers_shapes_from_model()
            self.pretrained_processor = None
            self.generation_config = None

        elif self.config.library == "timm":
            self.logger.info("\t+ Benchmarking a Timm model")
            self.pretrained_config = get_timm_pretrained_config(self.config.model)
            self.model_shapes = extract_timm_shapes_from_config(self.pretrained_config)
            self.automodel_loader = get_timm_model_creator()
            self.pretrained_processor = None
            self.generation_config = None

        elif self.config.library == "llama_cpp":
            self.logger.info("\t+ Benchmarking a LlamaCpp model")
            self.pretrained_processor = None
            self.pretrained_config = None
            self.generation_config = None
            self.automodel_loader = None
            self.model_shapes = {}

        else:
            self.logger.info("\t+ Benchmarking a Transformers model")
            self.automodel_loader = get_transformers_auto_model_class_for_task(self.config.task, self.config.model_type)
            self.generation_config = get_transformers_generation_config(self.config.model, **self.config.model_kwargs)
            self.pretrained_config = get_transformers_pretrained_config(self.config.model, **self.config.model_kwargs)
            self.pretrained_processor = get_transformers_pretrained_processor(
                self.config.processor, **self.config.processor_kwargs
            )
            self.model_shapes = extract_transformers_shapes_from_artifacts(
                self.pretrained_config, self.pretrained_processor
            )

    def seed(self) -> None:
        set_seed(self.config.seed)

    def download_pretrained_model(self) -> None:
        model_snapshot_folder = HF_API.snapshot_download(
            self.config.model,
            revision=self.config.model_kwargs.get("revision", None),
            cache_dir=self.config.model_kwargs.get("cache_dir", None),
            force_download=self.config.model_kwargs.get("force_download", False),
            local_files_only=self.config.model_kwargs.get("local_files_only", False),
        )

        if self.config.task in TEXT_GENERATION_TASKS:
            self.generation_config.eos_token_id = None
            self.generation_config.pad_token_id = None
            self.generation_config.save_pretrained(save_directory=model_snapshot_folder)

    def create_no_weights_model_fast(self) -> None:
        model_path = Path(
            HF_API.hf_hub_download(self.config.model, filename="config.json", cache_dir=self.tmpdir.name)
        ).parent
        save_model(model=torch.nn.Linear(1, 1), filename=model_path / "model.safetensors", metadata={"format": "pt"})

        if self.pretrained_processor is not None:
            self.pretrained_processor.save_pretrained(save_directory=model_path)
        if self.pretrained_config is not None:
            self.pretrained_config.save_pretrained(save_directory=model_path)

        if self.config.task in TEXT_GENERATION_TASKS:
            self.generation_config.eos_token_id = None
            self.generation_config.pad_token_id = None
            self.generation_config.save_pretrained(save_directory=model_path)

        self.no_weights_model_path = model_path

    def create_no_weights_model_slow(self) -> None:
        self.create_no_weights_model_fast()

        with fast_weights_init():
            # unlike Transformers, TXI won't accept any missing tensors so we need to materialize the model
            dummy = self.automodel_loader.from_pretrained(
                self.no_weights_model_path, device_map="auto", **self.config.model_kwargs
            )
            dummy.save_pretrained(self.no_weights_model_path)
            del dummy

        torch.cuda.empty_cache()
        gc.collect()

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method is used to prepare and register the inputs before passing them to the model.
        It can be used to move the inputs to the correct device, or rename their keys.
        """
        return inputs

    def load(self) -> None:
        raise NotImplementedError("Backend must implement load method")

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
