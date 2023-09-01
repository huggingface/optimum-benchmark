import os
import platform
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Type

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from .backends.neural_compressor.config import INCConfig
from .backends.onnxruntime.config import ORTConfig
from .backends.openvino.config import OVConfig
from .backends.pytorch.config import PyTorchConfig
from .benchmarks.inference.config import InferenceConfig
from .benchmarks.training.config import TrainingConfig
from .env_utils import get_cpu, get_cpu_ram_mb, get_gpus
from .import_utils import (
    accelerate_version,
    diffusers_version,
    optimum_version,
    transformers_version,
)
from .task_utils import infer_task_from_model_name_or_path

if TYPE_CHECKING:
    from .backends.base import Backend
    from .benchmarks.base import Benchmark

LOGGER = getLogger("experiment")

OmegaConf.register_new_resolver("infer_task", lambda model: infer_task_from_model_name_or_path(model))


@dataclass
class ExperimentConfig:
    # BACKEND CONFIGURATION
    backend: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # BENCHMARK CONFIGURATION
    benchmark: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # EXPERIMENT CONFIGURATION
    experiment_name: str
    # Model name or path (bert-base-uncased, google/vit-base-patch16-224, ...)
    model: str
    # Device name or path (cpu, cuda, cuda:0, ...)
    device: str
    # Task name (text-classification, image-classification, ...)
    task: str = "${infer_task:${model}}"

    # ADDITIONAL MODEL CONFIGURATION: Model revision, use_auth_token, trust_remote_code
    hub_kwargs: Dict = field(
        default_factory=lambda: {
            "revision": "main",
            "cache_dir": None,
            "force_download": False,
            "local_files_only": False,
        }
    )

    # ENVIRONMENT CONFIGURATION
    # TODO: add gpu info when available
    environment: Dict = field(
        default_factory=lambda: {
            "optimum_version": optimum_version(),
            "transformers_version": transformers_version(),
            "accelerate_version": accelerate_version(),
            "diffusers_version": diffusers_version(),
            "python_version": platform.python_version(),
            "system": platform.system(),
            "cpu": get_cpu(),
            "cpu_count": os.cpu_count(),
            "cpu_ram_mb": get_cpu_ram_mb(),
            "gpus": get_gpus(),
        }
    )

    def __post_init__(self) -> None:
        # if the number of available GPUs is 1, then we have no problem
        # torch and nvidia-smi will both index it as 0, otherwise:
        if "cuda" in self.device and len(self.environment["gpus"]) > 1:
            CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if CUDA_VISIBLE_DEVICES is None:
                raise ValueError(
                    "Multiple GPUs detected but CUDA_VISIBLE_DEVICES is not set. "
                    "This means that code might allocate resources from GPUs that are not intended to be used. "
                    "Please set `CUDA_VISIBLE_DEVICES` to the desired GPU ids."
                )
            CUDA_DEVICE_ORDER = os.environ.get("CUDA_DEVICE_ORDER", None)
            if CUDA_DEVICE_ORDER is None or CUDA_DEVICE_ORDER != "PCI_BUS_ID":
                LOGGER.warning(
                    "Multiple GPUs detected but CUDA_DEVICE_ORDER is not set. "
                    "This means that code might allocate resources from the wrong GPUs even if CUDA_VISIBLE_DEVICES is set. "
                    "Pytorch uses the `FASTEST_FIRST` order by default, which is not guaranteed to be the same as nvidia-smi. "
                    "`CUDA_DEVICE_ORDER` will be set to `PCI_BUS_ID` to ensure that the GPUs are allocated in the same order as nvidia-smi. "
                )
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# Register configurations
cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)
cs.store(group="backend", name="pytorch", node=PyTorchConfig)
cs.store(group="backend", name="onnxruntime", node=ORTConfig)
cs.store(group="backend", name="openvino", node=OVConfig)
cs.store(group="backend", name="neural_compressor", node=INCConfig)
cs.store(group="benchmark", name="inference", node=InferenceConfig)
cs.store(group="benchmark", name="training", node=TrainingConfig)


@hydra.main(version_base=None)
def run_experiment(experiment: DictConfig) -> None:
    # This is required to trigger __post_init__. Reference: https://github.com/omry/omegaconf/issues/377
    experiment: ExperimentConfig = OmegaConf.to_object(experiment)

    # Save the config
    OmegaConf.save(experiment, "hydra_config.yaml", resolve=True)

    # Allocate requested backend
    backend_factory: Type["Backend"] = get_class(experiment.backend._target_)
    backend: "Backend" = backend_factory(
        task=experiment.task,
        model=experiment.model,
        device=experiment.device,
        hub_kwargs=experiment.hub_kwargs,
    )
    try:
        # Configure the backend
        backend.configure(experiment.backend)
    except Exception as e:
        LOGGER.error("Error during backend configuration: %s", e)
        raise e

    # Allocate requested benchmark
    benchmark_factory: Type["Benchmark"] = get_class(experiment.benchmark._target_)
    benchmark: "Benchmark" = benchmark_factory()
    try:
        benchmark.configure(experiment.benchmark)
    except Exception as e:
        LOGGER.error("Error during benchmark configuration: %s", e)
        raise e

    try:
        # Run the benchmark
        benchmark.run(backend)
        # Save the benchmark results
        benchmark.save()
        # Clean up the backend
        backend.clean()
    except Exception as e:
        LOGGER.error("Error during benchmark execution: %s", e)
        backend.clean()
        raise e
