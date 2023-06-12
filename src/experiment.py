from dataclasses import dataclass, MISSING
from logging import getLogger
import platform
import os

from omegaconf import DictConfig
from optimum.version import __version__ as optimum_version
from transformers import __version__ as transformers_version

from src.backend.base import BackendConfig
from src.benchmark.inference import BenchmarkConfig
from src.utils import get_device_name, get_total_memory


LOGGER = getLogger("experiment")


@dataclass
class ExperimentConfig:
    # BACKEND CONFIGURATION
    # The backend to use for recording timing (pytorch, onnxruntime, ...)
    backend: BackendConfig = MISSING

    # BENCHMARK CONFIGURATION
    # The kind of benchmark to run (inference, training, ...)
    benchmark: BenchmarkConfig = MISSING

    # EXPERIMENT CONFIGURATION
    # Experiment name
    experiment_name: str = MISSING

    # MODEL CONFIGURATION
    # Name of the model to run (bert-base-uncased, ...)
    model: str = MISSING
    # Device on which the model is loaded and run (cpu, cuda, ...)
    device: str = MISSING
    # Task on which the model is run (sequence-classification, ...)
    task: str = "${infer_task:${model}}"

    # ENVIRONMENT CONFIGURATION
    environment: DictConfig = DictConfig(
        {
            "optimum_version": optimum_version,
            "transformers_version": transformers_version,
            "python_version": platform.python_version(),
            "system": platform.system(),
            "cpu": get_device_name(device="cpu"),
            "cpu_count": os.cpu_count(),
            "cpu_ram_mb": get_total_memory(device="cpu"),
            "gpu": get_device_name(device="cuda"),
            "gpu_vram_mb": get_total_memory(device="cuda"),
        }
    )
