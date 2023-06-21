from dataclasses import dataclass, MISSING
from logging import getLogger
import platform
import os

from optimum.exporters import TasksManager
from omegaconf import DictConfig, OmegaConf
from optimum.version import __version__ as optimum_version
from transformers import __version__ as transformers_version

from src.backend.base import BackendConfig
from src.benchmark.inference import BenchmarkConfig
from src.utils import get_device_name, get_total_memory

OmegaConf.register_new_resolver(
    "infer_task",
    lambda model, subfolder, revision: TasksManager.infer_task_from_model(
        model=model, subfolder=subfolder, revision=revision
    ),
)

LOGGER = getLogger("experiment")  # will be used in schema validation


@dataclass
class ExperimentConfig:
    # EXPERIMENT CONFIGURATION
    experiment_name: str = MISSING

    # BACKEND CONFIGURATION
    backend: BackendConfig = MISSING

    # BENCHMARK CONFIGURATION
    benchmark: BenchmarkConfig = MISSING

    # MODEL CONFIGURATION
    # Model name or path (bert-base-uncased, google/vit-base-patch16-224, ...)
    model: str = MISSING

    # DEVICE CONFIGURATION (might be moved to backend config in the future)
    # Device on which the model is loaded and run (cpu, cuda, hpu...)
    device: str = "cpu"

    # ADDITIONAL MODEL CONFIGURATION
    # Task (sequence-classification, token-classification, question-answering, ...)
    # TasksManager will try to infer the task from the model
    task: str = "${infer_task:${model},${cache_kwargs.subfolder},${cache_kwargs.revision}}"
    # Model revision, use_auth_token, trust_remote_code
    cache_kwargs: DictConfig = DictConfig(
        {
            "revision": "main",
            "subfolder": "",

            "cache_dir": None,
            "proxies": None,

            "force_download": False,
            "resume_download": False,
            "local_files_only": False,
            "use_auth_token": False,
        }
    )
    # ENVIRONMENT CONFIGURATION
    environment: DictConfig = DictConfig(
        {
            "optimum_version": optimum_version,
            "transformers_version": transformers_version,
            "python_version": platform.python_version(),
            "system": platform.system(),
            "cpu_count": os.cpu_count(),
            "cpu": get_device_name(device="cpu"),
            "cpu_ram_mb": get_total_memory(device="cpu"),
            "gpu": get_device_name(device="cuda"),
            "gpu_vram_mb": get_total_memory(device="cuda"),
        }
    )
