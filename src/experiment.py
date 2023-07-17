from dataclasses import dataclass, MISSING
from omegaconf import DictConfig, OmegaConf
from logging import getLogger
import platform
import os

from optimum.exporters import TasksManager
from optimum.version import __version__ as optimum_version
from transformers import __version__ as transformers_version

from src.backend.base import BackendConfig
from src.benchmark.inference import BenchmarkConfig
from src.utils import get_cpu, get_cpu_ram_mb

OmegaConf.register_new_resolver(
    "infer_task",
    lambda model, subfolder, revision : TasksManager.infer_task_from_model(
            model=model,
            subfolder=subfolder,
            revision=revision,
        ),
)

LOGGER = getLogger("experiment")


@dataclass
class ExperimentConfig:
    # BACKEND CONFIGURATION
    backend: BackendConfig = MISSING  # type: ignore

    # BENCHMARK CONFIGURATION
    benchmark: BenchmarkConfig = MISSING  # type: ignore

    # EXPERIMENT CONFIGURATION
    experiment_name: str = MISSING  # type: ignore
    # Model name or path (bert-base-uncased, google/vit-base-patch16-224, ...)
    model: str = MISSING  # type: ignore
    # Device name or path (cpu, cuda, cuda:0, ...)
    device: str = MISSING  # type: ignore
    # task
    task: str = "${infer_task:${model}, ${cache_kwargs.subfolder}, ${cache_kwargs.revision}}"

    # ADDITIONAL MODEL CONFIGURATION: Model revision, use_auth_token, trust_remote_code
    cache_kwargs: DictConfig = DictConfig(
        {
            "revision": "main",
            "subfolder": "",
            "cache_dir": None,
            # "proxies": None,
            "force_download": False,
            # "resume_download": False,
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
            "cpu": get_cpu(),
            "cpu_count": os.cpu_count(),
            "cpu_ram_mb": get_cpu_ram_mb(),
        }
    )
