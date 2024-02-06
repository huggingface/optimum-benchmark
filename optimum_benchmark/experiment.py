import os
import platform
from logging import getLogger
from dataclasses import dataclass, field
from typing import Any, Dict, Type, Optional, TYPE_CHECKING

from hydra.utils import get_class

from .benchmarks.config import BenchmarkConfig
from .launchers.config import LauncherConfig
from .backends.config import BackendConfig
from .import_utils import (
    transformers_version,
    accelerate_version,
    diffusers_version,
    optimum_version,
    timm_version,
    peft_version,
)
from .env_utils import (
    get_git_revision_hash,
    get_cpu_ram_mb,
    get_gpus,
    get_cpu,
)

if TYPE_CHECKING:
    # avoid importing any torch to be able to set
    # the CUDA_VISIBLE_DEVICES environment variable
    # in BackendConfig __post_init__
    from .benchmarks.base import Benchmark
    from .launchers.base import Launcher
    from .backends.base import Backend


LOGGER = getLogger("experiment")


@dataclass
class ExperimentConfig:
    # BACKEND CONFIGURATION
    backend: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # LAUNCHER CONFIGURATION
    launcher: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # BENCHMARK CONFIGURATION
    benchmark: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # Experiment name
    experiment_name: str

    task: Optional[str] = None  # deprecated
    model: Optional[str] = None  # deprecated
    device: Optional[str] = None  # deprecated
    library: Optional[str] = None  # deprecated

    # ENVIRONMENT CONFIGURATION
    environment: Dict = field(
        default_factory=lambda: {
            "cpu": get_cpu(),
            "gpus": get_gpus(),
            "cpu_count": os.cpu_count(),
            "system": platform.system(),
            "cpu_ram_mb": get_cpu_ram_mb(),
            "python_version": platform.python_version(),
            # libraries
            "transformers_version": transformers_version(),
            "transformers_commit": get_git_revision_hash("transformers", os.environ.get("TRANSFORMERS_PATH", None)),
            "accelerate_version": accelerate_version(),
            "accelerate_commit": get_git_revision_hash("accelerate", os.environ.get("ACCELERATE_PATH", None)),
            "optimum_version": optimum_version(),
            "optimum_commit": get_git_revision_hash("optimum", os.environ.get("OPTIMUM_PATH", None)),
            "diffusers_version": diffusers_version(),
            "diffusers_commit": get_git_revision_hash("diffusers", os.environ.get("DIFFUSERS_PATH", None)),
            "timm_version": timm_version(),
            "timm_commit": get_git_revision_hash("timm", os.environ.get("TIMM_PATH", None)),
            "peft_version": peft_version(),
            "peft_commit": get_git_revision_hash("peft", os.environ.get("PEFT_PATH", None)),
        }
    )


def run(benchmark_config: BenchmarkConfig, backend_config: BackendConfig) -> Dict[str, Any]:
    try:
        # Allocate requested backend
        backend_factory: Type[Backend] = get_class(backend_config._target_)
        backend: Backend = backend_factory(backend_config)
    except Exception as e:
        LOGGER.error(f"Error during backend allocation: {e}")
        raise e

    try:
        # Allocate requested benchmark
        benchmark_factory: Type[Benchmark] = get_class(benchmark_config._target_)
        benchmark: Benchmark = benchmark_factory(benchmark_config)
    except Exception as e:
        LOGGER.error(f"Error during benchmark allocation: {e}")
        backend.clean()
        raise e

    try:
        # Benchmark the backend
        benchmark.run(backend)
        backend.clean()
    except Exception as e:
        LOGGER.error("Error during benchmark execution: %s", e)
        backend.clean()
        raise e

    try:
        report = benchmark.report()
    except Exception as e:
        LOGGER.error("Error during report generation: %s", e)
        raise e

    return report


def launch(experiment_config: ExperimentConfig) -> Dict[str, Any]:
    launcher_config: LauncherConfig = experiment_config.launcher

    try:
        # Allocate requested launcher
        launcher_factory: Type[Launcher] = get_class(launcher_config._target_)
        launcher: Launcher = launcher_factory(launcher_config)
    except Exception as e:
        LOGGER.error(f"Error during launcher allocation: {e}")
        raise e

    backend_config: BackendConfig = experiment_config.backend
    benchmark_config: BenchmarkConfig = experiment_config.benchmark

    try:
        output = launcher.launch(run, benchmark_config, backend_config)
    except Exception as e:
        LOGGER.error(f"Error during experiment launching: {e}")
        raise e

    return output
