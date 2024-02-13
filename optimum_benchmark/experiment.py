import os
import platform
from logging import getLogger
from tempfile import TemporaryDirectory
from dataclasses import dataclass, field
from typing import Any, Dict, Type, Optional, TYPE_CHECKING

from hydra.utils import get_class

from .benchmarks.report import BenchmarkReport
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
    is_nvidia_system,
    is_rocm_system,
    get_gpu_vram_mb,
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
            "cpu_count": os.cpu_count(),
            "cpu_ram_mb": get_cpu_ram_mb(),
            "system": platform.system(),
            "python_version": platform.python_version(),
            # libraries
            "transformers_version": transformers_version(),
            "transformers_commit": get_git_revision_hash("transformers"),
            "accelerate_version": accelerate_version(),
            "accelerate_commit": get_git_revision_hash("accelerate"),
            "diffusers_version": diffusers_version(),
            "diffusers_commit": get_git_revision_hash("diffusers"),
            "optimum_version": optimum_version(),
            "optimum_commit": get_git_revision_hash("optimum"),
            "timm_version": timm_version(),
            "timm_commit": get_git_revision_hash("timm"),
            "peft_version": peft_version(),
            "peft_commit": get_git_revision_hash("peft"),
        }
    )

    def __post_init__(self):
        # adding GPU information to the environment
        if is_nvidia_system() or is_rocm_system():
            available_gpus = get_gpus()
            if len(available_gpus) > 0:
                self.environment["gpu"] = available_gpus[0]
                self.environment["gpu_count"] = len(available_gpus)
                self.environment["gpu_vram_mb"] = get_gpu_vram_mb()
            else:
                LOGGER.warning("Detected NVIDIA or ROCm system, but no GPUs found.")


def run(benchmark_config: BenchmarkConfig, backend_config: BackendConfig) -> BenchmarkReport:
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
        report = benchmark.get_report()
    except Exception as e:
        LOGGER.error("Error during report generation: %s", e)
        raise e

    return report


def launch(experiment_config: ExperimentConfig) -> BenchmarkReport:
    if os.environ.get("BENCHMARK_CLI", "0") == "0":
        LOGGER.info("Launching experiment in a temporary directory.")
        tmep_dir = TemporaryDirectory()
        original_dir = os.getcwd()
        os.chdir(tmep_dir.name)

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

    if os.environ.get("BENCHMARK_CLI", "0") == "0":
        os.chdir(original_dir)
        tmep_dir.cleanup()

    return output
