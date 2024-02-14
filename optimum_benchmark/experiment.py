import os
from logging import getLogger
from tempfile import TemporaryDirectory
from dataclasses import dataclass, field
from typing import Any, Dict, Type, Optional, TYPE_CHECKING

from hydra.utils import get_class
from transformers.configuration_utils import PushToHubMixin

from .env_utils import get_system_info
from .import_utils import get_hf_libs_info
from .benchmarks.report import BenchmarkReport
from .benchmarks.config import BenchmarkConfig
from .launchers.config import LauncherConfig
from .backends.config import BackendConfig

if TYPE_CHECKING:
    # avoid importing any torch to be able to set
    # the CUDA_VISIBLE_DEVICES environment variable
    # in BackendConfig __post_init__
    from .benchmarks.base import Benchmark
    from .launchers.base import Launcher
    from .backends.base import Backend


LOGGER = getLogger("experiment")


@dataclass
class ExperimentConfig(PushToHubMixin):
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
    environment: Dict = field(default_factory=lambda: {**get_system_info(), **get_hf_libs_info()})


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
