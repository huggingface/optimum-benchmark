import os
from dataclasses import dataclass, field
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from .backends.config import BackendConfig
from .benchmarks.config import BenchmarkConfig
from .benchmarks.report import BenchmarkReport
from .hub_utils import PushToHubMixin
from .import_utils import get_hf_libs_info
from .launchers.config import LauncherConfig
from .system_utils import get_system_info

if TYPE_CHECKING:
    # avoid importing any torch to be able to set
    # the CUDA_VISIBLE_DEVICES environment variable
    # in BackendConfig __post_init__
    from .backends.base import Backend
    from .benchmarks.base import Benchmark
    from .launchers.base import Launcher

from hydra.utils import get_class

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

    @property
    def file_name(self) -> str:
        return "experiment_config.json"


def run(benchmark_config: BenchmarkConfig, backend_config: BackendConfig) -> BenchmarkReport:
    """
    Runs a benchmark using specified backend and benchmark configurations
    """

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
    """
    Runs an experiment using specified launcher configuration/logic
    """

    # fix backend until deprecated model and device are removed
    if experiment_config.task is not None:
        LOGGER.warning("`task` is deprecated in experiment config. Use `backend.task` instead.")
        experiment_config.backend.task = experiment_config.task
    if experiment_config.model is not None:
        LOGGER.warning("`model` is deprecated in experiment config. Use `backend.model` instead.")
        experiment_config.backend.model = experiment_config.model
    if experiment_config.device is not None:
        LOGGER.warning("`device` is deprecated in experiment config. Use `backend.device` instead.")
        experiment_config.backend.device = experiment_config.device
    if experiment_config.library is not None:
        LOGGER.warning("`library` is deprecated in experiment config. Use `backend.library` instead.")
        experiment_config.backend.library = experiment_config.library

    original_dir = os.getcwd()
    tmpdir = TemporaryDirectory()

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        # to not pollute the user's environment
        LOGGER.info("Launching experiment in a temporary directory.")
        os.chdir(tmpdir.name)

    launcher_config: LauncherConfig = experiment_config.launcher

    try:
        # Allocate requested launcher
        launcher_factory: Type[Launcher] = get_class(launcher_config._target_)
        launcher: Launcher = launcher_factory(launcher_config)
    except Exception as e:
        LOGGER.error(f"Error during launcher allocation: {e}")
        os.chdir(original_dir)
        tmpdir.cleanup()
        raise e

    backend_config: BackendConfig = experiment_config.backend
    benchmark_config: BenchmarkConfig = experiment_config.benchmark

    try:
        output = launcher.launch(run, benchmark_config, backend_config)
    except Exception as e:
        LOGGER.error(f"Error during experiment launching: {e}")
        os.chdir(original_dir)
        tmpdir.cleanup()
        raise e

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        os.chdir(original_dir)
        tmpdir.cleanup()

    return output
