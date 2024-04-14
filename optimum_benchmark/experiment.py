import os
from dataclasses import dataclass, field
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, Type

from .backends.config import BackendConfig
from .benchmarks.config import BenchmarkConfig
from .benchmarks.report import BenchmarkReport
from .hub_utils import PushToHubMixin, classproperty
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
    # Experiment name
    experiment_name: str

    # BACKEND CONFIGURATION
    backend: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # LAUNCHER CONFIGURATION
    launcher: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # BENCHMARK CONFIGURATION
    benchmark: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # ENVIRONMENT CONFIGURATION
    environment: Dict = field(default_factory=lambda: {**get_system_info(), **get_hf_libs_info()})

    @classproperty
    def default_filename(cls) -> str:
        return "experiment_config.json"


def run(benchmark_config: BenchmarkConfig, backend_config: BackendConfig) -> BenchmarkReport:
    """
    Runs a benchmark using specified backend and benchmark configurations
    """

    # Allocate requested backend
    backend_factory: Type[Backend] = get_class(backend_config._target_)
    backend: Backend = backend_factory(backend_config)

    # Allocate requested benchmark
    benchmark_factory: Type[Benchmark] = get_class(benchmark_config._target_)
    benchmark: Benchmark = benchmark_factory(benchmark_config)

    # Benchmark the backend
    benchmark.run(backend)
    report = benchmark.get_report()

    return report


def launch(experiment_config: ExperimentConfig) -> BenchmarkReport:
    """
    Runs an experiment using specified launcher configuration/logic
    """

    # We keep track of the main benchmark process PID to be able to
    # track its memory usage in isolated and distributed setups
    os.environ["BENCHMARK_PID"] = str(os.getpid())

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        # We launch the experiment in a temporary directory to avoid
        # polluting the current working directory with temporary files
        LOGGER.info("Launching experiment in a temporary directory.")
        tmpdir = TemporaryDirectory()
        original_dir = os.getcwd()
        os.chdir(tmpdir.name)

    try:
        # Allocate requested launcher
        launcher_config: LauncherConfig = experiment_config.launcher
        launcher_factory: Type[Launcher] = get_class(launcher_config._target_)
        launcher: Launcher = launcher_factory(experiment_config.launcher)
        report = launcher.launch(run, experiment_config.benchmark, experiment_config.backend)
        error = None
    except Exception as e:
        LOGGER.error("Error during experiment")
        error = e

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        LOGGER.info("Cleaning up experiment temporary directory.")
        os.chdir(original_dir)
        tmpdir.cleanup()

    if error is not None:
        raise error

    return report
