import os
from dataclasses import dataclass, field
from logging import getLogger
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, Type

from .backends.config import BackendConfig
from .hub_utils import PushToHubMixin, classproperty
from .import_utils import get_hf_libs_info
from .launchers.config import LauncherConfig
from .report import BenchmarkReport
from .scenarios.config import ScenarioConfig
from .system_utils import get_system_info

if TYPE_CHECKING:
    # avoid importing any torch to be able to set
    # the CUDA_VISIBLE_DEVICES environment variable
    # in BackendConfig __post_init__
    from .backends.base import Backend
    from .launchers.base import Launcher
    from .scenarios.base import Scenario

from hydra.utils import get_class

LOGGER = getLogger("benchmark")


@dataclass
class BenchmarkConfig(PushToHubMixin):
    # Benchmark name
    benchmark_name: str

    # BACKEND CONFIGURATION
    backend: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # SCENARIO CONFIGURATION
    scenario: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386
    # LAUNCHER CONFIGURATION
    launcher: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # ENVIRONMENT CONFIGURATION
    environment: Dict = field(default_factory=lambda: {**get_system_info(), **get_hf_libs_info()})

    @classproperty
    def default_filename(cls) -> str:
        return "benchmark_config.json"


def run(benchmark_config: BenchmarkConfig) -> BenchmarkReport:
    """
    Runs a benchmark using specified backend and benchmark configurations
    """

    # Allocate requested backend
    backend_config: BackendConfig = benchmark_config.backend
    backend_factory: Type[Backend] = get_class(backend_config._target_)
    backend: Backend = backend_factory(backend_config)

    # Allocate requested scenario
    scenario_config: ScenarioConfig = benchmark_config.scenario
    scenario_factory: Type[Scenario] = get_class(scenario_config._target_)
    scenario: Scenario = scenario_factory(scenario_config)

    # Benchmark the backend
    try:
        report = scenario.run(backend)
    except Exception as error:
        LOGGER.error("Error during scenario execution", exc_info=True)
        backend.cleanup()
        raise error
    else:
        backend.cleanup()

    return report


def launch(benchmark_config: BenchmarkConfig) -> BenchmarkReport:
    """
    Runs an benchmark using specified launcher configuration/logic
    """

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        # We launch the benchmark in a temporary directory to avoid
        # polluting the current working directory with temporary files
        LOGGER.info("Launching benchmark in a temporary directory.")
        original_dir = os.getcwd()
        tmpdir = TemporaryDirectory()
        os.chdir(tmpdir.name)

    # Allocate requested launcher
    launcher_config: LauncherConfig = benchmark_config.launcher
    launcher_factory: Type[Launcher] = get_class(launcher_config._target_)
    launcher: Launcher = launcher_factory(launcher_config)

    try:
        report = launcher.launch(run, benchmark_config)
    except Exception as error:
        LOGGER.error("Error during benchmark launch", exc_info=True)
        exception = error
    else:
        exception = None

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        LOGGER.info("Cleaning up benchmark temporary directory.")
        os.chdir(original_dir)
        tmpdir.cleanup()

    if exception is not None:
        raise exception

    return report
