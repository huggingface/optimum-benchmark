import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, Type

from hydra.utils import get_class

from .backends.base import Backend
from .backends.config import BackendConfig
from .hub_utils import PushToHubMixin, classproperty
from .import_utils import get_hf_libs_info
from .launchers.base import Launcher
from .launchers.config import LauncherConfig
from .report import BenchmarkReport
from .scenarios.base import Scenario
from .scenarios.config import ScenarioConfig
from .system_utils import get_system_info

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

    def __post_init__(self):
        if os.environ.get("BENCHMARK_INTERFACE", "API") == "CLI":
            LOGGER.warning(
                "The `experiment` parent schema is deprecated and will be removed soon. "
                "Please use `benchmark` parent schema instead. "
                "You'll also need to change the `experiment_name` field to `name` "
                "and `benchmark` schema to `scenario` schema. "
                "See the repository README for more information."
            )
        else:
            LOGGER.warning(
                "The `ExperimentConfig` class and `launch` function are deprecated and will be removed soon. "
                "Please use `BenchmarkConfig` class with the `Benchmark` class and its `launch` method instead."
                "See the repository README for more information."
            )

    @classproperty
    def default_filename(cls) -> str:
        return "experiment_config.json"


def run(experiment_config: ExperimentConfig) -> BenchmarkReport:
    """
    Runs a benchmark using specified backend and benchmark configurations
    """

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        LOGGER.warning(
            "The `run` function is deprecated and will be removed soon. "
            "Please use the `Benchmark` class and its `run` method instead."
        )

    # Allocate requested backend
    backend_config: BackendConfig = experiment_config.backend
    backend_factory: Type[Backend] = get_class(backend_config._target_)
    backend: Backend = backend_factory(backend_config)

    # Allocate requested benchmark
    benchmark_config: ScenarioConfig = experiment_config.benchmark
    benchmark_factory: Type[Scenario] = get_class(benchmark_config._target_)
    benchmark: Scenario = benchmark_factory(benchmark_config)

    try:
        # Run the benchmark using the backend
        report = benchmark.run(backend)
    except Exception as error:
        LOGGER.error("Error during benchmark execution", exc_info=True)
        backend.cleanup()
        raise error
    else:
        backend.cleanup()

    return report


def launch(experiment_config: ExperimentConfig) -> BenchmarkReport:
    """
    Runs an experiment using specified launcher configuration/logic
    """

    if os.environ.get("BENCHMARK_INTERFACE", "API") == "API":
        LOGGER.warning(
            "The `launch` function is deprecated and will be removed soon. "
            "Please use the `Benchmark` class and its `launch` method instead."
        )

    try:
        # Allocate requested launcher
        launcher_config: LauncherConfig = experiment_config.launcher
        launcher_factory: Type[Launcher] = get_class(launcher_config._target_)
        launcher: Launcher = launcher_factory(launcher_config)
        # Launch the experiment
        report = launcher.launch(run, experiment_config)
    except Exception as exception:
        LOGGER.error("Error during experiment", exc_info=True)
        raise exception

    return report
