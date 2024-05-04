from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, List, Optional, Type

from hydra.utils import get_class

from . import BackendConfig
from .config import BenchmarkConfig
from .hub_utils import PushToHubMixin
from .launchers import LauncherConfig
from .logging_utils import get_logs
from .report import BenchmarkReport
from .scenarios import ScenarioConfig

if TYPE_CHECKING:
    # avoid importing any torch to be able to set
    # the CUDA_VISIBLE_DEVICES environment variable
    # in BackendConfig __post_init__
    from .backends.base import Backend
    from .launchers.base import Launcher
    from .scenarios.base import Scenario


LOGGER = getLogger("benchmark")


@dataclass
class Benchmark(PushToHubMixin):
    config: BenchmarkConfig
    report: Optional[BenchmarkReport] = None

    colored_logs: Optional[List[str]] = None
    logs: Optional[List[str]] = None

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = BenchmarkConfig(**self.config)
        elif not isinstance(self.config, BenchmarkConfig):
            raise ValueError("config must be either a dict or a BenchmarkConfig instance")

    def run(self) -> BenchmarkReport:
        """
        Runs a scenario using specified backend configuration/logic
        """

        # Allocate requested backend
        backend_config: BackendConfig = self.config.backend
        backend_factory: Type[Backend] = get_class(backend_config._target_)
        backend: Backend = backend_factory(backend_config)

        # Allocate requested scenario
        scenario_config: ScenarioConfig = self.config.scenario
        scenario_factory: Type[Scenario] = get_class(scenario_config._target_)
        scenario: Scenario = scenario_factory(scenario_config)

        try:
            # Run the scenario using the backend
            self.report = scenario.run(backend)
        except Exception as error:
            LOGGER.error("Error during scenario execution", exc_info=True)
            backend.cleanup()
            raise error
        else:
            backend.cleanup()

        self.colored_logs = get_logs(colored=True)
        self.logs = get_logs(colored=False)

        return self.report

    def launch(self):
        """
        Runs an benchmark using specified launcher configuration/logic
        """

        # Allocate requested launcher
        launcher_config: LauncherConfig = self.config.launcher
        launcher_factory: Type[Launcher] = get_class(launcher_config._target_)
        launcher: Launcher = launcher_factory(launcher_config)

        try:
            # Launch the benchmark using the launcher
            self.report = launcher.launch(self.run)
        except Exception as exception:
            LOGGER.error("Error during benchmark launch", exc_info=True)
            raise exception

        self.colored_logs = get_logs(colored=True)
        self.logs = get_logs(colored=False)

        return self.report

    def push_to_hub(self):
        """
        Pushes the benchmark to the Hugging Face Hub
        """

        self.config.push_to_hub = True

        if self.config.push_to_hub_kwargs.get("repo_id") is None:
            raise ValueError("repo_id must be specified in config.push_to_hub_kwargs")

        return super().push_to_hub(**self.config.push_to_hub_kwargs)
