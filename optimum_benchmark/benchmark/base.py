from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

from hydra.utils import get_class

from ..backends.config import BackendConfig
from ..hub_utils import PushToHubMixin, classproperty
from ..launchers import LauncherConfig
from ..scenarios import ScenarioConfig
from .config import BenchmarkConfig
from .report import BenchmarkReport

if TYPE_CHECKING:
    from ..backends.base import Backend
    from ..launchers.base import Launcher
    from ..scenarios.base import Scenario


@dataclass
class Benchmark(PushToHubMixin):
    config: BenchmarkConfig
    report: BenchmarkReport

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = BenchmarkConfig.from_dict(self.config)
        elif not isinstance(self.config, BenchmarkConfig):
            raise ValueError("config must be either a dict or a BenchmarkConfig instance")

        if isinstance(self.report, dict):
            self.report = BenchmarkReport.from_dict(self.report)
        elif not isinstance(self.report, BenchmarkReport):
            raise ValueError("report must be either a dict or a BenchmarkReport instance")

    @classmethod
    def launch(cls, config: BenchmarkConfig):
        """
        Runs an benchmark using specified launcher configuration/logic
        """

        # Allocate requested launcher
        launcher_config: LauncherConfig = config.launcher
        launcher_factory: Type[Launcher] = get_class(launcher_config._target_)
        launcher: Launcher = launcher_factory(launcher_config)

        # Launch the benchmark using the launcher
        report = launcher.launch(worker=cls.run, worker_args=[config])

        return report

    @classmethod
    def run(cls, config: BenchmarkConfig):
        """
        Runs a scenario using specified backend configuration/logic
        """

        # Allocate requested backend
        backend_config: BackendConfig = config.backend
        backend_factory: Type[Backend] = get_class(backend_config._target_)
        backend: Backend = backend_factory(backend_config)

        # Allocate requested scenario
        scenario_config: ScenarioConfig = config.scenario
        scenario_factory: Type[Scenario] = get_class(scenario_config._target_)
        scenario: Scenario = scenario_factory(scenario_config)

        # Run the scenario using the backend
        report = scenario.run(backend)

        return report

    @classproperty
    def default_filename(cls) -> str:
        return "benchmark.json"
