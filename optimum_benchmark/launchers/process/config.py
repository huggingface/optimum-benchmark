from dataclasses import dataclass
from logging import getLogger

from ..base import LauncherConfig

LOGGER = getLogger("process")


@dataclass
class ProcessConfig(LauncherConfig):
    name: str = "process"
    _target_: str = "optimum_benchmark.launchers.process.launcher.ProcessLauncher"

    start_method: str = "spawn"

    def __post_init__(self) -> None:
        if self.start_method not in ["spawn", "fork"]:
            raise ValueError(f"start_method must be one of ['spawn', 'fork'], got {self.start_method}")
