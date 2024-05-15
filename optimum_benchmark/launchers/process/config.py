from dataclasses import dataclass

from ..config import LauncherConfig


@dataclass
class ProcessConfig(LauncherConfig):
    name: str = "process"
    _target_: str = "optimum_benchmark.launchers.process.launcher.ProcessLauncher"

    start_method: str = "spawn"

    def __post_init__(self):
        super().__post_init__()

        if self.start_method not in ["spawn", "fork"]:
            raise ValueError(f"start_method must be one of ['spawn', 'fork'], got {self.start_method}")
