from dataclasses import dataclass

from ..config import LauncherConfig


@dataclass
class MPIrunConfig(LauncherConfig):
    name: str = "mpirun"
    _target_: str = "optimum_benchmark.launchers.mpirun.launcher.MPIrunLauncher"

    num_processes: int = 1

    def __post_init__(self):
        super().__post_init__()
