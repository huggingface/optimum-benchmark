from dataclasses import dataclass
from logging import getLogger

from ..config import LauncherConfig

LOGGER = getLogger("inline")


@dataclass
class InlineConfig(LauncherConfig):
    name: str = "inline"
    _target_: str = "optimum_benchmark.launchers.inline.launcher.InlineLauncher"

    def __post_init__(self):
        super().__post_init__()

        if self.device_isolation:
            raise ValueError("Device isolation is not supported with the inline launcher.")

        if self.device_isolation_action is not None:
            raise ValueError("Device isolation is not supported with the inline launcher.")
