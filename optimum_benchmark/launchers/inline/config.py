from dataclasses import dataclass
from logging import getLogger

from ..config import LauncherConfig

LOGGER = getLogger("inline")


@dataclass
class InlineConfig(LauncherConfig):
    name: str = "inline"
    _target_: str = "optimum_benchmark.launchers.inline.launcher.InlineLauncher"
