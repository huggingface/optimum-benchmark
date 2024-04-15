from abc import ABC
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, TypeVar

from ..system_utils import is_nvidia_system, is_rocm_system

LOGGER = getLogger("launcher")


@dataclass
class LauncherConfig(ABC):
    name: str
    _target_: str

    device_isolation: bool = False
    device_isolation_action: Optional[str] = None

    def __post_init__(self):
        if self.device_isolation and not is_nvidia_system() and not is_rocm_system():
            raise ValueError(
                "Device isolation is only supported on NVIDIA and ROCm systems. "
                "Please set `device_isolation` to False or make sure your drivers "
                "are correctly installed by running `nvidia-smi` or `rocm-smi`."
            )

        if self.device_isolation and self.device_isolation_action is None:
            LOGGER.warning(
                "Device isolation is enabled but no action is specified. "
                "Please set `device_isolation_action` to 'kill' or 'alert' to specify the action."
                "Defaulting to 'kill' for now."
            )
            self.device_isolation_action = "kill"

        assert self.device_isolation_action in [
            "kill",
            "alert",
        ], "Device isolation action must be one of 'kill' or 'alert'."


LauncherConfigT = TypeVar("LauncherConfigT", bound=LauncherConfig)
