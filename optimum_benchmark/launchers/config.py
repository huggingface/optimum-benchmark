from abc import ABC
from dataclasses import dataclass
from logging import getLogger
from typing import TypeVar

from ..system_utils import is_nvidia_system, is_rocm_system

LOGGER = getLogger("launcher")


@dataclass
class LauncherConfig(ABC):
    name: str
    _target_: str

    device_isolation: bool = False

    def __post_init__(self):
        if self.device_isolation and not is_nvidia_system() and not is_rocm_system():
            raise ValueError(
                "Device isolation is only supported on NVIDIA and ROCm systems. "
                "Please set `device_isolation` to False or make sure your drivers "
                "are correctly installed by running `nvidia-smi` or `rocm-smi`."
            )


LauncherConfigT = TypeVar("LauncherConfigT", bound=LauncherConfig)
