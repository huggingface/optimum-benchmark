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
        if not is_nvidia_system() and not is_rocm_system():
            raise ValueError("Device isolation is not supported on NVIDIA or ROCm systems")


LauncherConfigT = TypeVar("LauncherConfigT", bound=LauncherConfig)
