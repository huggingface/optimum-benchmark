from abc import ABC
from dataclasses import dataclass
from logging import getLogger
from typing import TypeVar

LOGGER = getLogger("launcher")


@dataclass
class LauncherConfig(ABC):
    name: str
    _target_: str

    device_isolation: bool = False

    def __post_init__(self):
        pass


LauncherConfigT = TypeVar("LauncherConfigT", bound=LauncherConfig)
