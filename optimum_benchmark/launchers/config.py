from abc import ABC
from typing import TypeVar
from logging import getLogger
from dataclasses import dataclass

LOGGER = getLogger("launcher")


@dataclass
class LauncherConfig(ABC):
    name: str
    _target_: str

    device_isolation: bool = False

    def __post_init__(self):
        pass


LauncherConfigT = TypeVar("LauncherConfigT", bound=LauncherConfig)
