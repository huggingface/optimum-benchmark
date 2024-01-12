from abc import ABC
from dataclasses import dataclass
from logging import getLogger
from typing import Callable, ClassVar, Generic, TypeVar

LOGGER = getLogger("launcher")


@dataclass
class LauncherConfig(ABC):
    name: str
    _target_: str

    device_isolation: bool = False


LauncherConfigT = TypeVar("LauncherConfigT", bound=LauncherConfig)


class Launcher(Generic[LauncherConfigT], ABC):
    NAME: ClassVar[str]

    config: LauncherConfigT

    def __init__(self) -> None:
        pass

    def configure(self, config: LauncherConfigT) -> None:
        LOGGER.info(f"Configuring {self.NAME} launcher")
        self.config = config

    def launch(self, worker: Callable, *worker_args):
        raise NotImplementedError("Launcher must implement launch method")
