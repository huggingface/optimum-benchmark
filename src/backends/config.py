from dataclasses import dataclass
from abc import abstractmethod
from logging import getLogger
from typing import Set

from hydra.types import TargetConf
from omegaconf import MISSING

LOGGER = getLogger('backend')


@dataclass
class BackendConfig(TargetConf):
    name: str = MISSING
    version: str = MISSING

    @staticmethod
    @abstractmethod
    def version():
        raise NotImplementedError()

    @staticmethod
    def supported_keys() -> Set[str]:
        return {"name", "version"}
