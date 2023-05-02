from dataclasses import dataclass
from typing import Optional, Set
from abc import abstractmethod
from logging import getLogger

from hydra.types import TargetConf
from omegaconf import MISSING

LOGGER = getLogger('backends')


@dataclass
class BackendConfig(TargetConf):
    name: str = MISSING
    version: str = MISSING

    device: str = 'cpu'
    inference: bool = True
    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    @staticmethod
    @abstractmethod
    def version():
        raise NotImplementedError()

    @staticmethod
    def supported_keys() -> Set[str]:
        return {'name', 'version', 'device', 'inference', 'inter_op_num_threads', 'intra_op_num_threads'}
