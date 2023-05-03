from dataclasses import dataclass
from typing import Optional, Set
from abc import abstractmethod
from logging import getLogger

from omegaconf import MISSING

LOGGER = getLogger('backend')


@dataclass
class BackendConfig:
    name: str = MISSING
    version: str = MISSING

    device: str = 'cpu'
    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    @staticmethod
    @abstractmethod
    def version():
        raise NotImplementedError()

    @staticmethod
    def supported_keys() -> Set[str]:
        return {'name', 'version', 'optimization', 'device', 'inter_op_num_threads', 'intra_op_num_threads'}
