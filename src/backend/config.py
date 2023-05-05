from dataclasses import dataclass, MISSING
from logging import getLogger
from typing import Optional
from abc import ABC


LOGGER = getLogger('backend')


@dataclass
class BackendConfig(ABC):
    name: str = MISSING
    version: str = MISSING

    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None