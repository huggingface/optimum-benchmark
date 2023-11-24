from abc import ABC
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, TypeVar

from psutil import cpu_count

LOGGER = getLogger("backend")


@dataclass
class BackendConfig(ABC):
    name: str
    version: str
    _target_: str

    # backend options
    seed: int = 42
    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    # device isolation options
    continuous_isolation: bool = True
    isolation_check_interval: Optional[float] = None

    # clean up options
    delete_cache: bool = False

    def __post_init__(self):
        if self.inter_op_num_threads is not None:
            if self.inter_op_num_threads == -1:
                self.inter_op_num_threads = cpu_count()

        if self.intra_op_num_threads is not None:
            if self.intra_op_num_threads == -1:
                self.intra_op_num_threads = cpu_count()

        if self.continuous_isolation and self.isolation_check_interval is None:
            self.isolation_check_interval = 1


BackendConfigT = TypeVar("BackendConfigT", bound=BackendConfig)
