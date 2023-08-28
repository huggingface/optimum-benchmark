from abc import ABC
from dataclasses import dataclass
from typing import Optional, TypeVar

from psutil import cpu_count


@dataclass
class BackendConfig(ABC):
    name: str
    version: str
    _target_: str

    # backend options
    seed: int = 42
    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    # isolation options
    initial_isolation_check: bool = True
    continous_isolation_check: bool = True

    # clean up options
    delete_cache: bool = False

    def __post_init__(self):
        if self.inter_op_num_threads is not None:
            if self.inter_op_num_threads == -1:
                self.inter_op_num_threads = cpu_count()

        if self.intra_op_num_threads is not None:
            if self.intra_op_num_threads == -1:
                self.intra_op_num_threads = cpu_count()


BackendConfigT = TypeVar("BackendConfigT", bound=BackendConfig)
