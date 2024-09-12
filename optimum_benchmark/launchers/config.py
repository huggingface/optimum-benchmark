from abc import ABC
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, Optional, TypeVar

from ..system_utils import is_nvidia_system, is_rocm_system

LOGGER = getLogger("launcher")


@dataclass
class LauncherConfig(ABC):
    name: str
    _target_: str

    device_isolation: bool = False
    device_isolation_action: Optional[str] = None

    numactl: bool = False
    numactl_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.device_isolation and not is_nvidia_system() and not is_rocm_system():
            raise ValueError(
                "Device isolation is only supported on NVIDIA and ROCm systems. "
                "Please set `device_isolation` to False or make sure your drivers "
                "are correctly installed by running `nvidia-smi` or `rocm-smi`."
            )

        if self.device_isolation and self.device_isolation_action is None:
            LOGGER.warning(
                "Device isolation is enabled but no action is specified. "
                "Please set `device_isolation_action` to either `error`, `warn`, or `kill`. "
                "Defaulting to `warn`."
            )
            self.device_isolation_action = "warn"

        elif self.device_isolation and self.device_isolation_action not in {"error", "warn", "kill"}:
            raise ValueError(
                f"Unsupported device isolation action {self.device_isolation_action}. "
                "Please set `device_isolation_action` to either `error`, `warn`, or `kill`."
            )


LauncherConfigT = TypeVar("LauncherConfigT", bound=LauncherConfig)
