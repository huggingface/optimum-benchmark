from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...import_utils import ipex_version
from ..config import BackendConfig


@dataclass
class IPEXConfig(BackendConfig):
    name: str = "ipex"
    version: Optional[str] = ipex_version()
    _target_: str = "optimum_benchmark.backends.ipex.backend.IPEXBackend"

    # load options
    no_weights: bool = False

    # export options
    export: bool = True

    def __post_init__(self):
        super().__post_init__()

        self.device = self.device.lower()
        if self.device not in ["cpu", "gpu"]:
            raise ValueError(f"IPEXBackend only supports CPU devices, got {self.device}")
