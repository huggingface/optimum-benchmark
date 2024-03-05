from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import BackendConfig


@dataclass
class RyzenAIConfig(BackendConfig):
    name: str = "ryzenai"
    version: Optional[str] = None
    _target_: str = "optimum_benchmark.backends.ryzenai.backend.RyzenAIBackend"

    # optimum-benchmark options
    no_weights: bool = False

    # export/load options
    export: bool = True
    use_cache: bool = True

    # provider options
    provider: Optional[str] = None
    provider_options: Dict[str, Any] = field(default_factory=dict)

    # ryzenai config
    vaip_config: Optional[str] = None  # /usr/bin/vaip_config.json

    def __post_init__(self):
        super().__post_init__()

        if self.device not in ["ipu", "npu"]:
            raise ValueError(f"RyzenAIBackend only supports IPU/NPU device, got {self.device}")
