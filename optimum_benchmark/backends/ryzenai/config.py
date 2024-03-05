from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import BackendConfig
from .utils import TASKS_TO_RYZENAIMODEL


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

    # session options
    session_options: Dict[str, Any] = field(default_factory=dict)

    # provider options
    provider: Optional[str] = None
    provider_options: Dict[str, Any] = field(default_factory=dict)

    # ryzenai config
    vaip_config: Optional[str] = None  # /usr/bin/vaip_config.json

    # auto quantization options
    auto_quantization: Optional[str] = None  # ipu_cnn_config, cpu_cnn_config
    auto_quantization_config: Dict[str, Any] = field(default_factory=dict)

    # manual quantization options
    quantization: bool = False
    quantization_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if self.device not in ["cpu", "ipu", "npu"]:
            raise ValueError(f"RyzenAIBackend only supports CPU & IPU/NPU devices, got {self.device}")

        if self.task not in TASKS_TO_RYZENAIMODEL:
            raise NotImplementedError(f"RyzenAIBackend does not support task {self.task}")
