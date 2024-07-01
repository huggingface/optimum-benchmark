from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...import_utils import torch_ort_version
from ..config import BackendConfig


@dataclass
class TorchORTConfig(BackendConfig):
    name: str = "torch-ort"
    version: Optional[str] = torch_ort_version()
    _target_: str = "optimum_benchmark.backends.torch_ort.backend.TorchORTBackend"

    # load options
    no_weights: bool = False
    torch_dtype: Optional[str] = None
    # sdpa, which has became default of many architectures, fails with torch ort
    attn_implementation: Optional[str] = "eager"

    # peft options
    peft_type: Optional[str] = None
    peft_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if self.device != "cuda":
            raise ValueError(f"TorchORTBackend only supports CUDA devices, got {self.device}")
