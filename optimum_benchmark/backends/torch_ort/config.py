from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import BackendConfig
from ...import_utils import torch_ort_version
from ..peft_utils import PEFT_CONFIGS, PEFT_TASKS_TYPES


@dataclass
class TorchORTConfig(BackendConfig):
    name: str = "torch-ort"
    version: Optional[str] = torch_ort_version
    _target_: str = "optimum_benchmark.backends.torch_ort.backend.TorchORTBackend"

    # load options
    no_weights: bool = False
    torch_dtype: Optional[str] = None

    # peft options
    peft_strategy: Optional[str] = None
    peft_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if self.device != "cuda":
            raise ValueError(f"TorchORTBackend only supports CUDA devices, got {self.device}")

        if self.peft_strategy is not None:
            if self.peft_strategy not in PEFT_CONFIGS:
                raise ValueError(f"`peft_strategy` must be one of {list(PEFT_CONFIGS.keys())}. Got {self.peft_strategy} instead.")
            PEFT_CONFIG = PEFT_CONFIGS[self.peft_strategy]
            self.peft_config = {**PEFT_CONFIG, **self.peft_config}

            if self.peft_config["task_type"] is None:
                raise ValueError(f"`peft_config.task_type` must be set to one of the following {PEFT_TASKS_TYPES}")
