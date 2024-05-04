from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...import_utils import torch_version
from ...system_utils import is_rocm_system
from ..config import BackendConfig

DEVICE_MAPS = ["auto", "sequential"]
AMP_DTYPES = ["bfloat16", "float16"]
TORCH_DTYPES = ["bfloat16", "float16", "float32", "auto"]

QUANTIZATION_CONFIGS = {"bnb": {"llm_int8_threshold": 0.0}, "gptq": {}, "awq": {}}


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = "pytorch"
    version: Optional[str] = torch_version()
    _target_: str = "optimum_benchmark.backends.pytorch.backend.PyTorchBackend"

    # load options
    no_weights: bool = False
    device_map: Optional[str] = None
    torch_dtype: Optional[str] = None

    # optimization options
    eval_mode: bool = True
    to_bettertransformer: bool = False
    low_cpu_mem_usage: Optional[bool] = None
    attn_implementation: Optional[str] = None
    cache_implementation: Optional[str] = None

    # automatic mixed precision options
    autocast_enabled: bool = False
    autocast_dtype: Optional[str] = None

    # torch compile options
    torch_compile: bool = False
    torch_compile_target: str = "forward"
    torch_compile_config: Dict[str, Any] = field(default_factory=dict)

    # quantization options
    quantization_scheme: Optional[str] = None
    quantization_config: Dict[str, Any] = field(default_factory=dict)

    # distributed inference options
    deepspeed_inference: bool = False
    deepspeed_inference_config: Dict[str, Any] = field(default_factory=dict)

    # peft options
    peft_type: Optional[str] = None
    peft_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if self.device_map is not None and self.device_map not in DEVICE_MAPS:
            raise ValueError(f"`device_map` must be one of {DEVICE_MAPS}. Got {self.device_map} instead.")

        if self.torch_dtype is not None and self.torch_dtype not in TORCH_DTYPES:
            raise ValueError(f"`torch_dtype` must be one of {TORCH_DTYPES}. Got {self.torch_dtype} instead.")

        if self.autocast_dtype is not None and self.autocast_dtype not in AMP_DTYPES:
            raise ValueError(f"`autocast_dtype` must be one of {AMP_DTYPES}. Got {self.autocast_dtype} instead.")

        if self.quantization_scheme is not None:
            if self.quantization_scheme not in QUANTIZATION_CONFIGS:
                raise ValueError(
                    f"`quantization_scheme` must be one of {list(QUANTIZATION_CONFIGS.keys())}. "
                    f"Got {self.quantization_scheme} instead."
                )

            if self.quantization_scheme == "bnb" and is_rocm_system():
                raise ValueError("BitsAndBytes is not supported on ROCm GPUs. Please disable it.")

            if self.quantization_config:
                QUANTIZATION_CONFIG = QUANTIZATION_CONFIGS[self.quantization_scheme]
                self.quantization_config = {**QUANTIZATION_CONFIG, **self.quantization_config}
