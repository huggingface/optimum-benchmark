from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, Optional

from ...import_utils import torch_version
from ..config import BackendConfig

AMP_DTYPES = ["bfloat16", "float16"]
TORCH_DTYPES = ["bfloat16", "float16", "float32", "auto"]

QUANTIZATION_CONFIGS = {"bnb": {"llm_int8_threshold": 0.0}}


LOGGER = getLogger(__name__)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = "pytorch"
    version: Optional[str] = torch_version()
    _target_: str = "optimum_benchmark.backends.pytorch.backend.PyTorchBackend"

    # load options
    no_weights: bool = False
    tp_plan: Optional[str] = None
    device_map: Optional[str] = None
    torch_dtype: Optional[str] = None

    # optimization options
    eval_mode: bool = True
    to_bettertransformer: bool = False
    low_cpu_mem_usage: Optional[bool] = None
    attn_implementation: Optional[str] = None
    cache_implementation: Optional[str] = None

    # tf32 options
    allow_tf32: bool = False

    # autocast options
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

        if self.model_kwargs.get("torch_dtype", None) is not None:
            raise ValueError(
                "`torch_dtype` is an explicit argument in the PyTorch backend config. "
                "Please remove it from the `model_kwargs` and set it in the backend config directly."
            )

        if self.torch_dtype is not None and self.torch_dtype not in TORCH_DTYPES:
            raise ValueError(f"`torch_dtype` must be one of {TORCH_DTYPES}. Got {self.torch_dtype} instead.")

        if self.autocast_dtype is not None and self.autocast_dtype not in AMP_DTYPES:
            raise ValueError(f"`autocast_dtype` must be one of {AMP_DTYPES}. Got {self.autocast_dtype} instead.")

        if self.quantization_scheme is not None:
            LOGGER.warning(
                "`backend.quantization_scheme` is deprecated and will be removed in a future version. "
                "Please use `quantization_config.quant_method` instead."
            )
            if self.quantization_config is None:
                self.quantization_config = {"quant_method": self.quantization_scheme}
            else:
                self.quantization_config["quant_method"] = self.quantization_scheme

        if self.quantization_config is not None:
            self.quantization_config = dict(
                QUANTIZATION_CONFIGS.get(self.quantization_scheme, {}),  # default config
                **self.quantization_config,  # user config (overwrites default)
            )
