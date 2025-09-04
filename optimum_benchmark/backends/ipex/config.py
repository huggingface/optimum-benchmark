from dataclasses import dataclass
from typing import Optional

from ...import_utils import ipex_version
from ..config import BackendConfig

TORCH_DTYPES = ["bfloat16", "float16", "float32", "auto"]


@dataclass
class IPEXConfig(BackendConfig):
    name: str = "ipex"
    version: Optional[str] = ipex_version()
    _target_: str = "optimum_benchmark.backends.ipex.backend.IPEXBackend"

    no_weights: bool = False

    # ipexmodel kwargs
    torch_dtype: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        self.device = self.device.lower()

        if self.device not in ["cpu", "gpu"]:
            raise ValueError(f"IPEXBackend only supports CPU devices, got {self.device}")

        if self.model_kwargs.get("torch_dtype", None) is not None:
            raise ValueError(
                "`torch_dtype` is an explicit argument in the PyTorch backend config. "
                "Please remove it from the `model_kwargs` and set it in the backend config directly."
            )

        if self.torch_dtype is not None and self.torch_dtype not in TORCH_DTYPES:
            raise ValueError(f"`torch_dtype` must be one of {TORCH_DTYPES}. Got {self.torch_dtype} instead.")
