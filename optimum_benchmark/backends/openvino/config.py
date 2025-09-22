from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...import_utils import openvino_version
from ..config import BackendConfig

TORCH_DTYPES = ["bfloat16", "float16", "float32", "auto"]


@dataclass
class OpenVINOConfig(BackendConfig):
    name: str = "openvino"
    version: Optional[str] = openvino_version()
    _target_: str = "optimum_benchmark.backends.openvino.backend.OpenVINOBackend"

    no_weights: bool = False

    # ovmodel kwargs
    export: Optional[bool] = None
    use_cache: Optional[bool] = None
    use_merged: Optional[bool] = None
    torch_dtype: Optional[str] = None
    load_in_8bit: Optional[bool] = None
    load_in_4bit: Optional[bool] = None
    ov_config: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None

    # compilation options
    half: bool = False
    compile: bool = False
    reshape: bool = False
    reshape_kwargs: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self.device = self.device.lower()
        if self.device not in ["cpu", "gpu"]:
            raise ValueError(f"OpenVINOBackend only supports CPU devices, got {self.device}")

        if self.model_kwargs.get("torch_dtype", None) is not None:
            raise ValueError(
                "`torch_dtype` is an explicit argument in the OpenVINO backend config. "
                "Please remove it from the `model_kwargs` and set it in the backend config directly."
            )

        if self.torch_dtype is not None and self.torch_dtype not in TORCH_DTYPES:
            raise ValueError(f"torch_dtype should be one of None or {TORCH_DTYPES}, got {self.torch_dtype}")

        if self.intra_op_num_threads is not None:
            raise NotImplementedError("OpenVINOBackend does not support intra_op_num_threads. Please use the ov_config")

        if self.inter_op_num_threads is not None:
            raise NotImplementedError("OpenVINOBackend does not support inter_op_num_threads. Please use the ov_config")
