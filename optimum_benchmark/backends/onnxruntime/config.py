from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...import_utils import onnxruntime_version
from ...task_utils import TEXT_GENERATION_TASKS
from ..config import BackendConfig

QUANTIZATION_CONFIG = {
    "is_static": False,
    "format": "QOperator",
    # is_static and format are mandatory
}

CALIBRATION_CONFIG = {
    "method": "MinMax"
    # method is mandatory
}

AUTO_QUANTIZATION_CONFIG = {
    "is_static": False
    # is_static is mandatory
}

IO_BINDING_LIBRARIES = ["transformers", "timm"]
IO_BINDING_PROVIDERS = ["CPUExecutionProvider", "CUDAExecutionProvider"]
DEVICE_PROVIDER_MAP = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}


@dataclass
class ORTConfig(BackendConfig):
    name: str = "onnxruntime"
    version: Optional[str] = onnxruntime_version()
    _target_: str = "optimum_benchmark.backends.onnxruntime.backend.ORTBackend"

    # load options
    no_weights: bool = False

    # export options
    export: bool = True
    use_cache: bool = True
    use_merged: bool = False
    torch_dtype: Optional[str] = None

    # provider options
    provider: Optional[str] = None
    provider_options: Dict[str, Any] = field(default_factory=dict)

    # inference options
    use_io_binding: Optional[bool] = None
    session_options: Dict[str, Any] = field(default_factory=dict)

    # null, O1, O2, O3, O4
    auto_optimization: Optional[str] = None
    auto_optimization_config: Dict[str, Any] = field(default_factory=dict)

    # null, arm64, avx2, avx512, avx512_vnni, tensorrt
    auto_quantization: Optional[str] = None
    auto_quantization_config: Dict[str, Any] = field(default_factory=dict)

    # minmax, entropy, l2norm, percentiles
    auto_calibration: Optional[str] = None
    auto_calibration_config: Dict[str, Any] = field(default_factory=dict)

    # manual optimization options
    optimization: bool = False
    optimization_config: Dict[str, Any] = field(default_factory=dict)

    # manual quantization options
    quantization: bool = False
    quantization_config: Dict[str, Any] = field(default_factory=dict)

    # manual calibration options
    calibration: bool = False
    calibration_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"ORTBackend only supports CPU and CUDA devices, got {self.device}")

        if not self.no_weights and not self.export and self.torch_dtype is not None:
            raise NotImplementedError("Can't convert an exported model's weights to a different dtype.")

        if self.provider is None:
            self.provider = DEVICE_PROVIDER_MAP[self.device]

        if self.use_io_binding is None:
            self.use_io_binding = self.provider in IO_BINDING_PROVIDERS and self.library in IO_BINDING_LIBRARIES

        if self.provider == "TensorrtExecutionProvider" and self.task in TEXT_GENERATION_TASKS:
            raise NotImplementedError("we don't support TensorRT for text generation tasks")

        if self.quantization:
            self.quantization_config = {**QUANTIZATION_CONFIG, **self.quantization_config}
            # raise ValueError if the quantization is static but calibration is not enabled
            if self.quantization_config["is_static"] and self.auto_calibration is None and not self.calibration:
                raise ValueError(
                    "Quantization is static but calibration is not enabled. "
                    "Please enable calibration or disable static quantization."
                )

        if self.auto_quantization is not None:
            self.auto_quantization_config = {**AUTO_QUANTIZATION_CONFIG, **self.auto_quantization_config}
            if self.auto_quantization_config["is_static"] and self.auto_calibration is None and not self.calibration:
                raise ValueError(
                    "Quantization is static but calibration is not enabled. "
                    "Please enable calibration or disable static quantization."
                )

        if self.calibration:
            self.calibration_config = {**CALIBRATION_CONFIG, **self.calibration_config}
