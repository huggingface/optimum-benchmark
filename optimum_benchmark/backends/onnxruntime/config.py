import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from ...import_utils import onnxruntime_version
from ..config import BackendConfig
from ..peft_utils import PEFT_CONFIGS, PEFT_TASKS_TYPES

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
    "is_static": False,
    # is_static is mandatory
}

TRT_PROVIDER_OPTIONS = {
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "/tmp/trt_cache",
}

DEVICE_PROVIDER_MAP = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
IO_BINDING_PROVIDERS = ["CPUExecutionProvider", "CUDAExecutionProvider"]

OmegaConf.register_new_resolver("onnxruntime_version", onnxruntime_version)
OmegaConf.register_new_resolver("infer_provider", lambda device: DEVICE_PROVIDER_MAP[device])
OmegaConf.register_new_resolver("is_profiling", lambda benchmark_name: benchmark_name == "profiling")
OmegaConf.register_new_resolver("ort_io_binding", lambda provider: provider in IO_BINDING_PROVIDERS)


@dataclass
class ORTConfig(BackendConfig):
    name: str = "onnxruntime"
    version: str = "${onnxruntime_version:}"
    _target_: str = "optimum_benchmark.backends.onnxruntime.backend.ORTBackend"

    no_weights: bool = False

    # export options
    export: bool = True
    use_cache: bool = True
    use_merged: bool = False
    torch_dtype: Optional[str] = None

    # provider options
    provider: str = "${infer_provider:${device}}"
    provider_options: Dict[str, Any] = field(default_factory=lambda: {})

    # inference options
    use_io_binding: bool = "${ort_io_binding:${device}}"
    session_options: Dict[str, Any] = field(
        default_factory=lambda: {"enable_profiling": "${is_profiling:${benchmark.name}}"}
    )

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

    # ort-training is basically a different package so we might need to separate these two backends in the future
    use_inference_session: bool = "${is_inference:${benchmark.name}}"

    # peft options
    peft_strategy: Optional[str] = None
    peft_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if not self.no_weights and not self.export and self.torch_dtype is not None:
            raise NotImplementedError("Can't convert an exported model's weights to a different dtype.")

        if self.provider == "TensorrtExecutionProvider":
            self.provider_options = OmegaConf.to_object(OmegaConf.merge(TRT_PROVIDER_OPTIONS, self.provider_options))
            os.makedirs(self.provider_options["trt_engine_cache_path"], exist_ok=True)

        if self.quantization:
            self.quantization_config = OmegaConf.to_object(
                OmegaConf.merge(QUANTIZATION_CONFIG, self.quantization_config)
            )
            # raise ValueError if the quantization is static but calibration is not enabled
            if self.quantization_config["is_static"] and self.auto_calibration is None and not self.calibration:
                raise ValueError(
                    "Quantization is static but calibration is not enabled. "
                    "Please enable calibration or disable static quantization."
                )

        if self.auto_quantization is not None:
            self.auto_quantization_config = OmegaConf.to_object(
                OmegaConf.merge(AUTO_QUANTIZATION_CONFIG, self.auto_quantization_config)
            )
            if self.auto_quantization_config["is_static"] and self.auto_calibration is None and not self.calibration:
                raise ValueError(
                    "Quantization is static but calibration is not enabled. "
                    "Please enable calibration or disable static quantization."
                )

        if self.calibration:
            self.calibration_config = OmegaConf.to_object(OmegaConf.merge(CALIBRATION_CONFIG, self.calibration_config))

        if self.peft_strategy is not None:
            if self.peft_strategy not in PEFT_CONFIGS:
                raise ValueError(
                    f"`peft_strategy` must be one of {list(PEFT_CONFIGS.keys())}. Got {self.peft_strategy} instead."
                )
            PEFT_CONFIG = PEFT_CONFIGS[self.peft_strategy]
            self.peft_config = OmegaConf.to_object(OmegaConf.merge(PEFT_CONFIG, self.peft_config))

            if self.peft_config["task_type"] is None:
                raise ValueError(f"`peft_config.task_type` must be set to one of the following {PEFT_TASKS_TYPES}")
