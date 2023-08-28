import importlib.metadata
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from ..base import BackendConfig
from .utils import infer_device_id


def onnxruntime_version():
    try:
        return "ort:" + importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        try:
            return "ort-gpu:" + importlib.metadata.version("onnxruntime-gpu")
        except importlib.metadata.PackageNotFoundError:
            return "ort:unknown"


OmegaConf.register_new_resolver(
    "is_gpu",
    lambda device: "cuda" in device.lower(),
)
OmegaConf.register_new_resolver(
    "is_profiling",
    lambda benchmark_name: benchmark_name == "profiling",
)
OmegaConf.register_new_resolver(
    "infer_provider",
    lambda device: "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider",
)
OmegaConf.register_new_resolver(
    "infer_device_id",
    lambda device: infer_device_id(device),
)
OmegaConf.register_new_resolver(
    "onnxruntime_version",
    lambda: onnxruntime_version(),
)

OPTIMIZATION_CONFIG = {
    "optimization_level": 1,  # 0, 1, 2, 99
    "optimize_for_gpu": "${is_gpu:${device}}",
    "fp16": False,
    "enable_transformers_specific_optimizations": True,
    "enable_gelu_approximation": False,
    "disable_gelu_fusion": False,
    "disable_layer_norm_fusion": False,
    "disable_attention_fusion": False,
    "disable_skip_layer_norm_fusion": True,
    "disable_bias_skip_layer_norm_fusion": False,
    "disable_bias_gelu_fusion": False,
    "use_mask_index": False,
    "no_attention_mask": False,
    "disable_embed_layer_norm_fusion": True,
    "disable_shape_inference": False,
    "use_multi_head_attention": False,
    "enable_gemm_fast_gelu_fusion": False,
    "use_raw_attention_mask": False,
    "disable_group_norm_fusion": True,
    "disable_packed_kv": True,
}

AUTO_OPTIMIZATION_CONFIG = {
    "for_gpu": "${is_gpu:${device}}",
    # full auto optimization config depends on the level so we keep it minimal
}

QUANTIZATION_CONFIG = {
    "is_static": False,
    "format": "QOperator",  # QOperator, QDQ
    "mode": "IntegerOps",  # QLinearOps, IntegerOps
    "activations_dtype": "QUInt8",  # QInt8, QUInt8
    "activations_symmetric": False,
    "weights_dtype": "QInt8",  # QInt8, QUInt8
    "weights_symmetric": True,
    "per_channel": False,
    "reduce_range": False,
    "operators_to_quantize": [
        "MatMul",
        "Add",
    ],
}

AUTO_QUANTIZATION_CONFIG = {
    "is_static": False,
    # full auto quantization config depends on the strategy so we keep it minimal
}

CALIBRATION_CONFIG = {
    "dataset_name": "glue",
    "num_samples": 300,
    "dataset_config_name": "sst2",
    "dataset_split": "train",
    "preprocess_batch": True,
    "preprocess_class": "optimum_benchmark.preprocessors.glue.GluePreprocessor",
}
PROVIDER_OPTIONS = {"device_id": "${infer_device_id:${device}}"}
SESSION_OPTIONS = {"enable_profiling": "${is_profiling:${benchmark.name}}"}


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
    device_id: Optional[int] = "${oc.deprecated:backend.provider_options.device_id}"
    provider_options: Dict[str, Any] = field(default_factory=lambda: PROVIDER_OPTIONS)

    # inference options
    use_io_binding: bool = "${is_gpu:${device}}"
    enable_profiling: bool = "${oc.deprecated:backend.session_options.enable_profiling}"
    session_options: Dict[str, Any] = field(default_factory=lambda: SESSION_OPTIONS)

    # optimization options
    optimization: bool = False
    optimization_config: Dict[str, Any] = field(default_factory=dict)

    # quantization options
    quantization: bool = False
    quantization_config: Dict[str, Any] = field(default_factory=dict)

    # calibration options
    calibration: bool = False
    calibration_config: Dict[str, Any] = field(default_factory=dict)

    # null, O1, O2, O3, O4
    auto_optimization: Optional[str] = None
    auto_optimization_config: Dict[str, Any] = field(default_factory=dict)

    # null, arm64, avx2, avx512, avx512_vnni, tensorrt
    auto_quantization: Optional[str] = None
    auto_quantization_config: Dict[str, Any] = field(default_factory=dict)

    # ort-training is basically a different package so we might need to seperate these two backends in the future
    use_inference_session: bool = "${is_inference:${benchmark.name}}"

    def __post_init__(self):
        if not self.no_weights and not self.export and self.torch_dtype is not None:
            raise NotImplementedError("Can't convert an exported model's weights to a different dtype.")

        if self.optimization:
            self.optimization_config = OmegaConf.to_container(
                OmegaConf.merge(OPTIMIZATION_CONFIG, self.optimization_config)
            )
        if self.quantization:
            self.quantization_config = OmegaConf.to_container(
                OmegaConf.merge(QUANTIZATION_CONFIG, self.quantization_config)
            )
            # raise ValueError if the quantization is static but calibration is not enabled
            if self.quantization_config["is_static"] and not self.calibration:
                raise ValueError(
                    "Quantization is static but calibration is not enabled. Please enable calibration or disable static quantization."
                )

        if self.auto_optimization is not None:
            self.auto_optimization_config = OmegaConf.to_container(
                OmegaConf.merge(AUTO_OPTIMIZATION_CONFIG, self.auto_optimization_config)
            )
        if self.auto_quantization is not None:
            self.auto_quantization_config = OmegaConf.to_container(
                OmegaConf.merge(AUTO_QUANTIZATION_CONFIG, self.auto_quantization_config)
            )
            if self.auto_quantization_config["is_static"] and not self.calibration:
                raise ValueError(
                    "Quantization is static but calibration is not enabled. Please enable calibration or disable static quantization."
                )

        if self.calibration:
            self.calibration_config = OmegaConf.to_container(
                OmegaConf.merge(CALIBRATION_CONFIG, self.calibration_config)
            )
