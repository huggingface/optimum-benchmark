import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, Dict

from omegaconf import OmegaConf

from ..base import BackendConfig

OmegaConf.register_new_resolver(
    "openvino_version",
    lambda: importlib.metadata.version("openvino"),
)

# https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/configuration.py#L81
QUANTIZATION_CONFIG = {
    "compression": None,
    "input_info": None,
    "save_onnx_model": False,
}

CALIBRATION_CONFIG = {
    "dataset_name": "glue",
    "num_samples": 300,
    "dataset_config_name": "sst2",
    "dataset_split": "train",
    "preprocess_batch": True,
    "preprocess_class": "optimum_benchmark.preprocessors.glue.GluePreprocessor",
}


@dataclass
class OVConfig(BackendConfig):
    name: str = "openvino"
    version: str = "${openvino_version:}"
    _target_: str = "optimum_benchmark.backends.openvino.backend.OVBackend"

    # export options
    export: bool = True
    use_cache: bool = True
    use_merged: bool = False

    # compiling options
    reshape: bool = False
    half: bool = False

    # quantization options
    quantization: bool = False
    quantization_config: Dict[str, Any] = field(default_factory=dict)

    # calibration options
    calibration: bool = False
    calibration_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.quantization:
            self.quantization_config = OmegaConf.to_container(
                OmegaConf.merge(QUANTIZATION_CONFIG, self.quantization_config)
            )
            if not self.calibration:
                raise ValueError("OpenVINO quantization requires enabling calibration.")
            else:
                self.calibration_config = OmegaConf.to_container(
                    OmegaConf.merge(CALIBRATION_CONFIG, self.calibration_config)
                )
