from dataclasses import dataclass, field
from typing import Any, Dict

from omegaconf import OmegaConf

from ...import_utils import openvino_version
from ..config import BackendConfig

OmegaConf.register_new_resolver("openvino_version", openvino_version)


@dataclass
class OVConfig(BackendConfig):
    name: str = "openvino"
    version: str = "${openvino_version:}"
    _target_: str = "optimum_benchmark.backends.openvino.backend.OVBackend"

    # load options
    no_weights: bool = False

    # export options
    export: bool = True
    use_cache: bool = True
    use_merged: bool = False

    # openvino config
    openvino_config: Dict[str, Any] = field(default_factory=dict)

    # compilation options
    half: bool = False
    reshape: bool = False

    # quantization options
    quantization: bool = False
    quantization_config: Dict[str, Any] = field(default_factory=dict)

    # calibration options
    calibration: bool = False
    calibration_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if self.quantization and not self.calibration:
            raise ValueError("OpenVINO quantization requires enabling calibration.")
