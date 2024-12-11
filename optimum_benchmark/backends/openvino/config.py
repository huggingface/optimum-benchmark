from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...import_utils import openvino_version
from ..config import BackendConfig


@dataclass
class OVConfig(BackendConfig):
    name: str = "openvino"
    version: Optional[str] = openvino_version()
    _target_: str = "optimum_benchmark.backends.openvino.backend.OVBackend"

    no_weights: bool = False

    # ovmodel kwargs
    export: Optional[bool] = None
    use_cache: Optional[bool] = None
    use_merged: Optional[bool] = None
    load_in_8bit: Optional[bool] = None
    load_in_4bit: Optional[bool] = None
    ov_config: Dict[str, Any] = field(default_factory=dict)

    # compilation options
    half: bool = False
    compile: bool = False
    reshape: bool = False
    reshape_kwargs: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self.device = self.device.lower()
        if self.device not in ["cpu", "gpu"]:
            raise ValueError(f"OVBackend only supports CPU devices, got {self.device}")

        if self.intra_op_num_threads is not None:
            raise NotImplementedError("OVBackend does not support intra_op_num_threads. Please use the ov_config")

        if self.inter_op_num_threads is not None:
            raise NotImplementedError("OVBackend does not support inter_op_num_threads. Please use the ov_config")
