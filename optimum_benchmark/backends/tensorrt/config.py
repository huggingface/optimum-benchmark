from dataclasses import dataclass
from logging import getLogger

from omegaconf import OmegaConf

from ...import_utils import tesnorrt_version
from ..config import BackendConfig

LOGGER = getLogger("tensorrt")

OmegaConf.register_new_resolver("tensorrt_version", tesnorrt_version)


@dataclass
class TRTConfig(BackendConfig):
    name: str = "tensorrt"
    version: str = "${tensorrt_version:}"
    _target_: str = "optimum_benchmark.backends.tensorrt.backend.TRTBackend"
