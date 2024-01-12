from dataclasses import dataclass
from logging import getLogger

from omegaconf import OmegaConf

from ...import_utils import tesnorrt_version
from ..config import BackendConfig

LOGGER = getLogger("tensorrt-llm")

OmegaConf.register_new_resolver("tensorrt_llm_version", tesnorrt_version)


@dataclass
class TRTLLMConfig(BackendConfig):
    name: str = "tensorrt_llm"
    version: str = "${tensorrt_llm_version:}"
    _target_: str = "optimum_benchmark.backends.tensorrt_llm.backend.TRTLLMBackend"
