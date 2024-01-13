from dataclasses import dataclass
from logging import getLogger

from omegaconf import OmegaConf

from ...import_utils import tesnorrt_version
from ..config import BackendConfig

LOGGER = getLogger("tensorrt-llm")

OmegaConf.register_new_resolver("tensorrt_llm_version", tesnorrt_version)

SUPPORTED_DTYPES = ["float16", "bfloat16", "float32"]


@dataclass
class TRTLLMConfig(BackendConfig):
    name: str = "tensorrt_llm"
    version: str = "${tensorrt_llm_version:}"
    _target_: str = "optimum_benchmark.backends.tensorrt_llm.backend.TRTLLMBackend"

    # build config
    tp: int = 1
    pp: int = 1
    use_fp8: bool = False
    dtype: str = "float16"
    optimization_level: int = 2
    use_cuda_graph: bool = False
    gpus_per_node: int = "${available_gpus:}"
    world_size: int = "${backend.gpus_per_node}"

    max_batch_size: int = "${benchmark.input_shapes.batch_size}"
    max_prompt_length: int = "${benchmark.input_shapes.sequence_length}"
    max_new_tokens: int = "${benchmark.new_tokens}"

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of float16, bfloat16, float32, got {self.dtype}")

        if self.gpus_per_node != self.world_size:
            raise ValueError(f"gpus_per_node ({self.gpus_per_node}) != world_size ({self.world_size})")

        if self.world_size != self.pp * self.tp:
            raise ValueError(f"world_size ({self.gpus_per_node}) != pp ({self.pp}) * tp ({self.tp})")
