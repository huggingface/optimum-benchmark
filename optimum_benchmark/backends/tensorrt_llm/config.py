from dataclasses import dataclass
from typing import Optional

from ...import_utils import tesnorrt_llm_version
from ..config import BackendConfig

SUPPORTED_DTYPES = ["float16", "bfloat16", "float32"]


@dataclass
class TRTLLMConfig(BackendConfig):
    name: str = "tensorrt-llm"
    version: Optional[str] = tesnorrt_llm_version()
    _target_: str = "optimum_benchmark.backends.tensorrt_llm.backend.TRTLLMBackend"

    # build config
    tp: int = 1
    pp: int = 1
    use_fp8: bool = False
    dtype: str = "float16"
    optimization_level: int = 2
    use_cuda_graph: bool = False

    world_size: int = 1
    gpus_per_node: int = 1

    max_prompt_length: int = 128
    max_new_tokens: int = -1
    max_batch_size: int = 1
    max_beam_width: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.device != "cuda":
            raise NotImplementedError(f"TRTLLMBackend only supports device cuda, got {self.device}")

        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of float16, bfloat16, float32, got {self.dtype}")

        if self.gpus_per_node != self.world_size:
            raise ValueError(f"gpus_per_node ({self.gpus_per_node}) != world_size ({self.world_size})")

        if self.world_size != self.pp * self.tp:
            raise ValueError(f"world_size ({self.gpus_per_node}) != pp ({self.pp}) * tp ({self.tp})")
