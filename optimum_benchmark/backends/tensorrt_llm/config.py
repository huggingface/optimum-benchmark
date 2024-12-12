from dataclasses import dataclass
from typing import Optional

from ...import_utils import tesnorrt_llm_version
from ..config import BackendConfig

SUPPORTED_DTYPES = [None, "float16", "bfloat16", "float32"]


@dataclass
class TRTLLMConfig(BackendConfig):
    name: str = "tensorrt-llm"
    version: Optional[str] = tesnorrt_llm_version()
    _target_: str = "optimum_benchmark.backends.tensorrt_llm.backend.TRTLLMBackend"

    no_weights: bool = False

    # trtllm kwargs
    tp: Optional[int] = None
    pp: Optional[int] = None
    dtype: Optional[str] = None
    use_fp8: Optional[bool] = None
    world_size: Optional[int] = None
    gpus_per_node: Optional[int] = None
    max_input_len: Optional[int] = None
    max_output_len: Optional[int] = None
    max_batch_size: Optional[int] = None
    max_new_tokens: Optional[int] = None
    max_prompt_length: Optional[int] = None
    optimization_level: Optional[int] = None
    use_cuda_graph: Optional[bool] = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.device != "cuda":
            raise NotImplementedError(f"TRTLLMBackend only supports device cuda, got {self.device}")

        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of float16, bfloat16, float32, got {self.dtype}")

        if self.gpus_per_node is not None and self.world_size is not None and self.gpus_per_node != self.world_size:
            raise ValueError(f"gpus_per_node ({self.gpus_per_node}) != world_size ({self.world_size})")

        if (
            self.world_size is not None
            and self.pp is not None
            and self.tp is not None
            and self.world_size != self.pp * self.tp
        ):
            raise ValueError(f"world_size ({self.gpus_per_node}) != pp ({self.pp}) * tp ({self.tp})")
