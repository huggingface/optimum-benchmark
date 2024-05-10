from dataclasses import dataclass
from typing import Optional

from ...import_utils import vllm_version
from ..config import BackendConfig


@dataclass
class VLLMConfig(BackendConfig):
    name: str = "vllm"
    version: Optional[str] = vllm_version()
    _target_: str = "optimum_benchmark.backends.vllm.backend.VLLMBackend"

    # optimum-benchmark
    no_weights: bool = False

    # tokenizer
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False

    # parallelism
    tensor_parallel_size: int = 1

    # precision
    dtype: str = "auto"
    quantization: Optional[str] = None

    # cuda graphs
    enforce_eager: bool = False
    max_context_len_to_capture: Optional[int] = None
    max_seq_len_to_capture: int = 8192

    # kernels
    disable_custom_all_reduce: bool = False

    # memory
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4

    def __post_init__(self):
        super().__post_init__()

        self.device = self.device.lower()

        if self.device not in ["cuda", "neuron", "cpu"]:
            raise ValueError(f"VLLM Backend only supports 'cpu', 'cuda' and 'neuron' devices, got {self.device}")
