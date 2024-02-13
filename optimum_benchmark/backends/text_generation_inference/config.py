import os
from dataclasses import dataclass
from typing import Optional

from ..config import BackendConfig


@dataclass
class TGIConfig(BackendConfig):
    name: str = "text-generation-inference"
    version: Optional[str] = "0.0.1"
    _target_: str = "optimum_benchmark.backends.text_generation_inference.backend.TGIBackend"

    # optimum benchmark specific
    no_weights: bool = False

    # docker options
    image: str = "ghcr.io/huggingface/text-generation-inference:latest"
    volume: str = f"{os.path.expanduser('~')}/.cache/huggingface/hub"
    address: str = "127.0.0.1"
    shm_size: str = "1g"
    port: int = 1111

    # torch options
    torch_dtype: Optional[str] = None  # None, float32, float16, bfloat16
    # optimization options
    disable_custom_kernels: bool = False  # True, False
    # quantization options
    quantization_scheme: Optional[str] = None  # None, bitsandbytes-nf4, bitsandbytes-fp4
    # sharding options
    sharded: Optional[bool] = None  # None, True, False
    num_shard: Optional[int] = None  # None, 1, 2, 4, 8, 16, 32, 64

    def __post_init__(self):
        super().__post_init__()

        if self.torch_dtype is not None:
            if self.torch_dtype not in ["float32", "float16", "bfloat16"]:
                raise ValueError(f"Invalid value for dtype: {self.torch_dtype}")
