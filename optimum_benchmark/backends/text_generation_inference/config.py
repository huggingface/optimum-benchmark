import os
from dataclasses import dataclass
from typing import Optional

from ..config import BackendConfig


@dataclass
class TGIConfig(BackendConfig):
    name: str = "tgi"
    version: str = "1.0.3"
    _target_: str = "optimum_benchmark.backends.text_generation_inference.backend.TGIBackend"

    # server options
    image: str = "ghcr.io/huggingface/text-generation-inference"
    volume: str = f"{os.path.expanduser('~')}/.cache/huggingface/hub"
    shm_size: str = "1g"
    address: str = "127.0.0.1"
    port: int = 1111

    # torch options
    torch_dtype: Optional[str] = None  # None, float32, float16, bfloat16
    # optimization options
    disable_custom_kernels: bool = False  # True, False
    # quantization options
    quantization: Optional[str] = None  # None, bitsandbytes-nf4, bitsandbytes-fp4

    def __post_init__(self):
        super().__post_init__()

        if self.torch_dtype is not None:
            if self.torch_dtype not in ["float32", "float16", "bfloat16"]:
                raise ValueError(f"Invalid value for torh_dtype: {self.torch_dtype}")
