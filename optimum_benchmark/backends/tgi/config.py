import os
from dataclasses import dataclass
from typing import Optional

from ..config import BackendConfig


@dataclass
class TGIConfig(BackendConfig):
    name: str = "tgi"
    version: str = "1.0.3"
    _target_: str = "optimum_benchmark.backends.tgi.backend.TGIBackend"

    # server options
    volume: str = f"{os.path.expanduser('~')}/.cache/huggingface/hub"
    shm_size: str = "1g"
    address: str = "127.0.0.1"
    port: int = 1111

    # client options
    timeout: int = 100

    # quantization options
    quantization_scheme: Optional[str] = None  # bitsandbytes-nf4, bitsandbytes-fp4
