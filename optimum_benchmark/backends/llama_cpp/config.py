from dataclasses import dataclass
from logging import getLogger
from typing import Optional

from ...import_utils import llama_cpp_version
from ..config import BackendConfig


@dataclass
class LlamaCppConfig(BackendConfig):
    name: str = "llama_cpp"
    version: Optional[str] = llama_cpp_version()
    _target_: str = "optimum_benchmark.backends.llama_cpp.backend.LlamaCppBackend"

    no_weights: bool = False
    library: str = "llama_cpp"
    filename: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        self.device = self.device.lower()  # type: ignore
        self.library = "llama_cpp"

        if self.device not in ["cuda", "mps", "cpu"]:
            raise ValueError(f"Llama.cpp Backend only supports 'cpu', 'mps' and 'cuda' devices, got {self.device}")
