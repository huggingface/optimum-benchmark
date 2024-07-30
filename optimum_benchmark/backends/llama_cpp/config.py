from dataclasses import dataclass
from typing import Optional

from ...import_utils import llama_cpp_version
from ..config import BackendConfig


@dataclass
class LlamaCppConfig(BackendConfig):
    name: str = "llama_cpp"
    version: Optional[str] = llama_cpp_version()
    _target_: str = "optimum_benchmark.backends.llama_cpp.backend.LlamaCppBackend"

    no_weights: bool = False
    filename: Optional[str] = None

    def __post_init__(self):
        self.library = "llama_cpp"
        self.model_type = "llama_cpp"

        super().__post_init__()

        if self.task not in ["feature-extraction", "text-generation"]:
            raise NotImplementedError(f"Task {self.task} is not supported by LlamaCpp backend.")

        if self.no_weights:
            raise NotImplementedError("`no_weights` benchmarking is not supported by LlamaCpp backend.")
