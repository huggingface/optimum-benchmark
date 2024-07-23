from dataclasses import dataclass
from typing import Optional

from optimum_benchmark.task_utils import TEXT_EMBEDDING_TASKS, TEXT_GENERATION_TASKS

from ...import_utils import llama_cpp_version
from ..config import BackendConfig


def llama_cpp_model_kwargs():
    return {"verbose": True}


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

        # if self.task not in TEXT_GENERATION_TASKS + TEXT_EMBEDDING_TASKS:
        #     raise NotImplementedError(f"Llama.cpp does not support task {self.task}")

        self.device = self.device.lower()  # type: ignore

        if self.device not in ["cuda", "mps", "cpu"]:
            raise ValueError(f"Llama.cpp Backend only supports 'cpu', 'mps' and 'cuda' devices, got {self.device}")
