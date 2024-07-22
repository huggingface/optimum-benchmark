from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ...import_utils import vllm_version
from ..config import BackendConfig


@dataclass
class VLLMConfig(BackendConfig):
    name: str = "vllm"
    version: Optional[str] = vllm_version()
    _target_: str = "optimum_benchmark.backends.vllm.backend.VLLMBackend"

    # creates a model from scratch with dummy weights
    no_weights: bool = False

    # decides whether to use the offline or online llm engine
    serving_mode: str = "online"

    # passed to EngineArgs
    engine_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # duplicates that are handled by the backend config directly
        if "model" in self.engine_args:
            raise ValueError("model should not be passed in `backend.engine_args`, use `backend.model` instead")

        if "tokenizer" in self.engine_args:
            raise ValueError("tokenizer should not be passed in `backend.engine_args`, use `backend.processor` instead")

        if "device" in self.engine_args:
            raise ValueError("device should not be passed in `backend.engine_args`, use `backend.device` instead")

        if self.serving_mode not in ["offline", "online"]:
            raise ValueError(f"Invalid serving_mode: {self.serving_mode}. Must be 'online' or 'offline'.")

        # needed for task/library/model_type inference
        self.model_kwargs = {
            "revision": self.engine_args.get("revision", "main"),
            "trust_remote_code": self.engine_args.get("trust_remote_code", False),
            **self.model_kwargs,
        }
        self.processor_kwargs = {
            "revision": self.engine_args.get("tokenizer_revision", "main"),
            "trust_remote_code": self.engine_args.get("trust_remote_code", False),
            **self.processor_kwargs,
        }

        super().__post_init__()

        if self.engine_args.get("disable_log_stats", None) is None:
            self.engine_args["disable_log_stats"] = True

        if self.serving_mode == "online":
            if self.engine_args.get("disable_log_requests", None) is None:
                self.engine_args["disable_log_requests"] = True

    def to_engine_args(self) -> Dict[str, Any]:
        return dict(
            model=self.model,
            tokenizer=self.processor,
            device=self.device,
            **self.engine_args,
        )
