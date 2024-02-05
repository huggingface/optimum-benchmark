from logging import getLogger
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from ..config import BenchmarkConfig
from ...env_utils import is_rocm_system

LOGGER = getLogger("inference")

INPUT_SHAPES = {
    # used with all tasks
    "batch_size": 2,
    # used with text input tasks
    "sequence_length": 16,
    # used with multiple choice tasks where input
    # is of shape (batch_size, num_choices, sequence_length)
    "num_choices": 1,
    # used with audio input tasks
    "feature_size": 80,
    "nb_max_frames": 3000,
}

GENERATE_CONFIG = {
    "num_return_sequences": 1,
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": 0,
    "temperature": 1.0,
    "num_beams": 1,
}


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = "inference"
    _target_: str = (
        "optimum_benchmark.benchmarks.inference.benchmark.InferenceBenchmark"
    )

    # benchmark options
    duration: int = 10
    warmup_runs: int = 10

    # additional/optional metrics
    memory: bool = False
    energy: bool = False

    # input options
    input_shapes: Dict = field(default_factory=dict)
    # output options
    new_tokens: Optional[int] = None

    # forward options
    forward_kwargs: Dict[str, Any] = field(default_factory=dict)
    # generation options
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self.input_shapes = {**INPUT_SHAPES, **self.input_shapes}
        self.generate_kwargs = {**GENERATE_CONFIG, **self.generate_kwargs}

        if (
            self.generate_kwargs["max_new_tokens"]
            != self.generate_kwargs["min_new_tokens"]
        ):
            raise ValueError(
                "`max_new_tokens` and `min_new_tokens` must be equal for fixed length output."
            )

        if self.new_tokens is not None:
            self.generate_kwargs["max_new_tokens"] = self.new_tokens
            self.generate_kwargs["min_new_tokens"] = self.new_tokens
        else:
            self.new_tokens = self.generate_kwargs["min_new_tokens"]

        if self.energy and is_rocm_system():
            raise ValueError(
                "Energy measurement through codecarbon is not yet available on ROCm-powered devices."
            )
