from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, Optional

from ...system_utils import is_rocm_system
from ..config import ScenarioConfig

LOGGER = getLogger("inference")

INPUT_SHAPES = {"batch_size": 2, "num_choices": 2, "sequence_length": 16}


@dataclass
class InferenceConfig(ScenarioConfig):
    name: str = "inference"
    _target_: str = "optimum_benchmark.scenarios.inference.scenario.InferenceScenario"

    # benchmark options
    iterations: int = field(
        default=10,
        metadata={"help": "Minimum number of iterations to run the benchmark, set to 0 to disable this constraint"},
    )
    duration: int = field(
        default=10,
        metadata={"help": "Minimum duration of the benchmark in seconds, set to 0 to disable this constraint"},
    )
    warmup_runs: int = field(
        default=10,
        metadata={"help": "Number of warmup runs to perform before benchmarking, set to 0 to disable warmup"},
    )

    # input/output config
    input_shapes: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Input shapes for the model. Missing keys will be filled with default values."},
    )
    new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "If set, `max_new_tokens` and `min_new_tokens` will be set to this value."},
    )

    # tracking options
    latency: bool = field(default=True, metadata={"help": "Measure latencies and throughputs"})
    memory: bool = field(default=False, metadata={"help": "Measure max memory usage"})
    energy: bool = field(default=False, metadata={"help": "Measure energy usage and efficiency"})

    # methods kwargs
    forward_kwargs: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Keyword arguments to pass to the forward method of the backend."}
    )
    generate_kwargs: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Keyword arguments to pass to the generate method of the backend."}
    )
    call_kwargs: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Keyword arguments to pass to the call method of the backend."}
    )

    def __post_init__(self):
        super().__post_init__()

        self.input_shapes = {**INPUT_SHAPES, **self.input_shapes}

        if self.new_tokens is not None:
            LOGGER.warning(
                "`new_tokens` is deprecated. Use `max_new_tokens` and `min_new_tokens` instead. "
                "Setting `max_new_tokens` and `min_new_tokens` to `new_tokens`."
            )
            self.generate_kwargs["max_new_tokens"] = self.new_tokens
            self.generate_kwargs["min_new_tokens"] = self.new_tokens

        if (
            "max_new_tokens" in self.generate_kwargs
            and "min_new_tokens" in self.generate_kwargs
            and self.generate_kwargs["max_new_tokens"] != self.generate_kwargs["min_new_tokens"]
        ):
            raise ValueError(
                "Setting `min_new_tokens` and `max_new_tokens` to different values results in non-deterministic behavior."
            )

        elif "max_new_tokens" in self.generate_kwargs and "min_new_tokens" not in self.generate_kwargs:
            LOGGER.warning(
                "Setting `max_new_tokens` without `min_new_tokens` results in non-deterministic behavior. "
                "Setting `min_new_tokens` to `max_new_tokens`."
            )
            self.generate_kwargs["min_new_tokens"] = self.generate_kwargs["max_new_tokens"]

        elif "min_new_tokens" in self.generate_kwargs and "max_new_tokens" not in self.generate_kwargs:
            LOGGER.warning(
                "Setting `min_new_tokens` without `max_new_tokens` results in non-deterministic behavior. "
                "Setting `max_new_tokens` to `min_new_tokens`."
            )
            self.generate_kwargs["max_new_tokens"] = self.generate_kwargs["min_new_tokens"]

        if self.energy and is_rocm_system():
            raise ValueError("Energy measurement through codecarbon is not yet available on ROCm-powered devices.")
