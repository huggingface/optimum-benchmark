import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from ...env_utils import is_rocm_system
from ...task_utils import DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ..base import BenchmarkConfig

LOGGER = getLogger("inference")

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

DIFUSION_CONFIG = {
    "num_images_per_prompt": 1,
}

OmegaConf.register_new_resolver("can_generate", lambda task: task in TEXT_GENERATION_TASKS)
OmegaConf.register_new_resolver("can_diffuse", lambda task: task in DIFFUSION_TASKS)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = "inference"
    _target_: str = "optimum_benchmark.benchmarks.inference.benchmark.InferenceBenchmark"

    # benchmark options
    duration: int = 10
    warmup_runs: int = 10

    # additional/optional metrics
    memory: bool = False
    energy: bool = False

    # input options
    input_shapes: Dict = field(
        default_factory=lambda: {
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
            "audio_sequence_length": 16000,
        },
    )

    new_tokens: Optional[int] = None
    can_diffuse: bool = "${can_diffuse:${task}}"
    can_generate: bool = "${can_generate:${task}}"

    # forward options
    forward_kwargs: Dict[str, Any] = field(default_factory=dict)

    # generation options
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.can_diffuse:
            self.forward_kwargs = OmegaConf.to_object(OmegaConf.merge(self.forward_kwargs, DIFUSION_CONFIG))

        if self.can_generate:
            self.generate_kwargs = OmegaConf.to_object(OmegaConf.merge(self.generate_kwargs, GENERATE_CONFIG))

            if self.generate_kwargs["max_new_tokens"] != self.generate_kwargs["min_new_tokens"]:
                raise ValueError("`max_new_tokens` and `min_new_tokens` must be equal for fixed length output.")

            if self.new_tokens is not None:
                self.generate_kwargs["max_new_tokens"] = self.new_tokens
                self.generate_kwargs["min_new_tokens"] = self.new_tokens
            else:
                self.new_tokens = self.generate_kwargs["min_new_tokens"]

        if self.energy and os.environ.get("CUDA_VISIBLE_DEVICES", None) and is_rocm_system():
            raise ValueError("Energy measurement through codecarbon is not yet available on ROCm-powered devices.")
