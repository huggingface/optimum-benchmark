from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict

from omegaconf import OmegaConf

from ..base import BenchmarkConfig

LOGGER = getLogger("training")

# resolvers
OmegaConf.register_new_resolver("is_cpu", lambda device: device == "cpu")


@dataclass
class TrainingConfig(BenchmarkConfig):
    name: str = "training"
    _target_: str = "optimum_benchmark.benchmarks.training.benchmark.TrainingBenchmark"

    # training options
    warmup_steps: int = 40  # still thinks this too high

    # dataset options
    dataset_shapes: Dict[str, Any] = field(
        default_factory=lambda: {
            # used with all tasks
            "dataset_size": 500,
            # used with text input tasks
            "sequence_length": 16,
            # used with multiple choice tasks where input
            # is of shape (batch_size, num_choices, sequence_length)
            "num_choices": 1,
            # used with audio input tasks
            "feature_size": 80,
            "nb_max_frames": 3000,
            "audio_sequence_length": 16000,
        }
    )

    # training options
    training_arguments: Dict[str, Any] = field(
        default_factory=lambda: {
            # these are arguments that we set by default
            # but can be overwritten by the user
            "skip_memory_metrics": True,
            # memory metrics are wrong when using multiple processes
            "output_dir": "./trainer_output",
            "use_cpu": "${is_cpu:${device}}",
            "ddp_find_unused_parameters": False,
            "do_train": True,
            "do_eval": False,
            "do_predict": False,
            "report_to": "none",
        }
    )
