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
    max_steps: int = 140
    warmup_steps: int = 40

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
            "per_device_train_batch_size": 2,
            "max_steps": "${benchmark.max_steps}",
            # saving trainer output in the experiment directory
            "output_dir": "./trainer_output",
            # infered from the device
            "use_cpu": "${is_cpu:${device}}",
            # we only benchmark training
            "do_train": True,
            "do_eval": False,
            "do_predict": False,
            # by default it reports to "all", so we disable it
            "report_to": "none",
            # from pytorch warning: "this flag results in an extra traversal of the autograd graph every iteration
            # which can adversely affect performance."
            "find_unused_parameters": False,
            # memory metrics are wrong when using multiple processes
            "skip_memory_metrics": True,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"`benchmark.warmup_steps` ({self.warmup_steps}) must be smaller than `benchmark.max_steps` ({self.max_steps})"
            )

        if self.max_steps != self.training_arguments["max_steps"]:
            # normally user will set `benchmark.max_steps` and `benchmark.training_arguments.max_steps` is infered from it
            # but in some cases user might set `benchmark.training_arguments.max_steps` directly
            # so we need to make sure they are the same
            LOGGER.warning(
                f"`benchmark.max_steps` ({self.max_steps}) and `benchmark.training_arguments.max_steps` ({self.training_arguments['max_steps']}) are different. "
                "Using `benchmark.training_arguments.max_steps`."
            )
            self.max_steps = self.training_arguments["max_steps"]
