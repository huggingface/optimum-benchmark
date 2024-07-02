from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict

from ..config import ScenarioConfig

LOGGER = getLogger("training")

TRAINING_ARGUMENT = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "output_dir": "./trainer_output",
    "evaluation_strategy": "no",
    "eval_strategy": "no",
    "save_strategy": "no",
    "do_train": True,
    "use_cpu": False,
    "max_steps": -1,
    # disable evaluation
    "do_eval": False,
    "do_predict": False,
    # disable custom logging
    "report_to": "none",
    # disbale transformers memory metrics
    "skip_memory_metrics": True,
    # from pytorch warning: "this flag results in an extra traversal of the
    # autograd graph every iteration which can adversely affect performance."
    "ddp_find_unused_parameters": False,
}

DATASET_SHAPES = {"dataset_size": 500, "sequence_length": 16, "num_choices": 1}


@dataclass
class TrainingConfig(ScenarioConfig):
    name: str = "training"
    _target_: str = "optimum_benchmark.scenarios.training.scenario.TrainingScenario"

    # training options
    max_steps: int = 140
    warmup_steps: int = 40

    # dataset options
    dataset_shapes: Dict[str, Any] = field(default_factory=dict)
    # training options
    training_arguments: Dict[str, Any] = field(default_factory=dict)

    # tracking options
    latency: bool = field(default=True, metadata={"help": "Measure latencies and throughputs"})
    memory: bool = field(default=False, metadata={"help": "Measure max memory usage"})
    energy: bool = field(default=False, metadata={"help": "Measure energy usage"})

    def __post_init__(self):
        super().__post_init__()

        self.dataset_shapes = {**DATASET_SHAPES, **self.dataset_shapes}
        self.training_arguments = {**TRAINING_ARGUMENT, **self.training_arguments}

        if self.training_arguments["max_steps"] == -1:
            self.training_arguments["max_steps"] = self.max_steps

        if self.max_steps != self.training_arguments["max_steps"]:
            LOGGER.warning(
                f"`scenario.max_steps` ({self.max_steps}) and `scenario.training_arguments.max_steps` "
                f"({self.training_arguments['max_steps']}) are different. "
                "Using `scenario.training_arguments.max_steps`."
            )
            self.max_steps = self.training_arguments["max_steps"]

        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"`scenario.warmup_steps` ({self.warmup_steps}) must be smaller than `scenario.max_steps` ({self.max_steps})"
            )
