from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, Union

from ...system_utils import is_rocm_system
from ..config import ScenarioConfig

LOGGER = getLogger("energy_star")

INPUT_SHAPES = {"batch_size": 1}


@dataclass
class EnergyStarConfig(ScenarioConfig):
    name: str = "energy_star"
    _target_: str = "optimum_benchmark.scenarios.energy_star.scenario.EnergyStarScenario"

    # dataset options
    dataset_name: str = field(default="", metadata={"help": "Name of the dataset on the HF Hub."})
    dataset_config: str = field(default="", metadata={"help": "Name of the config of the dataset."})
    dataset_split: str = field(default="train", metadata={"help": "Dataset split to use."})
    num_samples: int = field(default=-1, metadata={"help": "Number of samples to select in the dataset. -1 means all."})
    input_shapes: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Input shapes for the model. Missing keys will be filled with default values."},
    )

    # text dataset options
    text_column_name: str = field(default="text", metadata={"help": "Name of the column with the text input."})
    truncation: Union[bool, str] = field(default=False, metadata={"help": "To truncate the inputs."})
    max_length: int = field(
        default=-1, metadata={"help": "Maximum length to use by one of the truncation/padding parameters"}
    )

    # scenario options
    warmup_runs: int = field(default=10, metadata={"help": "Number of warmup runs to perform before scenarioing"})

    # tracking options
    energy: bool = field(default=True, metadata={"help": "Measure energy usage"})

    # methods kwargs
    forward_kwargs: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Keyword arguments to pass to the forward method of the model."}
    )
    generate_kwargs: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Keyword arguments to pass to the generate method of the model."}
    )
    call_kwargs: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Keyword arguments to pass to the __call__ method of the pipeline."}
    )

    def __post_init__(self):
        super().__post_init__()

        self.input_shapes = {**INPUT_SHAPES, **self.input_shapes}

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
