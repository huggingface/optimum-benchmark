from typing import Any, Dict
from dataclasses import dataclass, field
from logging import getLogger

from omegaconf import OmegaConf
from pandas import DataFrame

from ..backends.base import Backend
from .base import Benchmark, BenchmarkConfig
from ..generators.dataset_generator import DatasetGenerator
from .training_utils import MeasurementCallback, get_data_collator


LOGGER = getLogger("training")

# resolvers
OmegaConf.register_new_resolver("is_cpu", lambda device: device == "cpu")


@dataclass
class TrainingConfig(BenchmarkConfig):
    name: str = "training"
    _target_: str = "optimum_benchmark.benchmarks.training.TrainingBenchmark"

    # training options
    warmup_steps: int = 2

    # dataset options
    dataset_shapes: Dict = field(
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
    training_arguments: Dict = field(
        default_factory=lambda: {
            # these are arguments that we set by default
            # but can be overwritten by the user
            "skip_memory_metrics": False,
            "output_dir": "./trainer_output",
            "use_cpu": "${is_cpu:${device}}",
            "ddp_find_unused_parameters": False,
            "do_train": True,
            "do_eval": False,
            "do_predict": False,
        }
    )


class TrainingBenchmark(Benchmark):
    name: str = "training"
    config: TrainingConfig

    def __init__(self):
        # initialize training results
        self.training_metrics: Dict[str, Any] = {}

    def configure(self, config: TrainingConfig):
        super().configure(config)

    def run(self, backend: "Backend") -> None:
        LOGGER.info("Running training benchmark")
        task = backend.task
        dataset_shapes = {**self.config.dataset_shapes, **backend.model_shapes}
        dataset_generator = DatasetGenerator(task=task, dataset_shapes=dataset_shapes)

        training_dataset = dataset_generator.generate()
        training_data_collator = get_data_collator(task=task)
        training_callbacks = [MeasurementCallback(self.config.warmup_steps)]

        trainer_state = backend.train(
            training_dataset=training_dataset,
            training_callbacks=training_callbacks,
            training_data_collator=training_data_collator,
            training_arguments=self.config.training_arguments,
        )

        self.training_metrics = {
            # warmup metrics
            "warmup_runtime": trainer_state.warmup_runtime,
            "warmup_throughput()": trainer_state.warmup_samples_per_second,
            # training metrics
            "train_runtime": trainer_state.train_runtime,
            "training_throughput": trainer_state.train_samples_per_second,
            # overall training metrics
            "overall_train_runtime": trainer_state.overall_train_runtime,
            "overall_training_throughput": trainer_state.overall_train_samples_per_second,
        }

    def get_results_df(self) -> DataFrame:
        return DataFrame(self.training_metrics, index=[0])

    def save(self) -> None:
        LOGGER.info("Saving training results")
        results_df = self.get_results_df()
        results_df.to_csv("training_results.csv")
