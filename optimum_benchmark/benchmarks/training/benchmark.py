from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict

from pandas import DataFrame

from ...generators.dataset_generator import DatasetGenerator
from ..base import Benchmark
from ..utils import MeasurementCallback, get_data_collator
from .config import TrainingConfig

if TYPE_CHECKING:
    from ...backends.base import Backend

LOGGER = getLogger("training")


class TrainingBenchmark(Benchmark[TrainingConfig]):
    NAME = "training"

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
            "warmup.runtime(s)": trainer_state.warmup_runtime,
            "warmup.throughput(samples/s)": trainer_state.warmup_samples_per_second,
            # training metrics
            "training.runtime(s)": trainer_state.training_runtime,
            "training.throughput(samples/s)": trainer_state.training_samples_per_second,
            # overall training metrics
            "overall_training.runtime(s)": trainer_state.overall_training_runtime,
            "overall_training.throughput(samles/s)": (trainer_state.overall_training_samples_per_second),
        }

    def get_results_df(self) -> DataFrame:
        return DataFrame(self.training_metrics, index=[0])

    def save(self) -> None:
        LOGGER.info("Saving training results")
        results_df = self.get_results_df()
        results_df.to_csv("training_results.csv")
