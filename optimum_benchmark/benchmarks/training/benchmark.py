import time
from typing import Any, Dict
from logging import getLogger

from transformers import (
    default_data_collator,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from ...generators.dataset_generator import DatasetGenerator
from ...backends.base import Backend
from .config import TrainingConfig
from ..base import Benchmark

LOGGER = getLogger("training")


class TrainingBenchmark(Benchmark[TrainingConfig]):
    NAME = "training"

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running training benchmark")

        LOGGER.info("\t+ Creating dataset generator")
        dataset_generator = DatasetGenerator(
            task=backend.config.task,
            model_shapes=backend.model_shapes,
            dataset_shapes=self.config.dataset_shapes,
        )

        LOGGER.info("\t+ Generating training dataset")
        training_dataset = dataset_generator.generate()

        LOGGER.info("\t+ Creating training callbacks")
        training_callbacks = [MeasurementCallback(warmup_steps=self.config.warmup_steps)]

        self.trainer_state = backend.train(
            training_dataset=training_dataset,
            training_callbacks=training_callbacks,
            training_data_collator=default_data_collator,
            training_arguments=self.config.training_arguments,
        )

        LOGGER.info(f"Training runtime: {self.trainer_state.training_runtime:.3g} (s)")
        LOGGER.info(f"Training throughput: {self.trainer_state.training_throughput:.3g} (samples/s)")

        return self.report()

    def report(self) -> Dict[str, Any]:
        return {
            # warmup metrics
            "warmup.runtime(s)": self.trainer_state.warmup_runtime,
            "warmup.throughput(samples/s)": self.trainer_state.warmup_throughput,
            # training metrics
            "training.runtime(s)": self.trainer_state.training_runtime,
            "training.throughput(samples/s)": self.trainer_state.training_throughput,
            # overall metrics
            "overall.runtime(s)": self.trainer_state.overall_runtime,
            "overall.throughput(samples/s)": (self.trainer_state.overall_throughput),
        }


class MeasurementCallback(TrainerCallback):
    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        state.warmup_start = time.perf_counter_ns() * 1e-9
        state.overall_start = time.perf_counter_ns() * 1e-9

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == self.warmup_steps:
            state.warmup_end = time.perf_counter_ns() * 1e-9
            state.training_start = time.perf_counter_ns() * 1e-9

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        state.training_end = time.perf_counter_ns() * 1e-9
        state.overall_end = time.perf_counter_ns() * 1e-9

        state.total_training_batch_size = args.train_batch_size * args.gradient_accumulation_steps

        # warmup metrics
        state.warmup_runtime = state.warmup_end - state.warmup_start
        state.num_warmup_samples = self.warmup_steps * state.total_training_batch_size
        state.warmup_throughput = state.num_warmup_samples / state.warmup_runtime
        state.warmup_steps_per_second = self.warmup_steps / state.warmup_runtime

        # training metrics
        state.training_runtime = state.training_end - state.training_start
        state.num_training_steps = state.max_steps - self.warmup_steps
        state.num_training_samples = state.num_training_steps * state.total_training_batch_size
        state.training_throughput = state.num_training_samples / state.training_runtime
        state.training_steps_per_second = state.num_training_steps / state.training_runtime

        # overall training metrics
        state.overall_runtime = state.training_end - state.warmup_start
        state.num_overall_samples = state.num_warmup_samples + state.num_training_samples
        state.overall_throughput = state.num_overall_samples / state.overall_runtime
        state.overall_steps_per_second = state.num_overall_samples / state.overall_runtime


# def get_data_collator(task: str):
#     if task == "object-detection":
#         return object_detection_data_collator
#     else:
#         return default_data_collator


# def object_detection_data_collator(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     pixel_values = torch.stack([example["pixel_values"] for example in batch])
#     labels = [example["labels"] for example in batch]
#     return {
#         "pixel_values": pixel_values,
#         "labels": labels,
#     }
