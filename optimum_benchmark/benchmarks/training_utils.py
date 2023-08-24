from typing import Any, Dict, TYPE_CHECKING
from dataclasses import dataclass
import time

from transformers import default_data_collator
from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainerState, TrainingArguments, TrainerControl


@dataclass
class MeasurementCallback(TrainerCallback):
    warmup_steps: int

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if state.max_steps <= self.warmup_steps:
            # This check is here because max_steps is set only once the training
            # is launched, thus we can not check before calling trainer.train().
            raise ValueError(
                f"Total training steps {state.max_steps} is smaller "
                "than the number of warmup steps {self.warmup_steps}. "
                "Please increase the total number of steps (for example by "
                "increasing the dataset size)."
            )

        state.warmup_start = time.time_ns() * 1e-9
        state.overall_train_start = time.time_ns() * 1e-9

    def on_step_begin(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        if state.global_step == self.warmup_steps:
            state.warmup_end = time.time_ns() * 1e-9
            state.training_start = time.time_ns() * 1e-9
        elif state.global_step > state.max_steps - 1:
            raise ValueError("global_step > state.max_steps - 1")

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs,
    ):
        state.training_end = time.time_ns() * 1e-9
        state.overall_train_end = time.time_ns() * 1e-9

        state.total_train_batch_size = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        # warmup metrics
        state.warmup_runtime = state.warmup_end - state.warmup_start
        state.num_warmup_samples = self.warmup_steps * state.total_train_batch_size
        state.warmup_samples_per_second = (
            state.num_warmup_samples / state.warmup_runtime
        )
        # state.warmup_steps_per_second = self.warmup_steps / state.warmup_runtime

        # training metrics
        state.train_runtime = state.training_end - state.training_start
        state.num_train_steps = state.max_steps - self.warmup_steps
        state.num_train_samples = state.num_train_steps * state.total_train_batch_size
        state.train_samples_per_second = state.num_train_samples / state.train_runtime
        # state.train_steps_per_second = state.num_train_steps / state.train_runtime

        # overall training metrics
        state.overall_train_runtime = state.training_end - state.warmup_start
        state.overall_train_samples_per_second = (
            state.num_train_samples / state.overall_train_runtime
        )
        # state.overall_train_steps_per_second = (
        #     state.num_train_steps / state.overall_train_runtime
        # )


def get_data_collator(task: str) -> callable:
    if task == "object-detection":
        return object_detection_data_collator
    else:
        return default_data_collator


def object_detection_data_collator(batch) -> Dict[str, Any]:
    import torch

    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = [example["labels"] for example in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }
