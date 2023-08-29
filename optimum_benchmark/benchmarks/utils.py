import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict

from transformers import TrainerCallback, default_data_collator

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


def extract_three_significant_digits(x: float) -> float:
    return float(f"{x:.3g}")


def three_significant_digits_wrapper(func: Callable[..., float]) -> Callable[..., float]:
    def wrapper(*args, **kwargs):
        return extract_three_significant_digits(func(*args, **kwargs))

    return wrapper


@dataclass
class MeasurementCallback(TrainerCallback):
    warmup_steps: int

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if state.max_steps <= self.warmup_steps:
            # This check is here because max_steps is set only once the training
            # is launched, thus we can not check before calling trainer.train().
            raise ValueError(
                f"Total training steps {state.max_steps} is smaller "
                f"than the number of warmup steps {self.warmup_steps}. "
                "Please increase the total number of steps (for example by "
                "increasing the dataset size)."
            )

        state.warmup_start = time.time_ns() * 1e-9
        state.overall_training_start = time.time_ns() * 1e-9

    def on_step_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if state.global_step == self.warmup_steps:
            state.warmup_end = time.time_ns() * 1e-9
            state.training_start = time.time_ns() * 1e-9
        elif state.global_step > state.max_steps - 1:
            raise ValueError("global_step > state.max_steps - 1")

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        state.training_end = time.time_ns() * 1e-9
        state.overall_training_end = time.time_ns() * 1e-9

        state.total_training_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        # warmup metrics
        state.warmup_runtime = state.warmup_end - state.warmup_start
        state.num_warmup_samples = self.warmup_steps * state.total_training_batch_size
        state.warmup_samples_per_second = state.num_warmup_samples / state.warmup_runtime
        state.warmup_steps_per_second = self.warmup_steps / state.warmup_runtime

        # training metrics
        state.training_runtime = state.training_end - state.training_start
        state.num_training_steps = state.max_steps - self.warmup_steps
        state.num_training_samples = state.num_training_steps * state.total_training_batch_size
        state.training_samples_per_second = state.num_training_samples / state.training_runtime
        state.training_steps_per_second = state.num_training_steps / state.training_runtime

        # overall training metrics
        state.overall_training_runtime = state.training_end - state.warmup_start
        state.overall_training_samples_per_second = state.num_training_samples / state.overall_training_runtime
        state.overall_training_steps_per_second = state.num_training_steps / state.overall_training_runtime


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
