from typing import Any, Dict

from ..task_utils import TEXT_GENERATION_WITH_INPUT_TEXT_TASKS, TEXT_GENERATION_WITHOUT_INPUT_TEXT_TASKS


def compute_forward_volume(input_shapes: Dict[str, Any]) -> int:
    return input_shapes["batch_size"]


def compute_prefill_volume(task: str, input_shapes: Dict[str, Any], generate_kwargs: Dict[str, Any]) -> int:
    if task in TEXT_GENERATION_WITHOUT_INPUT_TEXT_TASKS:
        return input_shapes["batch_size"] * generate_kwargs["num_return_sequences"]
    elif task in TEXT_GENERATION_WITH_INPUT_TEXT_TASKS:
        return input_shapes["batch_size"] * input_shapes["sequence_length"] * generate_kwargs["num_return_sequences"]
    else:
        raise ValueError(f"Task {task} is not supported for prefill tokens volume calculation")


def compute_decode_volume(input_shapes: Dict[str, Any], generate_kwargs: Dict[str, Any]) -> int:
    return (
        input_shapes["batch_size"] * generate_kwargs["num_return_sequences"] * (generate_kwargs["max_new_tokens"] - 1)
    )


def compute_call_volume(input_shapes: Dict[str, Any], call_kwargs: Dict[str, Any]) -> int:
    return input_shapes["batch_size"] * call_kwargs["num_images_per_prompt"]
