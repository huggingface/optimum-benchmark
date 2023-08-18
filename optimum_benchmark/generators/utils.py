import torch
from torch import Tensor
from typing import List


def generate_input_ids(
    vocab_size: int, batch_size: int, sequence_length: int
) -> Tensor:
    return torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length,
        ),
    )


def generate_token_labels(
    num_labels: int, batch_size: int, sequence_length: int
) -> Tensor:
    return torch.randint(
        0,
        num_labels,
        (
            batch_size,
            sequence_length,
        ),
    )


def generate_sequence_labels(batch_size: int, num_labels: int = 2) -> Tensor:
    return torch.randint(
        0,
        num_labels,
        (batch_size,),
    )


def generate_token_type_ids(batch_size: int, sequence_length: int) -> Tensor:
    return torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.long,
    )


def generate_multiple_choice_input_ids(
    vocab_size: int, batch_size: int, sequence_length: int, num_choices: int
) -> Tensor:
    return torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            num_choices,
            sequence_length,
        ),
    )


def generate_multiple_choice_token_type_ids(
    batch_size: int, sequence_length: int, num_choices: int
) -> Tensor:
    return torch.zeros(
        (batch_size, num_choices, sequence_length),
        dtype=torch.long,
    )


def generate_multiple_choice_labels(batch_size: int, num_choices: int) -> Tensor:
    return torch.randint(
        0,
        num_choices,
        (batch_size,),
    )


def generate_start_positions(batch_size: int) -> Tensor:
    return torch.full((batch_size,), 0)


def generate_end_positions(batch_size: int, sequence_length: int) -> Tensor:
    return torch.full((batch_size,), sequence_length - 1)


def generate_attention_mask(input_ids_or_values: Tensor) -> Tensor:
    return torch.ones_like(input_ids_or_values)


def generate_pixel_values(
    batch_size: int, num_channels: int, height: int, width: int
) -> Tensor:
    return torch.randint(
        0,
        255,
        (
            batch_size,
            num_channels,
            height,
            width,
        ),
    )


def generate_object_detection_labels(
    batch_size: int, num_labels: int, num_queries: int
) -> List:
    return [
        {
            "class_labels": torch.randint(
                0,
                num_labels,
                (num_queries,),
            ),
            "boxes": torch.rand(
                (
                    num_queries,
                    4,
                ),
            ),
        }
        for _ in range(batch_size)
    ]


def generate_semantic_segmentation_labels(
    batch_size: int, num_labels: int, height: int, width: int
) -> Tensor:
    return torch.randint(
        0,
        num_labels,
        (
            batch_size,
            height,
            width,
        ),
    )


def generate_prompt(batch_size: int) -> Tensor:
    return ["Surrealist painting of a floating island."] * batch_size
