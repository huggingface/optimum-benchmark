import torch
from typing import Dict
from torch import Tensor

from transformers import PretrainedConfig


def parse_pretrained_config(pretrained_config: PretrainedConfig) -> Dict[str, int]:
    vocab_size = (
        pretrained_config.vocab_size
        if hasattr(pretrained_config, "vocab_size")
        else None
    )
    num_labels = (
        pretrained_config.num_labels
        if hasattr(pretrained_config, "num_labels")
        else None
    )

    height = (
        pretrained_config.image_size[0]
        if hasattr(pretrained_config, "image_size")
        and type(pretrained_config.image_size) == tuple
        else (
            pretrained_config.image_size
            if hasattr(pretrained_config, "image_size")
            else None
        )
    )
    width = (
        pretrained_config.image_size[1]
        if hasattr(pretrained_config, "image_size")
        and type(pretrained_config.image_size) == tuple
        else (
            pretrained_config.image_size
            if hasattr(pretrained_config, "image_size")
            else None
        )
    )
    num_channels = (
        pretrained_config.num_channels
        if hasattr(pretrained_config, "num_channels")
        else None
    )
    return {
        "vocab_size": vocab_size,
        "num_labels": num_labels,
        "height": height,
        "width": width,
        "num_channels": num_channels,
    }


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


def generate_sequence_labels(num_labels: int, batch_size: int) -> Tensor:
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
    batch_size: int, num_labels: int, num_boxes: int
) -> Tensor:
    return torch.randint(
        0,
        num_labels,
        (
            batch_size,
            num_boxes,
        ),
    )
