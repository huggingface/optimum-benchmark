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

    return {
        "vocab_size": vocab_size,
        "num_labels": num_labels,
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


def generate_start_positions(batch_size: int) -> Tensor:
    return torch.full((batch_size,), 0)


def generate_end_positions(batch_size: int, sequence_length: int) -> Tensor:
    return torch.full((batch_size,), sequence_length - 1)


def generate_attention_mask(input_ids_or_values: Tensor) -> Tensor:
    return torch.ones_like(input_ids_or_values)
