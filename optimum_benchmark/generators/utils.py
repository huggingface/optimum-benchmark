import torch
from typing import Dict, List, Optional, Union
from torch import Tensor

from transformers import (
    PretrainedConfig,
    PreTrainedTokenizer,
    ImageProcessingMixin,
    FeatureExtractionMixin,
    ProcessorMixin,
)


def get_model_config(
    pretrained_config: PretrainedConfig,
    pretrained_preprocessor: Optional[
        Union[
            PreTrainedTokenizer,
            ImageProcessingMixin,
            FeatureExtractionMixin,
            ProcessorMixin,
        ]
    ],
) -> Dict[str, int]:
    pretrained_config_dict = parse_config(pretrained_config)
    pretrained_preprocessor_dict = parse_config(pretrained_preprocessor)

    model_config = {k: v for k, v in pretrained_config_dict.items() if v is not None}
    model_config.update(
        {k: v for k, v in pretrained_preprocessor_dict.items() if v is not None}
    )

    return model_config


def parse_config(
    pretrained_config: Optional[
        Union[
            PretrainedConfig,
            PreTrainedTokenizer,
            ImageProcessingMixin,
            FeatureExtractionMixin,
            ProcessorMixin,
        ]
    ],
) -> Dict[str, int]:
    if hasattr(pretrained_config, "vocab_size"):
        vocab_size = pretrained_config.vocab_size
    else:
        vocab_size = None

    if hasattr(pretrained_config, "num_labels"):
        num_labels = pretrained_config.num_labels
    else:
        num_labels = None

    if hasattr(pretrained_config, "num_queries"):
        num_queries = pretrained_config.num_queries
    else:
        num_queries = None

    if hasattr(pretrained_config, "image_size"):
        if type(pretrained_config.image_size) in [int, float]:
            height = pretrained_config.image_size
            width = pretrained_config.image_size
        elif type(pretrained_config.image_size) in [list, tuple]:
            height = pretrained_config.image_size[0]
            width = pretrained_config.image_size[1]
        elif type(pretrained_config.image_size) == dict:
            height = list(pretrained_config.image_size.values())[0]
            width = list(pretrained_config.image_size.values())[1]

    elif hasattr(pretrained_config, "size"):
        if type(pretrained_config.size) in [int, float]:
            height = pretrained_config.size
            width = pretrained_config.size
        elif type(pretrained_config.size) in [list, tuple]:
            height = pretrained_config.size[0]
            width = pretrained_config.size[1]
        elif type(pretrained_config.size) == dict:
            height = list(pretrained_config.size.values())[0]
            width = list(pretrained_config.size.values())[1]
    else:
        height = None
        width = None

    if hasattr(pretrained_config, "num_channels"):
        num_channels = pretrained_config.num_channels
    else:
        num_channels = None

    return {
        "vocab_size": vocab_size,
        "num_labels": num_labels,
        "num_queries": num_queries,
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
