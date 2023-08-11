from transformers import default_data_collator
from torch import Tensor
from typing import Dict
import torch


def object_detection_data_collator(batch) -> Dict[str, Tensor]:
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = [example["labels"] for example in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def get_data_collator(task: str) -> callable:
    if task == "object-detection":
        return object_detection_data_collator
    else:
        return default_data_collator
