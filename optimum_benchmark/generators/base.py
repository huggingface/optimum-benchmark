import logging
import random
import string
from abc import ABC
from typing import Dict, List, Tuple

import torch

LOGGER = logging.getLogger("generators")


class BaseGenerator(ABC):
    def __init__(self, shapes: Dict[str, int], with_labels: bool):
        self.shapes = shapes
        self.with_labels = with_labels

    def assert_not_missing_shapes(self, required_shapes: List[str]):
        for shape in required_shapes:
            assert self.shapes.get(shape, None) is not None, (
                f"{shape} either couldn't be inferred automatically from model artifacts or should be provided by the user. "
                f"Please provide it under `scenario.input_shapes.{shape}` or open an issue/PR in optimum-benchmark repository. "
            )

    @staticmethod
    def generate_constant_integers(value: int, shape: Tuple[int]):
        return torch.full(shape, value, dtype=torch.int64)

    @staticmethod
    def generate_constant_floats(value: float, shape: Tuple[int]):
        return torch.full(shape, value, dtype=torch.float32)

    @staticmethod
    def generate_random_integers(min_value: int, max_value: int, shape: Tuple[int]):
        return torch.randint(min_value, max_value, shape)

    @staticmethod
    def generate_random_floats(min_value: float, max_value: float, shape: Tuple[int]):
        return torch.rand(shape) * (max_value - min_value) + min_value

    @staticmethod
    def generate_ranges(start: int, stop: int, shape: Tuple[int]):
        return torch.arange(start, stop).repeat(shape[0], 1)

    @staticmethod
    def generate_random_strings(num_seq: int) -> List[str]:
        return [
            "".join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(10, 100)))
            for _ in range(num_seq)
        ]

    def __call__(self):
        raise NotImplementedError("Generator must implement __call__ method")
