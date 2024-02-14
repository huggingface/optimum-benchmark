from typing import List
from statistics import mean, stdev


def compute_max(values: List[int]) -> int:
    return max(values) if len(values) > 0 else 0


def compute_mean(values: List[int]) -> float:
    return mean(values) if len(values) > 0 else 0.0


def compute_stdev(values: List[int]) -> float:
    return stdev(values) if len(values) > 1 else 0.0
