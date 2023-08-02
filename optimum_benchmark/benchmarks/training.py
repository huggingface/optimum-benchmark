from dataclasses import dataclass
from omegaconf import DictConfig
from typing import List, Tuple
from logging import getLogger
from pandas import DataFrame
import torch

from optimum_benchmark.backends.base import Backend
from optimum_benchmark.trackers.memory import MemoryTracker
from optimum_benchmark.trackers.latency import LatencyTracker
from optimum_benchmark.benchmarks.base import Benchmark, BenchmarkConfig


LOGGER = getLogger("training")


@dataclass
class TrainingConfig(BenchmarkConfig):
    name: str = "training"
    _target_: str = "optimum_benchmark.benchmarks.training.TrainingBenchmark"

    # dataset options
    dataset_size: int = 500
    sequence_length: int = 16

    # training options
    batch_size: int = 32
    epochs: int = 10

    # optimizer options
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0005

    # data loader options
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True


class TrainingBenchmark(Benchmark):
    def __init__(self):
        super().__init__()

        training_throughput: float = 0
        training_runtime: float = 0

    def configure(self, config: TrainingConfig):
        super().configure(config)

        from datasets import Dataset

        self.training_dataset = {
            "input_ids": torch.randint(
                100, 30000, (config.dataset_size, config.sequence_length)
            ),
            "labels": torch.randint(0, 1, (config.dataset_size,)),
        }

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running training benchmark")

        backend.prepare_for_training(self.training_dataset)
        results = backend.train().metrics

        self.training_throughput = results["train_samples_per_second"]
        self.training_runtime = results["train_runtime"]

    def get_results_df(self) -> DataFrame:
        results_dict = dict()

        results_dict["training_throughput"] = self.training_throughput
        results_dict["training_runtime"] = self.training_runtime

        return DataFrame(results_dict, index=[0])

    def save(self) -> None:
        LOGGER.info("Saving training results")
        results_df = self.get_results_df()
        results_df.to_csv("training_results.csv")


def significant_figures(x):
    return float(f"{x:.3g}")
