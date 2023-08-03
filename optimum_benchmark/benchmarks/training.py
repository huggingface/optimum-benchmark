from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments
from dataclasses import dataclass
from logging import getLogger
from pandas import DataFrame
from datasets import Dataset
import torch


from optimum_benchmark.backends.base import Backend
from optimum_benchmark.benchmarks.base import Benchmark, BenchmarkConfig


LOGGER = getLogger("training")

# resolvers
OmegaConf.register_new_resolver("is_cpu", lambda device: device == "cpu")


@dataclass
class TrainingConfig(BenchmarkConfig):
    name: str = "training"
    _target_: str = "optimum_benchmark.benchmarks.training.TrainingBenchmark"

    # dataset options
    dataset_size: int = 500
    sequence_length: int = 16

    # training options
    trainer_config: DictConfig = DictConfig(
        {
            "output_dir": "./trainer_output",
            "use_cpu": "${is_cpu:${device}}",
            # add any other training arguments here
        }
    )


class TrainingBenchmark(Benchmark):
    def __init__(self):
        super().__init__()

        self.training_throughput: float = 0
        self.training_runtime: float = 0

    def configure(self, config: TrainingConfig):
        super().configure(config)
        self.trainer_config = config.trainer_config

        self.training_dataset = Dataset.from_dict(
            {
                "input_ids": torch.randint(
                    0, 100, (config.dataset_size, config.sequence_length)
                ),
                "labels": torch.randint(0, 1, (config.dataset_size,)),
            }
        )
        self.training_dataset.set_format(
            type="torch",
            columns=["input_ids", "labels"],
        )

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running training benchmark")

        backend.prepare_for_training(
            training_dataset=self.training_dataset,
            training_arguments=self.trainer_config,
        )
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
