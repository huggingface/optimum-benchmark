from dataclasses import dataclass
from logging import getLogger
from typing import List


import statistics
from pandas import DataFrame

from src.dummy_input_generator import DummyInputGenerator
from src.backend.base import Backend
from src.benchmark.base import Benchmark, BenchmarkConfig

BENCHMARK_NAME = "inference"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = BENCHMARK_NAME

    warmup_runs: int = 5
    benchmark_duration: int = 5


class InferenceBenchmark(Benchmark):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

        self.inference_latencies: List[float] = []
        self.dummy_input_generator = DummyInputGenerator(
            self.model, self.task, self.device
        )

    def configure(self, config: InferenceConfig):
        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

    def run(self, backend: Backend) -> None:
        dummy_inputs = self.dummy_input_generator.generate()
        self.inference_latencies.extend(
            backend.run_inference(
                dummy_inputs, self.warmup_runs, self.benchmark_duration
            )
        )

    @property
    def results(self) -> DataFrame:
        LOGGER.info("Generating inference results")
        return DataFrame(
            {
                "Model latency mean (s)": statistics.mean(self.inference_latencies)
                if len(self.inference_latencies) > 0
                else float("nan"),
                "Model latency median (s)": statistics.median(self.inference_latencies)
                if len(self.inference_latencies) > 0
                else float("nan"),
                "Model latency stdev (s)": statistics.stdev(self.inference_latencies)
                if len(self.inference_latencies) > 1
                else float("nan"),
                "Model Throughput (s^-1)": len(self.inference_latencies)
                / self.benchmark_duration,
            },
            index=[0],
        )

    @property
    def objective(self) -> float:
        return (
            statistics.mean(self.inference_latencies)
            if len(self.inference_latencies) > 0
            else float("inf")
        )

    def save(self, path: str = "") -> None:
        LOGGER.info("Saving inference results")
        self.results.to_csv(path + "inference_results.csv")
