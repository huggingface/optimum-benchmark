from dataclasses import dataclass
from logging import getLogger
from typing import List, Tuple


import statistics
from pandas import DataFrame

from src.backend.base import Backend
from src.benchmark.base import Benchmark, BenchmarkConfig
from src.dummy_input_generator import DummyInputGenerator
from src.trackers import LatencyTracker


BENCHMARK_NAME = "inference"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = BENCHMARK_NAME

    profiling: bool = False
    warmup_runs: int = 5
    benchmark_duration: int = 5


class InferenceBenchmark(Benchmark):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

        self.profiling_latencies: List[Tuple[str, str, float]] = []
        self.inference_latencies: List[float] = []
        self.dummy_input_generator = DummyInputGenerator(
            self.model, self.task, self.device
        )

    def configure(self, config: InferenceConfig):
        self.profiling = config.profiling
        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

    def run(self, backend: Backend) -> None:
        dummy_inputs = self.dummy_input_generator.generate()

        LOGGER.info("Running inference")

        LOGGER.info("\t+ Warming up the model")
        for _ in range(self.warmup_runs):
            backend.inference(dummy_inputs)

        LOGGER.info("\t+ Tracking latencies")
        tracker = LatencyTracker(device=self.device)
        while sum(tracker.tracked_latencies) < self.benchmark_duration:
            with tracker.track_latency():
                backend.inference(dummy_inputs)
        self.inference_latencies = tracker.tracked_latencies

        if self.profiling:
            backend.prepare_for_profiling(self.dummy_input_generator.input_names)

            LOGGER.info("Running profiling run")
            backend.inference(dummy_inputs)
            self.profiling_latencies = backend.pretrained_model.get_profiling_results()  # type: ignore

    @property
    def inference_results(self) -> DataFrame:
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
    def profiling_results(self) -> DataFrame:
        return DataFrame(
            self.profiling_latencies,
            columns=["Node/Kernel", "Operator", "Latency (s)"],
        )

    def save(self, path: str = "") -> None:
        LOGGER.info("Saving inference results")
        self.inference_results.to_csv(path + "inference_results.csv")

        if self.profiling:
            LOGGER.info("Saving profiling results")
            self.profiling_results.to_csv(path + "profiling_results.csv")

    @property
    def objective(self) -> float:
        return (
            statistics.mean(self.inference_latencies)
            if len(self.inference_latencies) > 0
            else float("inf")
        )
