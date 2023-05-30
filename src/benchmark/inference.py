from dataclasses import dataclass
from logging import getLogger
from typing import List, Tuple

import statistics
from pandas import DataFrame

from src.backend.base import Backend
from src.benchmark.base import Benchmark, BenchmarkConfig

from src.dummy_input_generator import DummyInputGenerator
from src.trackers.memory import PeakMemoryTracker
from src.trackers.latency import LatencyTracker
from src.utils import bytes_to_mega_bytes

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

        self.inference_latencies: List[float] = []
        self.inference_peak_memory: List[int] = []
        self.profiling_records: List[Tuple[str, str, float]] = []

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
        latency_tracker = LatencyTracker(device=self.device)
        for _ in latency_tracker.track(duration=self.benchmark_duration):
            backend.inference(dummy_inputs)

        self.inference_latencies = latency_tracker.get_tracked_latencies()
        average_latency = statistics.mean(self.inference_latencies)
        LOGGER.info(f"\t+ Average latency: {average_latency}s")

        LOGGER.info("\t+ Tracking peak memory")
        peak_memory_tracker = PeakMemoryTracker(device=self.device)
        with peak_memory_tracker.track(interval=average_latency / 10):
            backend.inference(dummy_inputs)

        self.inference_peak_memory = peak_memory_tracker.get_tracked_peak_memories()
        average_peak_memory = self.inference_peak_memory[0]
        LOGGER.info(
            f"\t+ Average peak memory: {bytes_to_mega_bytes(average_peak_memory)}MB"
        )

        if self.profiling:
            LOGGER.info("Preparing for profiling")
            backend.prepare_for_profiling(self.dummy_input_generator.input_names)

            LOGGER.info("Running profiling")
            backend.inference(dummy_inputs)
            self.profiling_records = backend.pretrained_model.get_profiling_records()  # type: ignore

    @property
    def inference_results(self) -> DataFrame:
        return DataFrame(
            {
                "latency.mean(s)": statistics.mean(self.inference_latencies)
                if len(self.inference_latencies) > 0
                else float("nan"),
                "latency.median(s)": statistics.median(self.inference_latencies)
                if len(self.inference_latencies) > 0
                else float("nan"),
                "latency.stdev(s)": statistics.stdev(self.inference_latencies)
                if len(self.inference_latencies) > 1
                else float("nan"),
                "memory.peak(MB)": bytes_to_mega_bytes(
                    statistics.mean(self.inference_peak_memory)
                ),
                "throughput(s^-1)": len(self.inference_latencies)
                / self.benchmark_duration,
            },
            index=[0],
        )

    @property
    def profiling_results(self) -> DataFrame:
        return DataFrame(
            self.profiling_records,
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
