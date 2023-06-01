from dataclasses import dataclass
from typing import List, Tuple
from logging import getLogger

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

    profile: bool = False
    track_memory: bool = False
    is_generator: bool = False

    warmup_runs: int = 5
    benchmark_duration: int = 5


class InferenceBenchmark(Benchmark):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

        self.model_peak_memory: int = 0
        self.model_latencies: List[float] = []
        self.generation_records: List[Tuple[int, float]] = []
        self.profiling_records: List[Tuple[str, str, float]] = []

        self.dummy_input_generator = DummyInputGenerator(
            self.model, self.task, self.device
        )

    def configure(self, config: InferenceConfig):
        self.profile = config.profile
        self.track_memory = config.track_memory
        self.is_generator = config.is_generator

        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running inference")

        self._run_with_forward_latency_tracking(backend)

        if self.track_memory:
            self._run_with_memory_tracking(backend)

        if self.is_generator:
            self._run_with_generate_latency_tracking(backend)

        if self.profile:
            self._run_with_model_profile(backend)

    def _run_with_forward_latency_tracking(self, backend: Backend) -> None:
        LOGGER.info("\t+ Warming up the model")
        warmup_inputs = self.dummy_input_generator.generate(mode="forward")
        for _ in range(self.warmup_runs):
            backend.forward(warmup_inputs)

        LOGGER.info("\t+ Tracking model latency and throughput")
        forward_inputs = self.dummy_input_generator.generate(mode="forward")
        latency_tracker = LatencyTracker(device=self.device)
        for _ in latency_tracker.track(duration=self.benchmark_duration):
            outputs = backend.forward(forward_inputs)

        self.model_latencies = latency_tracker.get_tracked_latencies()
        LOGGER.info(f"\t+ Model Latency: {self.model_latency:.2e} (s)")
        LOGGER.info(f"\t+ Model Throughput: {self.model_throughput:.2f} (iter/s)")

    def _run_with_generate_latency_tracking(self, backend: Backend) -> None:
        LOGGER.info("\t+ Tracking generation throughput")
        generate_inputs = self.dummy_input_generator.generate(mode="generate")
        latency_tracker = LatencyTracker(device=self.device)
        num_generated_tokens = []
        for _ in latency_tracker.track(duration=self.benchmark_duration):
            outputs = backend.generate(generate_inputs)  # type: ignore
            num_generated_tokens.append(outputs.shape[-1])

        self.generation_records = list(
            zip(num_generated_tokens, latency_tracker.get_tracked_latencies())
        )
        LOGGER.info(
            f"\t+ Generation Throughput: {self.generation_throughput:.2f} (tok/s)"
        )

    def _run_with_memory_tracking(self, backend: Backend) -> None:
        LOGGER.info("\t+ Tracking model peak memory")
        memory_inputs = self.dummy_input_generator.generate(mode="forward")
        peak_memory_tracker = PeakMemoryTracker(device=self.device)
        with peak_memory_tracker.track(interval=self.model_latency / 10):
            outputs = backend.forward(memory_inputs)

        self.model_peak_memory = bytes_to_mega_bytes(
            peak_memory_tracker.get_tracked_peak_memory()
        )
        LOGGER.info(f"\t+ Model Peak Memory: {self.model_peak_memory} (MB)")

    def _run_with_model_profile(self, backend: Backend) -> None:
        LOGGER.info("Preparing for profiling")
        profile_inputs = self.dummy_input_generator.generate(mode="forward")
        backend.prepare_for_profiling(list(profile_inputs.keys()))
        LOGGER.info("Running profiling")
        backend.forward(profile_inputs)
        self.profiling_records = backend.pretrained_model.get_profiling_records()  # type: ignore

    @property
    def model_latency(self) -> float:
        return (
            statistics.mean(self.model_latencies)
            if len(self.model_latencies) > 0
            else float("inf")
        )

    @property
    def model_throughput(self) -> float:
        return (
            len(self.model_latencies) / self.benchmark_duration
            if self.benchmark_duration > 0
            else float("-inf")
        )

    @property
    def generation_throughput(self) -> float:
        return (
            statistics.mean([t / l for t, l in self.generation_records])
            if len(self.generation_records) > 0
            else float("-inf")
        )

    @property
    def results_df(self) -> DataFrame:
        results_dict = dict()

        results_dict["Model Latency (s)"] = self.model_latency
        results_dict["Model Throughput (iter/s)"] = self.model_throughput

        if self.is_generator:
            results_dict["Generation Throughput (tok/s)"] = self.generation_throughput

        if self.track_memory:
            results_dict["Model Peak Memory (MB)"] = self.model_peak_memory

        return DataFrame(results_dict, index=[0])

    @property
    def profile_df(self) -> DataFrame:
        return DataFrame(
            self.profiling_records,
            columns=["Node/Kernel", "Operator", "Latency (s)"],
        )

    def save(self) -> None:
        LOGGER.info("Saving inference results")
        self.results_df.to_csv("inference_results.csv")

        if self.profile:
            LOGGER.info("Saving profiling results")
            self.profile_df.to_csv("inference_profile.csv")

    @property
    def objective(self) -> float:
        return self.model_latency
