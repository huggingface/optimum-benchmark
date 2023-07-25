from dataclasses import dataclass
from omegaconf import DictConfig
from typing import List, Tuple
from logging import getLogger
from pandas import DataFrame
import statistics


from optimum_benchmark.backends.base import Backend
from optimum_benchmark.trackers.memory import MemoryTracker
from optimum_benchmark.trackers.latency import LatencyTracker
from optimum_benchmark.benchmarks.base import Benchmark, BenchmarkConfig


LOGGER = getLogger("inference")


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = "inference"
    _target_: str = "optimum_benchmark.benchmarks.inference.InferenceBenchmark"

    # run options
    memory: bool = False
    profile: bool = False

    # loop options
    warmup_runs: int = 10
    benchmark_duration: int = 10

    # input options
    input_shapes: DictConfig = DictConfig(
        {
            "batch_size": 1,
            # text
            "sequence_length": 16,
            "num_choices": 4,
            # image
            "width": 64,
            "height": 64,
            "num_channels": 3,
            "point_batch_size": 3,
            "nb_points_per_image": 2,
            # audio
            "feature_size": 80,
            "nb_max_frames": 3000,
            "audio_sequence_length": 16000,
        }
    )
    # output options
    new_tokens: int = 100


class InferenceBenchmark(Benchmark):
    def __init__(self):
        super().__init__()

        # initialize inference results
        self.forward_peak_memory: int = 0
        self.forward_latencies: List[float] = []
        self.generate_latencies: List[float] = []
        # might be better to seperate profiling benchmark from inference benchmark
        # kernel/node, op, time
        self.forward_profile: List[Tuple[str, str, float]] = []

    def configure(self, config: InferenceConfig):
        super().configure(config)
        self.memory = config.memory
        self.profile = config.profile

        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

        self.new_tokens = config.new_tokens
        self.input_shapes = config.input_shapes

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running inference benchmark")
        if self.memory:
            # if requested, run memory tracking
            self.run_memory_tracking(backend)

        # ALWAYS run forward pass
        self.run_forward_tracking(backend)

        if backend.can_generate():
            # if possible, run generation pass
            self.run_generate_tracking(backend)
            self.can_generate = True
        else:
            self.can_generate = False

        if self.profile:
            self.run_forward_profile(backend)

    def run_forward_tracking(self, backend: Backend) -> None:
        forward_input, forward_input_shapes = backend.generate_dummy_input(
            mode="forward", **self.input_shapes  # type: ignore
        )
        backend.prepare_for_forward(forward_input_shapes)

        LOGGER.info("\t+ Warming up the forward pass")
        for _ in range(self.warmup_runs):
            _ = backend.forward(forward_input)

        LOGGER.info("\t+ Tracking forward pass latency and throughput")
        latency_tracker = LatencyTracker(device=backend.device)
        while sum(latency_tracker.get_latencies()) < self.benchmark_duration:
            with latency_tracker.track():
                _ = backend.forward(forward_input)

        self.forward_latencies = latency_tracker.get_latencies()
        LOGGER.info(f"\t+ Forward pass latency: {self.forward_latency:.2e} (s)")
        LOGGER.info(
            f"\t+ Forward pass throughput: {self.forward_throughput:.2f} (samples/s)"
        )

    def run_generate_tracking(self, backend: Backend) -> None:
        generate_input, _ = backend.generate_dummy_input(
            mode="generate", **self.input_shapes  # type: ignore
        )

        LOGGER.info("\t+ Warming up the generation pass")
        _ = backend.generate(generate_input, new_tokens=self.new_tokens)

        LOGGER.info("\t+ Tracking generation latency and throughput")
        latency_tracker = LatencyTracker(device=backend.device)
        while sum(latency_tracker.get_latencies()) < self.benchmark_duration:
            with latency_tracker.track():
                _ = backend.generate(generate_input, new_tokens=self.new_tokens)

        self.generate_latencies = latency_tracker.get_latencies()
        LOGGER.info(f"\t+ Generation pass latency: {self.generate_latency:.2e} (s)")

        LOGGER.info(
            f"\t+ Generation pass throughput: {self.generate_throughput:.2f} (tokens/s)"
        )

    def run_memory_tracking(self, backend: Backend) -> None:
        memory_input, memory_input_shapes = backend.generate_dummy_input(
            mode="forward", **self.input_shapes  # type: ignore
        )
        backend.prepare_for_forward(memory_input_shapes)

        LOGGER.info("\t+ Tracking forward pass peak memory")
        memory_tracker = MemoryTracker(device=backend.device)
        with memory_tracker.track(interval=self.benchmark_duration // 100):
            outputs = backend.forward(memory_input)

        self.forward_peak_memory = memory_tracker.get_peak_memory()
        LOGGER.info(f"\t+ Forward pass peak memory: {self.forward_peak_memory} (MB)")

    def run_forward_profile(self, backend: Backend) -> None:
        profile_input, profile_input_shapes = backend.generate_dummy_input(
            mode="forward", **self.input_shapes  # type: ignore
        )
        backend.prepare_for_forward(profile_input_shapes)
        backend.prepare_for_profiling(list(profile_input.keys()))

        LOGGER.info("\t+ Running profiling")
        backend.forward(profile_input)

        self.forward_profile = backend.pretrained_model.get_forward_profile()  # type: ignore

    # Metrics
    @property
    def forward_latency(self) -> float:
        return significant_figures(statistics.mean(self.forward_latencies))

    @property
    def forward_throughput(self) -> float:
        return significant_figures(self.input_shapes.batch_size / self.forward_latency)

    @property
    def generate_latency(self) -> float:
        return significant_figures(statistics.mean(self.generate_latencies))

    @property
    def generate_throughput(self) -> float:
        return significant_figures(
            self.new_tokens * self.input_shapes.batch_size / self.generate_latency
        )

    def get_results_df(self) -> DataFrame:
        results_dict = dict()

        results_dict["forward.latency(s)"] = self.forward_latency
        results_dict["forward.throughput(samples/s)"] = self.forward_throughput

        if self.memory:
            results_dict["forward.peak_memory(MB)"] = self.forward_peak_memory

        if self.can_generate:
            results_dict["generate.latency(s)"] = self.generate_latency
            results_dict["generate.throughput(tokens/s)"] = self.generate_throughput

        return DataFrame(results_dict, index=[0])

    def get_profile_df(self) -> DataFrame:
        return DataFrame(
            self.forward_profile,
            columns=["Node/Kernel", "Operator", "Latency (s)"],
        )

    def save(self) -> None:
        LOGGER.info("Saving inference results")
        results_df = self.get_results_df()
        results_df.to_csv("inference_results.csv")

        if self.profile:
            LOGGER.info("Saving profiling results")
            profile_df = self.get_profile_df()
            profile_df.to_csv("inference_profile.csv")


def significant_figures(x):
    return float(f"{x:.3g}")
