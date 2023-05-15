from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger

import time
import torch
import statistics
from tqdm import trange
from pandas import DataFrame

from src.input.base import InputGenerator
from src.backend.base import Backend
from src.benchmark.base import Benchmark, BenchmarkConfig

BENCHMARK_NAME = "inference"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = BENCHMARK_NAME

    # Number of forward pass to run before recording any performance counters.
    warmup_runs: int = 5
    # Duration in seconds the benchmark will collect performance counters
    benchmark_duration: int = 5
    # Whether to use symbolic profiling or not
    symbolic_profiling: bool = False


class InferenceBenchmark(Benchmark):
    NAME = BENCHMARK_NAME

    def __init__(self, model: str, task: str, device: str):
        super().__init__(model, task, device)

    def configure(self, config: InferenceConfig):
        self.model_latencies = []
        self.nodes_latencies = {}

        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration
        self.symbolic_profiling = config.symbolic_profiling

    def populate(self, backend: Backend, input_generator: InputGenerator) -> None:
        dummy_inputs = input_generator.generate()

        # Warmup the backend
        LOGGER.info("Warming up the backend")
        for _ in trange(self.warmup_runs, desc="Warming up"):
            backend.run_inference_with_model(dummy_inputs)

        # Track inference latency
        LOGGER.info("Running inference")
        while sum(self.model_latencies) < self.benchmark_duration:
            with self.track_latency():
                backend.run_inference_with_model(dummy_inputs)

        if self.symbolic_profiling:
            # Convert to symbolic one
            LOGGER.info("Converting model to symbolic")
            backend.symbolic_trace_model(
                dummy_inputs, warmup_runs=self.warmup_runs)

            # Track nodes latency
            LOGGER.info("Running symbolic profiling")
            while sum(backend.profiler.model_latencies) < self.benchmark_duration:
                backend.run_inference_with_profiler(dummy_inputs)

            # Save nodes latencies
            self.nodes_latencies.update(backend.profiler.nodes_latencies)

    def save_results(self, path: str = '') -> None:
        self.stats_df.to_csv(path + 'stats.csv')
        if self.nodes_latencies:
            self.profiling_df.to_csv(path + 'profiling.csv')

    @property
    def profiling_df(self) -> DataFrame:
        return DataFrame(
            [
                {'Node': str(node), 'Op': str(node.op),
                 'Node latency mean (s)': statistics.mean(node_latency),
                 'Node latency std (s)': statistics.stdev(node_latency)}
                for node, node_latency in self.nodes_latencies.items()
            ],
        ).sort_values(
            by=['Node latency mean (s)'], ascending=False
        ).reset_index(drop=True)

    @property
    def stats_df(self) -> DataFrame:
        return DataFrame({
            "Model latency mean (s)": statistics.mean(self.model_latencies),
            "Model latency std (s)": statistics.stdev(self.model_latencies),
            "Model Throughput (s^-1)": len(self.model_latencies) / self.benchmark_duration
        }, index=[0])

    @contextmanager
    def track_latency(self):
        if self.device == 'cuda':
            yield from self._cuda_latency_tracker()
        else:
            yield from self._cpu_latency_tracker()

    def _cpu_latency_tracker(self):
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9

        self.model_latencies.append(latency)
        LOGGER.debug(f'Tracked CPU latency took: {latency}s)')

    def _cuda_latency_tracker(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        yield
        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        latency = latency_ms / 1e3

        self.model_latencies.append(latency)
        LOGGER.debug(f'Tracked CUDA latency took: {latency}s)')
