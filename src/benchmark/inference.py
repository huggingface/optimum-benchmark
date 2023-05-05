from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from typing import Dict

import time
import torch
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


class InferenceBenchmark(Benchmark):
    NAME = BENCHMARK_NAME

    def __init__(self, model: str, task: str, device: str):
        super().__init__(model, task, device)

    def configure(self, config: InferenceConfig):
        self.latencies = []
        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

    def execute(self, backend: Backend, input_generator: InputGenerator) -> None:
        inputs = input_generator.generate()
        # TODO: get rid of this
        for input_name, input_tensor in inputs.items():
            inputs[input_name] = input_tensor.to(self.device)

        # Warmup
        for _ in trange(self.warmup_runs, desc="Warming up"):
            backend.run_inference(inputs)

        # Run benchmark
        while sum(self.latencies) < self.benchmark_duration:
            with self.track_latency(device=self.device):
                backend.run_inference(inputs)

    @contextmanager
    def track_latency(self, device: str):
        if device == 'cpu':
            start = time.perf_counter_ns()
            yield
            end = time.perf_counter_ns()

            latency_ns = end - start
            latency = latency_ns / 1e9

            self.latencies.append(latency)
            LOGGER.debug(f'Tracked CPU latency took: {latency}s)')

        elif device == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            yield
            end_event.record()
            torch.cuda.synchronize()

            latency_ms = start_event.elapsed_time(end_event)
            latency = latency_ms / 1e3

            self.latencies.append(latency)
            LOGGER.debug(f'Tracked CUDA latency took: {latency}s)')
        else:
            raise ValueError(f"Unsupported device type {device}")

    @property
    def throughput(self) -> float:
        return self.num_runs / self.benchmark_duration

    @property
    def num_runs(self) -> int:
        return len(self.latencies)

    @property
    def runs_duration(self) -> float:
        return sum(self.latencies)

    @property
    def details_df(self) -> DataFrame:
        df = DataFrame(self.latencies, columns=["latency"])
        return df

    @property
    def stats_dict(self) -> Dict:
        return {
            # only depend on the latencies
            "latency.mean": self.details_df.latency.mean(),
            "latency.std": self.details_df.latency.std(),
            # depend the benchmark duration
            "throughput": self.throughput
        }
