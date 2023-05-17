from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List

from src.backend.base import Backend
from src.input.base import InputGenerator
from src.benchmark.base import Benchmark, BenchmarkConfig

BENCHMARK_NAME = "profiling"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class ProfilingConfig(BenchmarkConfig):
    name: str = BENCHMARK_NAME

    warmup_runs: int = 5
    benchmark_duration: int = 5


class ProfilingBenchmark(Benchmark):
    NAME = BENCHMARK_NAME

    def __init__(self, model: str, task: str, device: str):
        super().__init__(model, task, device)

    def configure(self, config: ProfilingConfig):
        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

    def run(self, backend: Backend, input_generator: InputGenerator) -> None:
        LOGGER.info(f"Generating dummy input")
        dummy_inputs = input_generator.generate()

        LOGGER.info(f"Running profiling benchmark")
        self.profiling_results = backend.run_profiling(
            dummy_inputs, self.warmup_runs, self.benchmark_duration)

    def save_results(self, path: str = '') -> None:
        LOGGER.info('Saving profiling results')
        self.profiling_results.to_csv(path + 'profiling_results.csv')
