from dataclasses import dataclass
from logging import getLogger

from pandas import DataFrame

from src.backend.base import Backend
from src.benchmark.inference import InferenceBenchmark, InferenceConfig

BENCHMARK_NAME = "profiling"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class ProfilingConfig(InferenceConfig):
    name: str = BENCHMARK_NAME

    warmup_runs: int = 5
    benchmark_duration: int = 5


class ProfilingBenchmark(InferenceBenchmark):
    def run(self, backend: Backend) -> None:
        LOGGER.info(f"Generating dummy input")
        dummy_inputs = self.generate_dummy_inputs()

        LOGGER.info(f"Running profiling benchmark")
        self.profiling_results = backend.run_profiling(
            dummy_inputs, self.warmup_runs, self.benchmark_duration)

    @property
    def results(self) -> DataFrame:
        return self.profiling_results

    def save(self, path: str = '') -> None:
        LOGGER.info('Saving profiling results')
        self.profiling_results.to_csv(path + 'profiling_results.csv')
