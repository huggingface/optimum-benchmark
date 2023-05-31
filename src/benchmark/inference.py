from dataclasses import dataclass
from functools import partial
from logging import getLogger
from multiprocessing import Process, Queue
from typing import Callable, List, Tuple

import gc
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
    inference_mode: str = "forward"


class InferenceBenchmark(Benchmark):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

        self.inference_memory: int = 0
        self.inference_latencies: List[float] = []
        self.profiling_records: List[Tuple[str, str, float]] = []

        self.dummy_input_generator = DummyInputGenerator(
            self.model, self.task, self.device
        )

    def configure(self, config: InferenceConfig):
        self.profiling = config.profiling
        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

        self.inference_mode = config.inference_mode
        self.dummy_input_generator.configure(self.inference_mode)

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running inference")

        self.dummy_inputs = self.dummy_input_generator.generate()
        self.inference_func = getattr(backend, self.inference_mode)

        self.run_with_latency_tracking()
        self.run_with_memory_tracking()

        if self.profiling:
            self.run_with_profiling(backend)

    def run_with_latency_tracking(self) -> None:
        LOGGER.info("\t+ Warming up the model")
        for _ in range(self.warmup_runs):
            self.inference_func(self.dummy_inputs)

        LOGGER.info("\t+ Tracking latencies")
        latency_tracker = LatencyTracker(device=self.device)
        for _ in latency_tracker.track(duration=self.benchmark_duration):
            self.inference_func(self.dummy_inputs)
        self.inference_latencies = latency_tracker.get_tracked_latencies()
        LOGGER.info(f"\t+ Latency: {statistics.mean(self.inference_latencies)}s")

    def run_with_memory_tracking(self) -> None:
        LOGGER.info("\t+ Tracking peak memory")
        peak_memory_tracker = PeakMemoryTracker(device=self.device)
        with peak_memory_tracker.track(interval=self.inference_latencies[-1] / 10):
            self.inference_func(self.dummy_inputs)
        self.inference_memory = peak_memory_tracker.get_tracked_peak_memory()
        LOGGER.info(f"\t+ Memory: {bytes_to_mega_bytes(self.inference_memory)}MB")

    def run_with_profiling(self, backend: Backend) -> None:
        LOGGER.info("Preparing for profiling")
        backend.prepare_for_profiling(self.dummy_input_generator.input_names)
        LOGGER.info("Running profiling")
        self.inference_func(self.dummy_inputs)
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
                "throughput(s^-1)": len(self.inference_latencies)
                / self.benchmark_duration,
                "memory.peak(MB)": bytes_to_mega_bytes(self.inference_memory),
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


def run_separate_process(func: Callable) -> Callable:
    def multi_process_func(*args, **kwargs):
        def wrapper_func(queue: Queue, *args):
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                LOGGER.error(e)
                print(e)
                result = "N/A"
            queue.put(result)

        queue = Queue()
        p = Process(target=wrapper_func, args=[queue] + list(args))
        p.start()
        result = queue.get()
        p.join()
        return result

    return multi_process_func
