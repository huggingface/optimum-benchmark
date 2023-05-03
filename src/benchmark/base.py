from dataclasses import dataclass, field
from typing import List, Optional, Dict
from contextlib import contextmanager
from logging import getLogger

from pandas import DataFrame
import torch
import time

LOGGER = getLogger("benchmark")


@dataclass
class Benchmark:
    latencies: List[float] = field(default_factory=list)
    throughput: Optional[float] = float('-inf')

    @property
    def num_runs(self) -> int:
        return len(self.latencies)

    @property
    def runs_duration(self) -> float:
        return sum(self.latencies)

    @property
    def stats_dict(self) -> Dict:
        return {
            "mean_latency": self.mean_latency,
            "throughput": self.throughput
        }

    @property
    def details_df(self) -> DataFrame:
        return DataFrame(self.latencies, columns=["latency"])

    @contextmanager
    def track(self, device: str):
        if device == "cpu":
            start = time.perf_counter_ns()
            yield
            end = time.perf_counter_ns()

            latency_ns = end - start
            latency = latency_ns / 1e9

            self.latencies.append(latency)
            LOGGER.debug(f'Tracked CPU latency took: {latency}s)')

        elif device == "cuda":
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

    def finalize(self, benchmark_duration: int):
        self.throughput = self.num_runs / \
            benchmark_duration if benchmark_duration else float('-inf')
        self.mean_latency = self.runs_duration / \
            self.num_runs if self.num_runs else float('inf')
