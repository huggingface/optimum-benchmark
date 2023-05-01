#  Copyright 2021 Hugging Face Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import List, Optional
from dataclasses import dataclass, field
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
    def perfs(self) -> DataFrame:
        return DataFrame({
            "mean_latency": self.runs_duration / self.num_runs,
            "throughput": self.throughput
        }, index=[0])

    @property
    def details(self) -> DataFrame:
        return DataFrame({
            "latencies": self.latencies,
        }, index=range(self.num_runs))

    @contextmanager
    def track_cpu_latency(self):
        
        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()

        latency_ns = end - start
        latency = latency_ns / 1e9

        self.latencies.append(latency)

        LOGGER.debug(
            f'Tracked CPU latency took: {latency}s)')

    @contextmanager
    def track_cuda_latency(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        yield
        end_event.record()

        torch.cuda.synchronize()  # Wait for the events to be recorded!

        latency_ms = start_event.elapsed_time(end_event)
        latency = latency_ms / 1e3

        self.latencies.append(latency)

        LOGGER.debug(
            f'Tracked CUDA latency took: {latency}s)')

    def finalize(self, benchmark_duration: int):
        self.throughput = self.num_runs / benchmark_duration

    @staticmethod
    def merge(benchmarks: List['Benchmark']) -> 'Benchmark':
        latencies, throughputs = [], []

        for b in benchmarks:

            assert len(b.latencies) > 0, \
                "Empty benchmark (0 latency measurements recorded)"
            assert b.throughput > 0., \
                f"Benchmark has not been finalized, throughput < 0 ({b.throughput})"

            latencies += b.latencies
            throughputs.append(b.throughput)

        # Return all the latencies measured and the mean throughput over all instances
        mean_throughput = sum(throughputs) / len(throughputs)

        return Benchmark(
            latencies,
            mean_throughput
        )
