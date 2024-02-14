from typing import List, Literal, Union
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from functools import reduce
import time

from .utils import compute_mean, compute_stdev
from ..import_utils import is_torch_distributed_available

if is_torch_distributed_available():
    import torch.distributed

from transformers import TrainerCallback, LogitsProcessor
import torch

LOGGER = getLogger("latency")

LATENCY_UNIT = "s"
Latency_Unit_Literal = Literal["s"]
Throughput_Unit_Literal = Literal["samples/s", "tokens/s", "images/s", "steps/s"]


@dataclass
class Latency:
    unit: Latency_Unit_Literal

    mean: float
    stdev: float
    values: List[float]

    def __getitem__(self, index: int) -> float:
        if isinstance(index, slice):
            return Latency.from_values(values=self.values[index], unit=self.unit)
        else:
            return self.values[index]

    def __sub__(self, scalar: float) -> "Latency":
        if not isinstance(scalar, (int, float)):
            raise ValueError(f"Cannot subtract non-scalar value from latency: {scalar}")

        latencies = [lat - scalar for lat in self.values]

        return Latency.from_values(values=latencies, unit=self.unit)

    def __add__(self, other: "Latency") -> "Latency":
        if self.unit != other.unit:
            raise ValueError(f"Cannot add latencies with different units: {self.unit} and {other.unit}")

        return Latency.from_values(values=self.values + other.values, unit=self.unit)

    @staticmethod
    def aggregate(latencies: List["Latency"]) -> "Latency":
        if len(latencies) == 0 or all(latency is None for latency in latencies):
            return None
        elif any(latency is None for latency in latencies):
            raise ValueError("Some latency measurements are missing")

        return reduce(lambda x, y: x + y, latencies)

    @staticmethod
    def from_values(values: List[float], unit: str) -> "Latency":
        return Latency(
            mean=compute_mean(values),
            stdev=compute_stdev(values),
            values=values,
            unit=unit,
        )

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} latency: {self.mean:f} ± 2 x {self.stdev:f} ({self.unit})")


@dataclass
class Throughput:
    unit: Throughput_Unit_Literal

    mean: float
    stdev: float

    def __add__(self, other: "Throughput") -> "Throughput":
        if self.unit != other.unit:
            raise ValueError(f"Cannot add throughputs with different units: {self.unit} and {other.unit}")

        return Throughput(
            mean=self.mean + other.mean,
            stdev=(self.stdev**2 + other.stdev**2) ** 0.5,
            unit=self.unit,
        )

    @staticmethod
    def aggregate(throughputs: List["Throughput"]) -> "Throughput":
        if len(throughputs) == 0 or all(throughput is None for throughput in throughputs):
            return None
        elif any(throughput is None for throughput in throughputs):
            raise ValueError("Some throughput measurements are missing")

        return reduce(lambda x, y: x + y, throughputs)

    @staticmethod
    def from_values(values: List[float], unit: str) -> "Throughput":
        return Throughput(mean=compute_mean(values), stdev=compute_stdev(values), unit=unit)

    @staticmethod
    def from_latency(latency: "Latency", volume: int, unit: str) -> "Throughput":
        throughputs = [volume / lat if lat > 0 else 0 for lat in latency.values]
        return Throughput.from_values(values=throughputs, unit=unit)

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} throughput: {self.mean:f} ± 2 x {self.stdev:f} ({self.unit})")


class LatencyTracker:
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend

        if is_torch_distributed_available() and torch.distributed.is_initialized():
            self.distributed = True
        else:
            self.distributed = False

        self.start_events: List[Union[float, torch.cuda.Event]] = []
        self.end_events: List[Union[float, torch.cuda.Event]] = []
        self.start_time: float = time.perf_counter()

        if self.backend == "pytorch" and self.device == "cuda":
            LOGGER.info("\t+ Tracking Pytorch CUDA latency")
        else:
            LOGGER.info("\t+ Tracking CPU latency")

    def reset(self):
        self.start_time = time.perf_counter()
        self.start_events = []
        self.end_events = []

    @contextmanager
    def track(self):
        if self.backend == "pytorch" and self.device == "cuda":
            yield from self._pytorch_cuda_latency()
        else:
            yield from self._cpu_latency()

    def _pytorch_cuda_latency(self):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.start_events.append(start)

        yield

        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.end_events.append(end)

    def _cpu_latency(self):
        start = time.perf_counter()
        self.start_events.append(start)

        yield

        end = time.perf_counter()
        self.end_events.append(end)

    def get_elapsed_time(self) -> float:
        # we measured in cpu to not synchronize all events
        return time.perf_counter() - self.start_time

    def get_latency(self) -> Latency:
        if self.backend == "pytorch" and self.device == "cuda":
            # synchronize the last event to make sure it has been recorded
            self.start_events[-1].synchronize()
            self.end_events[-1].synchronize()

            latencies_list = [
                self.start_events[i].elapsed_time(self.end_events[i]) / 1e3 for i in range(len(self.start_events))
            ]
        else:
            latencies_list = [(self.end_events[i] - self.start_events[i]) for i in range(len(self.start_events))]

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)


class LatencyTrainerCallback(TrainerCallback):
    def __init__(self, device: str, backend: str) -> None:
        self.device = device
        self.backend = backend

        self.events: List[Union[float, torch.cuda.Event]] = []

    def reset(self):
        self.events = []

    def on_step_begin(self, *args, **kwargs):
        if self.device == "cuda" and self.backend == "pytorch":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.append(event)
        else:
            self.events.append(time.perf_counter())

    def on_train_end(self, *args, **kwargs):
        # one last record to measure the time of the last step
        if self.device == "cuda" and self.backend == "pytorch":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.append(event)
        else:
            self.events.append(time.perf_counter())

    def get_latency(self) -> Latency:
        if self.device == "cuda" and self.backend == "pytorch":
            # synchronize the device to make sure all events have been recorded
            torch.cuda.synchronize()
            latencies_list = [
                self.events[i - 1].elapsed_time(self.events[i]) / 1e3 for i in range(1, len(self.events))
            ]
        else:
            latencies_list = [(self.events[i] - self.events[i - 1]) for i in range(1, len(self.events))]

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)


class LatencyLogitsProcessor(LogitsProcessor):
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend
        self.reset()

    def reset(self):
        if self.device == "cuda" and self.backend == "pytorch":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events = [event]
        else:
            self.events = [time.perf_counter()]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.device == "cuda" and self.backend == "pytorch":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.append(event)
        else:
            self.events.append(time.perf_counter())

        return scores

    def get_latency(self) -> Latency:
        if self.device == "cuda" and self.backend == "pytorch":
            # synchronize the device to make sure all events have been recorded
            torch.cuda.synchronize()
            latencies_list = [
                self.events[i - 1].elapsed_time(self.events[i]) / 1e3 for i in range(1, len(self.events))
            ]
        else:
            latencies_list = [(self.events[i] - self.events[i - 1]) for i in range(1, len(self.events))]

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)
