import time
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from typing import List, Literal, Union

from ..import_utils import is_torch_distributed_available

if is_torch_distributed_available():
    import torch.distributed

import torch
from transformers import LogitsProcessor, TrainerCallback

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

    def __getitem__(self, index) -> float:
        if isinstance(index, slice):
            return Latency.from_values(values=self.values[index], unit=self.unit)
        elif isinstance(index, int):
            return Latency.from_values(values=[self.values[index]], unit=self.unit)
        else:
            raise ValueError(f"Invalid index type: {type(index)}, expected int or slice")

    def __sub__(self, latency: "Latency") -> "Latency":
        if not isinstance(latency, Latency):
            raise ValueError(f"Cannot subtract {type(latency)} from Latency")

        latencies = [lat - latency.mean for lat in self.values]

        assert not any(latency < 0 for latency in latencies), "Negative latency detected"

        return Latency.from_values(values=latencies, unit=self.unit)

    @staticmethod
    def aggregate(latencies: List["Latency"]) -> "Latency":
        if len(latencies) == 0 or all(latency is None for latency in latencies):
            return None
        elif any(latency is None for latency in latencies):
            raise ValueError("Some latency measurements are missing")

        unit = latencies[0].unit
        values = sum((lat.values for lat in latencies), [])
        return Latency.from_values(values=values, unit=unit)

    @staticmethod
    def from_values(values: List[float], unit: str) -> "Latency":
        mean = sum(values) / len(values) if len(values) > 0 else 0
        stdev = (sum((val - mean) ** 2 for val in values) / len(values)) ** 0.5 if len(values) > 1 else 0
        return Latency(mean=mean, stdev=stdev, values=values, unit=unit)

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} latency: {self.mean:f} ± {self.stdev:f} ({self.unit})")


@dataclass
class Throughput:
    unit: Throughput_Unit_Literal

    value: float

    @staticmethod
    def aggregate(throughputs: List["Throughput"]) -> "Throughput":
        if len(throughputs) == 0:
            raise ValueError("No throughput measurements to aggregate")
        elif any(throughput is None for throughput in throughputs):
            raise ValueError("Some throughput measurements are missing")

        unit = throughputs[0].unit
        value = sum(throughput.value for throughput in throughputs)

        return Throughput(value=value, unit=unit)

    @staticmethod
    def from_latency(latency: Latency, volume: int, unit: str) -> "Throughput":
        value = volume / latency.mean if latency.mean > 0 else 0
        return Throughput(value=value, unit=unit)

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} throughput: {self.value:f} {self.unit}")


class Timer:
    def __init__(self):
        self.start_event = None

    def reset(self):
        self.start_event = None

    def elapsed(self):
        if self.start_event is None:
            self.start_event = time.perf_counter()

        return time.perf_counter() - self.start_event


class LatencyTracker:
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend
        self.use_cuda_events = self.backend == "pytorch" and self.device == "cuda"
        self.distributed = is_torch_distributed_available() and torch.distributed.is_initialized()

        if self.use_cuda_events:
            LOGGER.info("\t+ Tracking latency using Pytorch CUDA events")
        else:
            LOGGER.info("\t+ Tracking latency using CPU performance counter")

        self.start_events: List[Union[float, torch.cuda.Event]] = []
        self.end_events: List[Union[float, torch.cuda.Event]] = []

    def reset(self):
        self.start_events = []
        self.end_events = []

    @contextmanager
    def track(self):
        if self.distributed:
            torch.distributed.barrier()

        if self.use_cuda_events:
            yield from self._pytorch_cuda_latency()
        else:
            yield from self._cpu_latency()

        if self.distributed:
            torch.distributed.barrier()

    def _pytorch_cuda_latency(self):
        self.start_events.append(torch.cuda.Event(enable_timing=True))
        self.start_events[-1].record()

        yield

        self.end_events.append(torch.cuda.Event(enable_timing=True))
        self.end_events[-1].record()

    def _cpu_latency(self):
        self.start_events.append(time.perf_counter())

        yield

        self.end_events.append(time.perf_counter())

    def get_latency(self) -> Latency:
        if self.backend == "pytorch" and self.device == "cuda":
            torch.cuda.synchronize()  # synchronize the device to make sure all events have been recorded
            latencies_list = [
                self.start_events[i].elapsed_time(self.end_events[i]) / 1e3 for i in range(len(self.start_events))
            ]
        else:
            latencies_list = [(self.end_events[i] - self.start_events[i]) for i in range(len(self.start_events))]

        assert not any(latency < 0 for latency in latencies_list), "Negative latency detected"

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)


class StepLatencyTrainerCallback(TrainerCallback):
    def __init__(self, device: str, backend: str) -> None:
        self.device = device
        self.backend = backend
        self.use_cuda_events = self.backend == "pytorch" and self.device == "cuda"

        self.start_events: List[Union[float, torch.cuda.Event]] = []
        self.end_events: List[Union[float, torch.cuda.Event]] = []

    def reset(self):
        self.start_events = []
        self.end_events = []

    def on_step_begin(self, *args, **kwargs):
        if self.use_cuda_events:
            self.start_events.append(torch.cuda.Event(enable_timing=True))
            self.start_events[-1].record()
        else:
            self.start_events.append(time.perf_counter())

    def on_step_end(self, *args, **kwargs):
        if self.use_cuda_events:
            self.end_events.append(torch.cuda.Event(enable_timing=True))
            self.end_events[-1].record()
        else:
            self.end_events.append(time.perf_counter())

    def get_latency(self) -> Latency:
        if self.use_cuda_events:
            torch.cuda.synchronize()  # synchronize the device to make sure all events have been recorded
            latencies_list = [
                self.start_events[i].elapsed_time(self.end_events[i]) / 1e3 for i in range(len(self.start_events))
            ]
        else:
            latencies_list = [(self.end_events[i] - self.start_events[i]) for i in range(len(self.start_events))]

        assert not any(latency < 0 for latency in latencies_list), "Negative latency detected"

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)


class PerTokenLatencyLogitsProcessor(LogitsProcessor):
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend
        self.use_cuda_events = self.backend == "pytorch" and self.device == "cuda"
        self.distributed = is_torch_distributed_available() and torch.distributed.is_initialized()

        self.per_token_events: List[Union[float, torch.cuda.Event]] = []
        self.prefill_start_events: List[Union[float, torch.cuda.Event]] = []
        self.prefill_end_events: List[Union[float, torch.cuda.Event]] = []
        self.decode_start_events: List[Union[float, torch.cuda.Event]] = []
        self.decode_end_events: List[Union[float, torch.cuda.Event]] = []

    def reset(self):
        self.per_token_events = []
        self.prefill_start_events = []
        self.prefill_end_events = []
        self.decode_start_events = []
        self.decode_end_events = []

    @contextmanager
    def track(self):
        if self.distributed:
            torch.distributed.barrier()

        if self.use_cuda_events:
            self.prefill_start_events.append(torch.cuda.Event(enable_timing=True))
            self.prefill_start_events[-1].record()
        else:
            self.prefill_start_events.append(time.perf_counter())

        self.next_is_prefill_end_decode_start = True  # this is used to record the end of prefill and start of decode

        yield  # this is where generate is called, and for each decoded token, we record an event

        if self.use_cuda_events:
            self.decode_end_events.append(torch.cuda.Event(enable_timing=True))
            self.decode_end_events[-1].record()
        else:
            self.decode_end_events.append(time.perf_counter())

        if self.distributed:
            torch.distributed.barrier()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()

        if self.next_is_prefill_end_decode_start:
            self.prefill_end_events.append(event)
            self.decode_start_events.append(event)
            self.next_is_prefill_end_decode_start = False
        else:
            self.per_token_events.append(event)

        return scores

    def get_prefill_latency(self) -> Latency:
        if self.use_cuda_events:
            torch.cuda.synchronize()  # synchronize the device to make sure all events have been recorded
            latencies_list = [
                self.prefill_start_events[i].elapsed_time(self.prefill_end_events[i]) / 1e3
                for i in range(len(self.prefill_start_events))
            ]
        else:
            latencies_list = [
                (self.prefill_end_events[i] - self.prefill_start_events[i])
                for i in range(len(self.prefill_start_events))
            ]

        assert not any(latency < 0 for latency in latencies_list), "Negative latency detected"

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)

    def get_decode_latency(self) -> Latency:
        if self.use_cuda_events:
            torch.cuda.synchronize()  # synchronize the device to make sure all events have been recorded
            latencies_list = [
                self.decode_start_events[i].elapsed_time(self.decode_end_events[i]) / 1e3
                for i in range(len(self.decode_start_events))
            ]
        else:
            latencies_list = [
                (self.decode_end_events[i] - self.decode_start_events[i]) for i in range(len(self.decode_start_events))
            ]

        assert not any(latency < 0 for latency in latencies_list), "Negative latency detected"

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)

    def get_per_token_latency(self) -> Latency:
        if self.use_cuda_events:
            torch.cuda.synchronize()  # synchronize the device to make sure all events have been recorded
            latencies_list = [
                self.per_token_events[i].elapsed_time(self.per_token_events[i + 1]) / 1e3
                for i in range(0, len(self.per_token_events) - 1)
            ]
        else:
            latencies_list = [
                (self.per_token_events[i + 1] - self.per_token_events[i])
                for i in range(0, len(self.per_token_events) - 1)
            ]

        assert not any(latency < 0 for latency in latencies_list), "Negative latency detected"

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)
