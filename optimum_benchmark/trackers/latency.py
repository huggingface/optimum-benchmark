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

    def __getitem__(self, index: int) -> float:
        if isinstance(index, slice):
            return Latency.from_values(values=self.values[index], unit=self.unit)
        else:
            return Latency.from_values(values=[self.values[index]], unit=self.unit)

    def __sub__(self, latency: "Latency") -> "Latency":
        if not isinstance(latency, Latency):
            raise ValueError(f"Cannot subtract {type(latency)} from Latency")

        latencies = [lat - latency.mean for lat in self.values]
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
        LOGGER.info(f"\t\t+ {prefix} latency: {self.mean:f} ± 2 x {self.stdev:f} ({self.unit})")


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


class LatencyTracker:
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend
        self.distributed = is_torch_distributed_available() and torch.distributed.is_initialized()

        if self.backend == "pytorch" and self.device == "cuda":
            LOGGER.info("\t+ Tracking Pytorch CUDA latency")
        elif self.backend == "openvino":
            LOGGER.info(f"\t+ Tracking OpenVINO {self.device.upper()} latency")
        else:
            LOGGER.info("\t+ Tracking CPU latency")

        self.reset()

    def reset(self):
        self.start_events: List[Union[float, torch.cuda.Event]] = []
        self.end_events: List[Union[float, torch.cuda.Event]] = []
        self.start_time: float = time.perf_counter()

    @contextmanager
    def track(self):
        if self.distributed:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()] if self.device == "cuda" else None)

        if self.backend == "pytorch" and self.device == "cuda":
            yield from self._pytorch_cuda_latency()
        else:
            yield from self._cpu_latency()

        if self.distributed:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()] if self.device == "cuda" else None)

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
        # we measure it in cpu to not synchronize all events
        return time.perf_counter() - self.start_time

    def get_latency(self) -> Latency:
        if self.backend == "pytorch" and self.device == "cuda":
            # synchronize the device to make sure all events have been recorded
            torch.cuda.synchronize()

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

        self.reset()

    def reset(self):
        self.events: List[Union[float, torch.cuda.Event]] = []

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
            latencies_list = [self.events[i - 1].elapsed_time(self.events[i]) / 1e3 for i in range(1, len(self.events))]
        else:
            latencies_list = [(self.events[i] - self.events[i - 1]) for i in range(1, len(self.events))]

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)


class LatencyLogitsProcessor(LogitsProcessor):
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend
        self.distributed = is_torch_distributed_available() and torch.distributed.is_initialized()

        self.reset()

    def reset(self):
        # for each generate (run) pass, we store the time of each token
        self.run_events: List[List[Union[float, torch.cuda.Event]]] = []
        self.start_time: float = time.perf_counter()

    def get_elapsed_time(self) -> float:
        return time.perf_counter() - self.start_time

    @contextmanager
    def track(self):
        if self.distributed:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()] if self.device == "cuda" else None)

        self.tok_events: List[Union[float, torch.cuda.Event]] = []

        if self.device == "cuda" and self.backend == "pytorch":
            prefill_event = torch.cuda.Event(enable_timing=True)
            prefill_event.record()
            self.tok_events.append(prefill_event)

        yield  # this is where generate is called, and for each token, we record an event

        self.run_events.append(self.tok_events)

        if self.distributed:
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()] if self.device == "cuda" else None)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.device == "cuda" and self.backend == "pytorch":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.tok_events.append(event)
        else:
            self.tok_events.append(time.perf_counter())

        return scores

    def get_prefill_latency(self) -> Latency:
        if self.device == "cuda" and self.backend == "pytorch":
            # synchronize the device to make sure all events have been recorded
            torch.cuda.synchronize()
            latencies_list = [
                self.run_events[i][0].elapsed_time(self.run_events[i][1]) / 1e3 for i in range(len(self.run_events))
            ]
        else:
            latencies_list = [(self.run_events[i][1] - self.run_events[i][0]) for i in range(len(self.run_events))]

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)

    def get_per_token_latency(self) -> Latency:
        latencies_list = []
        for tok_events in self.run_events:
            if self.device == "cuda" and self.backend == "pytorch":
                # synchronize the device to make sure all events have been recorded
                torch.cuda.synchronize()
                latencies_list.extend(
                    [tok_events[i].elapsed_time(tok_events[i + 1]) / 1e3 for i in range(1, len(tok_events) - 1)]
                )
            else:
                latencies_list.extend([(tok_events[i] - tok_events[i + 1]) for i in range(1, len(tok_events) - 1)])

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)

    def get_decode_latency(self) -> Latency:
        latencies_list = []
        for tok_events in self.run_events:
            if self.device == "cuda" and self.backend == "pytorch":
                # synchronize the device to make sure all events have been recorded
                torch.cuda.synchronize()
                latencies_list.append(
                    sum([tok_events[i - 1].elapsed_time(tok_events[i]) / 1e3 for i in range(1, len(tok_events))])
                )
            else:
                latencies_list.append(sum([(tok_events[i] - tok_events[i - 1]) for i in range(1, len(tok_events))]))

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)
