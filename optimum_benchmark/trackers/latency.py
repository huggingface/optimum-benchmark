import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from rich.console import Console
from rich.markdown import Markdown
from transformers import TrainerCallback

CONSOLE = Console()
LOGGER = getLogger("latency")

LATENCY_UNIT = "s"

Latency_Unit_Literal = Literal["s"]
Throughput_Unit_Literal = Literal["samples/s", "tokens/s", "images/s", "steps/s"]


@dataclass
class Latency:
    unit: Latency_Unit_Literal

    values: List[float]

    count: int
    total: float
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    stdev: float
    stdev_: float

    def __getitem__(self, index) -> float:
        if isinstance(index, slice):
            return Latency.from_values(values=self.values[index], unit=self.unit)
        elif isinstance(index, int):
            return Latency.from_values(values=[self.values[index]], unit=self.unit)
        else:
            raise ValueError(f"Invalid index type: {type(index)}, expected int or slice")

    def __sub__(self, latency: "Latency") -> "Latency":
        latencies = [lat - latency.mean for lat in self.values]

        assert all(latency >= 0 for latency in latencies), (
            "Found some negative latencies while performing substraction. "
            "Please increase the dimensions of your benchmark or the number of warmup runs."
        )

        return Latency.from_values(values=latencies, unit=self.unit)

    @staticmethod
    def aggregate_across_processes(latencies: List["Latency"]) -> "Latency":
        if len(latencies) == 0:
            raise ValueError("No latency measurements to aggregate")
        elif any(latency is None for latency in latencies):
            raise ValueError("Some latency measurements are missing")

        # we combine the lists of latencies and statistics are then computed on this list
        values = sum((lat.values for lat in latencies), [])

        unit = latencies[0].unit

        return Latency.from_values(values=values, unit=unit)

    @staticmethod
    def from_values(values: List[float], unit: str) -> "Latency":
        return Latency(
            unit=unit,
            values=values,
            count=len(values),
            total=sum(values),
            mean=np.mean(values),
            p50=np.percentile(values, 50),
            p90=np.percentile(values, 90),
            p95=np.percentile(values, 95),
            p99=np.percentile(values, 99),
            stdev=np.std(values) if len(values) > 1 else 0,
            stdev_=(np.std(values) / np.abs(np.mean(values))) * 100 if len(values) > 1 else 0,
        )

    def to_plain_text(self) -> str:
        plain_text = ""
        plain_text += "\t\t+ count: {count}\n"
        plain_text += "\t\t+ total: {total:.6f} ({unit})\n"
        plain_text += "\t\t+ mean: {mean:.6f} ({unit})\n"
        plain_text += "\t\t+ p50: {p50:.6f} ({unit})\n"
        plain_text += "\t\t+ p90: {p90:.6f} ({unit})\n"
        plain_text += "\t\t+ p95: {p95:.6f} ({unit})\n"
        plain_text += "\t\t+ p99: {p99:.6f} ({unit})\n"
        plain_text += "\t\t+ stdev: {stdev:.6f} ({unit})\n"
        plain_text += "\t\t+ stdev_: {stdev_:.2f} (%)\n"
        return plain_text.format(**asdict(self))

    def log(self):
        for line in self.to_plain_text().split("\n"):
            if line:
                LOGGER.info(line)

    def to_markdown_text(self) -> str:
        markdown_text = ""
        markdown_text += "| metric | value        | unit   |\n"
        markdown_text += "| :----- | -----------: |------: |\n"
        markdown_text += "| count  |      {count} |      - |\n"
        markdown_text += "| total  |    {total:f} | {unit} |\n"
        markdown_text += "| mean   |     {mean:f} | {unit} |\n"
        markdown_text += "| p50    |      {p50:f} | {unit} |\n"
        markdown_text += "| p90    |      {p90:f} | {unit} |\n"
        markdown_text += "| p95    |      {p95:f} | {unit} |\n"
        markdown_text += "| p99    |      {p99:f} | {unit} |\n"
        markdown_text += "| stdev  |    {stdev:f} | {unit} |\n"
        markdown_text += "| stdev_ | {stdev_:.2f} |      % |\n"
        return markdown_text.format(**asdict(self))

    def print(self):
        CONSOLE.print(Markdown(self.to_markdown_text()))


@dataclass
class Throughput:
    unit: Throughput_Unit_Literal

    value: float

    @staticmethod
    def aggregate_across_processes(throughputs: List[Optional["Throughput"]]) -> Optional["Throughput"]:
        if len(throughputs) == 0:
            raise ValueError("No throughput measurements to aggregate")
        elif any(throughput is None for throughput in throughputs):
            raise ValueError("Some throughput measurements are missing")

        # we compute throughputs on the whole input level so we just take the average
        value = sum(throughput.value for throughput in throughputs) / len(throughputs)
        unit = throughputs[0].unit

        return Throughput(value=value, unit=unit)

    @staticmethod
    def from_latency(latency: Latency, volume: int, unit: str) -> "Throughput":
        value = volume / latency.mean if latency.mean > 0 else 0
        return Throughput(value=value, unit=unit)

    def to_plain_text(self) -> str:
        plain_text = ""
        plain_text += "\t\t+ throughput: {value:.2f} ({unit})\n"
        return plain_text.format(**asdict(self))

    def log(self):
        for line in self.to_plain_text().split("\n"):
            if line:
                LOGGER.info(line)

    def to_markdown_text(self) -> str:
        markdown_text = ""
        markdown_text += "| metric     |     value   |   unit |\n"
        markdown_text += "| :--------- | --------:   | -----: |\n"
        markdown_text += "| throughput | {value:.2f} | {unit} |\n"
        return markdown_text.format(**asdict(self))

    def print(self):
        CONSOLE.print(Markdown(self.to_markdown_text()))


class LatencyTracker:
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend

        self.is_pytorch_cuda = (self.backend, self.device) == ("pytorch", "cuda")

        if self.is_pytorch_cuda:
            LOGGER.info("\t\t+ Tracking latency using Pytorch CUDA events")
        else:
            LOGGER.info("\t\t+ Tracking latency using CPU performance counter")

        self.start_event: Optional[Union[float, torch.cuda.Event]] = None
        self.end_event: Optional[Union[float, torch.cuda.Event]] = None

    @contextmanager
    def track(self):
        if self.is_pytorch_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

            self.start_event.record()
            yield
            self.end_event.record()
        else:
            self.start_event = time.perf_counter()
            yield
            self.end_event = time.perf_counter()

    def get_latency(self) -> Latency:
        assert self.start_event is not None and self.end_event is not None

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()
            latency = self.start_event.elapsed_time(self.end_event) / 1e3
        else:
            latency = self.end_event - self.start_event

        assert latency >= 0

        return Latency.from_values([latency], unit=LATENCY_UNIT)


class LatencySessionTracker:
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend

        self.is_pytorch_cuda = (self.backend, self.device) == ("pytorch", "cuda")

        if self.is_pytorch_cuda:
            LOGGER.info("\t\t+ Tracking latency using Pytorch CUDA events")
        else:
            LOGGER.info("\t\t+ Tracking latency using CPU performance counter")

        self.start_events: List[Union[float, torch.cuda.Event]] = []
        self.end_events: List[Union[float, torch.cuda.Event]] = []

        self.start_time: Optional[float] = None

    @contextmanager
    def session(self):
        assert self.start_time is None

        self.start_events = []
        self.end_events = []

        self.start_time = time.time()
        yield
        self.start_time = None

    def count(self) -> int:
        assert self.start_time is not None, "This method can only be called inside of a '.session()' context"
        assert len(self.start_events) == len(self.end_events)

        return len(self.start_events)

    def elapsed(self):
        assert self.start_time is not None, "This method can only be called inside of a '.session()' context"

        return time.time() - self.start_time

    @contextmanager
    def track(self):
        if self.is_pytorch_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            yield
            end_event.record()
        else:
            start_event = time.perf_counter()
            yield
            end_event = time.perf_counter()

        self.start_events.append(start_event)
        self.end_events.append(end_event)

    def get_latency(self) -> Latency:
        assert len(self.end_events) == len(self.start_events) >= 0

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()
            latencies = [
                start_event.elapsed_time(end_event) / 1e3
                for start_event, end_event in zip(self.start_events, self.end_events)
            ]
        else:
            latencies = [
                (end_event - start_event) for start_event, end_event in zip(self.start_events, self.end_events)
            ]

        assert all(latency >= 0 for latency in latencies), (
            "Found some negative latencies while performing substraction. "
            "Please increase the dimensions of your benchmark or the number of warmup runs."
        )

        return Latency.from_values(latencies, unit=LATENCY_UNIT)


class PerTokenLatencySessionTrackerLogitsProcessor:
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend

        self.is_pytorch_cuda = (self.backend, self.device) == ("pytorch", "cuda")

        if self.is_pytorch_cuda:
            LOGGER.info("\t\t+ Tracking latency using Pytorch CUDA events")
        else:
            LOGGER.info("\t\t+ Tracking latency using CPU performance counter")

        self.prefill_start_events: List[Union[float, torch.cuda.Event]] = []
        self.prefill_end_events: List[Union[float, torch.cuda.Event]] = []
        self.per_token_start_events: List[Union[float, torch.cuda.Event]] = []
        self.per_token_end_events: List[Union[float, torch.cuda.Event]] = []
        self.per_token_events: List[Union[float, torch.cuda.Event]] = []
        self.decode_start_events: List[Union[float, torch.cuda.Event]] = []
        self.decode_end_events: List[Union[float, torch.cuda.Event]] = []

        self.start_time: Optional[float] = None

    @contextmanager
    def session(self):
        assert self.start_time is None

        self.prefill_start_events = []
        self.prefill_end_events = []
        self.per_token_start_events = []
        self.per_token_end_events = []
        self.per_token_events = []
        self.decode_start_events = []
        self.decode_end_events = []

        self.start_time = time.time()
        yield
        self.start_time = None

    def count(self) -> int:
        assert self.start_time is not None, "This method can only be called inside of a '.session()' context"
        assert (
            len(self.prefill_start_events)
            == len(self.prefill_end_events)
            == len(self.decode_start_events)
            == len(self.decode_end_events)
        )

        return len(self.prefill_start_events)

    def elapsed(self):
        assert self.start_time is not None, "This method can only be called inside of a '.session()' context"

        return time.time() - self.start_time

    @contextmanager
    def track(self):
        if self.is_pytorch_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            yield
            end_event.record()
        else:
            start_event = time.perf_counter()
            yield
            end_event = time.perf_counter()

        self.prefill_start_events.append(start_event)
        self.decode_end_events.append(end_event)

        self.per_token_start_events.extend(self.per_token_events[:-1])
        self.per_token_end_events.extend(self.per_token_events[1:])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.is_pytorch_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()

        if len(self.prefill_start_events) == len(self.prefill_end_events):
            # on the first call (prefill), there will be the same number of prefill/decode start/end events
            self.prefill_end_events.append(event)
            self.decode_start_events.append(event)

        self.per_token_events.append(event)

        return scores

    def get_generate_latency(self) -> Latency:
        assert len(self.prefill_start_events) == len(self.prefill_end_events) > 0
        assert len(self.decode_start_events) == len(self.decode_end_events) > 0

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies = [
                start_event.elapsed_time(end_event) / 1e3
                for start_event, end_event in zip(self.prefill_start_events, self.decode_end_events)
            ]
        else:
            latencies = [
                (end_event - start_event)
                for start_event, end_event in zip(self.prefill_start_events, self.decode_end_events)
            ]

        assert all(latency >= 0 for latency in latencies), (
            "Found some negative latencies while performing substraction. "
            "Please increase the dimensions of your benchmark or the number of warmup runs."
        )

        return Latency.from_values(latencies, unit=LATENCY_UNIT)

    def get_prefill_latency(self) -> Latency:
        assert len(self.prefill_start_events) == len(self.prefill_end_events) > 0

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies = [
                start_event.elapsed_time(end_event) / 1e3
                for start_event, end_event in zip(self.prefill_start_events, self.prefill_end_events)
            ]
        else:
            latencies = [
                (end_event - start_event)
                for start_event, end_event in zip(self.prefill_start_events, self.prefill_end_events)
            ]

        assert all(latency >= 0 for latency in latencies), (
            "Found some negative latencies while performing substraction. "
            "Please increase the dimensions of your benchmark or the number of warmup runs."
        )

        return Latency.from_values(latencies, unit=LATENCY_UNIT)

    def get_decode_latency(self) -> Latency:
        assert len(self.decode_start_events) == len(self.decode_end_events) > 0

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies = [
                start_event.elapsed_time(end_event) / 1e3
                for start_event, end_event in zip(self.decode_start_events, self.decode_end_events)
            ]
        else:
            latencies = [
                (end_event - start_event)
                for start_event, end_event in zip(self.decode_start_events, self.decode_end_events)
            ]

        assert all(latency >= 0 for latency in latencies), (
            "Found some negative latencies while performing substraction. "
            "Please increase the dimensions of your benchmark or the number of warmup runs."
        )

        return Latency.from_values(latencies, unit=LATENCY_UNIT)

    def get_per_token_latency(self) -> Latency:
        assert len(self.per_token_start_events) == len(self.per_token_end_events) > 0

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies = [
                start_event.elapsed_time(end_event) / 1e3
                for start_event, end_event in zip(self.per_token_start_events, self.per_token_end_events)
            ]
        else:
            latencies = [
                (end_event - start_event)
                for start_event, end_event in zip(self.per_token_start_events, self.per_token_end_events)
            ]

        assert all(latency >= 0 for latency in latencies), (
            "Found some negative latencies while performing substraction. "
            "Please increase the dimensions of your benchmark or the number of warmup runs."
        )

        return Latency.from_values(latencies, unit=LATENCY_UNIT)


class PerStepLatencySessionTrackerPipelineCallback:
    tensor_inputs = []

    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend

        self.is_pytorch_cuda = (self.backend, self.device) == ("pytorch", "cuda")

        if self.is_pytorch_cuda:
            LOGGER.info("\t\t+ Tracking latency using Pytorch CUDA events")
        else:
            LOGGER.info("\t\t+ Tracking latency using CPU performance counter")

        self.call_start_events: List[Union[float, torch.cuda.Event]] = []
        self.call_end_events: List[Union[float, torch.cuda.Event]] = []
        self.per_step_start_events: List[Union[float, torch.cuda.Event]] = []
        self.per_step_end_events: List[Union[float, torch.cuda.Event]] = []
        self.per_step_events: List[Union[float, torch.cuda.Event]] = []

        self.start_time: Optional[float] = None

    @contextmanager
    def session(self):
        assert self.start_time is None

        self.call_start_events = []
        self.call_end_events = []
        self.per_step_start_events = []
        self.per_step_end_events = []
        self.per_step_events = []

        self.start_time = time.time()
        yield
        self.start_time = None

    def count(self) -> int:
        assert self.start_time is not None, "This method can only be called inside of a '.session()' context"
        assert len(self.call_start_events) == len(self.call_start_events)

        return len(self.call_start_events)

    def elapsed(self):
        assert self.start_time is not None, "This method can only be called inside of a '.session()' context"

        return time.time() - self.start_time

    @contextmanager
    def track(self):
        if self.is_pytorch_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            yield
            end_event.record()
        else:
            start_event = time.perf_counter()
            yield
            end_event = time.perf_counter()

        self.call_start_events.append(start_event)
        self.call_end_events.append(end_event)

        self.per_step_start_events.extend(self.per_step_events[:-1])
        self.per_step_end_events.extend(self.per_step_events[1:])

    def __call__(self, pipeline, step_index, timestep, callback_kwargs):
        if self.is_pytorch_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()

        self.per_step_events.append(event)

        return callback_kwargs

    def get_step_latency(self) -> Latency:
        assert len(self.per_step_start_events) == len(self.per_step_end_events) > 0

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies = [
                start_event.elapsed_time(end_event) / 1e3
                for start_event, end_event in zip(self.per_step_start_events, self.per_step_end_events)
            ]
        else:
            latencies = [
                (end_event - start_event)
                for start_event, end_event in zip(self.per_step_start_events, self.per_step_end_events)
            ]

        assert all(latency >= 0 for latency in latencies), (
            "Found some negative latencies while performing substraction. "
            "Please increase the dimensions of your benchmark or the number of warmup runs."
        )

        return Latency.from_values(latencies, unit=LATENCY_UNIT)

    def get_call_latency(self) -> Latency:
        assert len(self.call_start_events) == len(self.call_end_events) > 0

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies = [
                start_event.elapsed_time(end_event) / 1e3
                for start_event, end_event in zip(self.call_start_events, self.call_end_events)
            ]
        else:
            latencies = [
                (end_event - start_event)
                for start_event, end_event in zip(self.call_start_events, self.call_end_events)
            ]

        assert all(latency >= 0 for latency in latencies), (
            "Found some negative latencies while performing substraction. "
            "Please increase the dimensions of your benchmark or the number of warmup runs."
        )

        return Latency.from_values(latencies, unit=LATENCY_UNIT)


class StepLatencyTrackerTrainerCallback(TrainerCallback):
    def __init__(self, device: str, backend: str) -> None:
        self.device = device
        self.backend = backend

        self.is_pytorch_cuda = (self.backend, self.device) == ("pytorch", "cuda")

        if self.is_pytorch_cuda:
            LOGGER.info("\t\t+ Tracking latency using Pytorch CUDA events")
        else:
            LOGGER.info("\t\t+ Tracking latency using CPU performance counter")

        self.start_events: List[Union[float, torch.cuda.Event]] = []
        self.end_events: List[Union[float, torch.cuda.Event]] = []

    def on_step_begin(self, *args, **kwargs):
        if self.is_pytorch_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()

        self.start_events.append(event)

    def on_step_end(self, *args, **kwargs):
        if self.is_pytorch_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()

        self.end_events.append(event)

    def get_latency(self) -> Latency:
        assert len(self.start_events) == len(self.end_events) > 0

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()
            latencies = [
                start_event.elapsed_time(end_event) / 1e3
                for start_event, end_event in zip(self.start_events, self.end_events)
            ]
        else:
            latencies = [
                (end_event - start_event) for start_event, end_event in zip(self.start_events, self.end_events)
            ]

        assert all(latency >= 0 for latency in latencies), (
            "Found some negative latencies while performing substraction. "
            "Please increase the dimensions of your benchmark or the number of warmup runs."
        )

        return Latency.from_values(latencies, unit=LATENCY_UNIT)
