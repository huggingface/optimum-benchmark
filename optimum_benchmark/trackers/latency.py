import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from rich.console import Console
from rich.markdown import Markdown
from transformers import LogitsProcessor, TrainerCallback

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

        assert not any(
            latency < 0 for latency in latencies
        ), "Negative latency detected. Please increase the dimensions of your benchmark (inputs/warmup/iterations)."

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

        self.start_time: Optional[float] = None
        self.start_events: List[Union[float, torch.cuda.Event]] = []
        self.end_events: List[Union[float, torch.cuda.Event]] = []

    def reset(self):
        self.start_time = None
        self.start_events = []
        self.end_events = []

    @contextmanager
    def track(self):
        if self.is_pytorch_cuda:
            yield from self._pytorch_cuda_latency()
        else:
            yield from self._cpu_latency()

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
        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies_list = [
                self.start_events[i].elapsed_time(self.end_events[i]) / 1e3 for i in range(len(self.start_events))
            ]
        else:
            latencies_list = [(self.end_events[i] - self.start_events[i]) for i in range(len(self.start_events))]

        assert not any(
            latency < 0 for latency in latencies_list
        ), "Negative latency detected. Please increase the dimensions of your benchmark (inputs/warmup/iterations)."

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)

    def count(self):
        assert len(self.start_events) == len(
            self.end_events
        ), "Mismatched number of start and end events, count() should only be called outside of track() context"

        return len(self.start_events)

    def elapsed(self):
        if self.start_time is None:
            assert (
                len(self.start_events) == 0 and len(self.end_events) == 0
            ), "Number of recorded events is not zero, make sure to reset() the tracker properly"

            self.start_time = time.perf_counter()

        return time.perf_counter() - self.start_time


class StepLatencyTrainerCallback(TrainerCallback):
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

    def reset(self):
        self.start_events = []
        self.end_events = []

    def on_step_begin(self, *args, **kwargs):
        if self.is_pytorch_cuda:
            self.start_events.append(torch.cuda.Event(enable_timing=True))
            self.start_events[-1].record()
        else:
            self.start_events.append(time.perf_counter())

    def on_step_end(self, *args, **kwargs):
        if self.is_pytorch_cuda:
            self.end_events.append(torch.cuda.Event(enable_timing=True))
            self.end_events[-1].record()
        else:
            self.end_events.append(time.perf_counter())

    def get_latency(self) -> Latency:
        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies_list = [
                self.start_events[i].elapsed_time(self.end_events[i]) / 1e3 for i in range(len(self.start_events))
            ]
        else:
            latencies_list = [(self.end_events[i] - self.start_events[i]) for i in range(len(self.start_events))]

        assert not any(
            latency < 0 for latency in latencies_list
        ), "Negative latency detected. Please increase the dimensions of your benchmark (inputs/warmup/iterations)."

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)


class PerTokenLatencyLogitsProcessor(LogitsProcessor):
    def __init__(self, device: str, backend: str):
        self.device = device
        self.backend = backend

        self.is_pytorch_cuda = (self.backend, self.device) == ("pytorch", "cuda")

        if self.is_pytorch_cuda:
            LOGGER.info("\t\t+ Tracking latency using Pytorch CUDA events")
        else:
            LOGGER.info("\t\t+ Tracking latency using CPU performance counter")

        self.start_time: Optional[float] = None
        self.prefilled: Optional[bool] = None

        self.per_token_events: List[List[Union[float, torch.cuda.Event]]] = []
        self.prefill_start_events: List[Union[float, torch.cuda.Event]] = []
        self.prefill_end_events: List[Union[float, torch.cuda.Event]] = []
        self.decode_start_events: List[Union[float, torch.cuda.Event]] = []
        self.decode_end_events: List[Union[float, torch.cuda.Event]] = []

    def reset(self):
        self.start_time = None
        self.prefilled = None

        self.per_token_events = []
        self.prefill_start_events = []
        self.prefill_end_events = []
        self.decode_start_events = []
        self.decode_end_events = []

    @contextmanager
    def track(self):
        self.prefilled = False
        self.per_token_events.append([])

        if self.is_pytorch_cuda:
            self.prefill_start_events.append(torch.cuda.Event(enable_timing=True))
            self.prefill_start_events[-1].record()
        else:
            self.prefill_start_events.append(time.perf_counter())

        yield

        if self.is_pytorch_cuda:
            self.decode_end_events.append(torch.cuda.Event(enable_timing=True))
            self.decode_end_events[-1].record()
        else:
            self.decode_end_events.append(time.perf_counter())

        self.prefilled = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        assert (
            self.prefilled is not None
        ), "PerTokenLatencyLogitsProcessor should only be called inside of track() context"

        if self.is_pytorch_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        else:
            event = time.perf_counter()

        if not self.prefilled:
            self.prefill_end_events.append(event)
            self.decode_start_events.append(event)
            self.prefilled = True

        self.per_token_events[-1].append(event)

        return scores

    def get_prefill_latency(self) -> Latency:
        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies_list = [
                self.prefill_start_events[i].elapsed_time(self.prefill_end_events[i]) / 1e3
                for i in range(len(self.prefill_start_events))
            ]
        else:
            latencies_list = [
                (self.prefill_end_events[i] - self.prefill_start_events[i])
                for i in range(len(self.prefill_start_events))
            ]

        assert not any(
            latency < 0 for latency in latencies_list
        ), "Negative latency detected. Please increase the dimensions of your benchmark (inputs/warmup/iterations)."

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)

    def get_decode_latency(self) -> Latency:
        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies_list = [
                self.decode_start_events[i].elapsed_time(self.decode_end_events[i]) / 1e3
                for i in range(len(self.decode_start_events))
            ]
        else:
            latencies_list = [
                (self.decode_end_events[i] - self.decode_start_events[i]) for i in range(len(self.decode_start_events))
            ]

        assert not any(
            latency < 0 for latency in latencies_list
        ), "Negative latency detected. Please increase the dimensions of your benchmark (inputs/warmup/iterations)."

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)

    def get_per_token_latency(self) -> Latency:
        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

            latencies_list = [
                self.per_token_events[i][j].elapsed_time(self.per_token_events[i][j + 1]) / 1e3
                for i in range(len(self.per_token_events))
                for j in range(0, len(self.per_token_events[i]) - 1)
            ]
        else:
            latencies_list = [
                (self.per_token_events[i][j + 1] - self.per_token_events[i][j])
                for i in range(len(self.per_token_events))
                for j in range(0, len(self.per_token_events[i]) - 1)
            ]

        assert not any(
            latency < 0 for latency in latencies_list
        ), "Negative latency detected. Please increase the dimensions of your benchmark (inputs/warmup/iterations)."

        return Latency.from_values(latencies_list, unit=LATENCY_UNIT)

    def count(self):
        assert len(self.prefill_start_events) == len(
            self.prefill_end_events
        ), "Mismatched number of start and end events, count() should only be called outside of track() context"

        return len(self.prefill_start_events)

    def elapsed(self):
        if self.start_time is None:
            assert (
                len(self.prefill_start_events) == 0 and len(self.prefill_end_events) == 0
            ), "Number of recorded events is not zero, make sure to reset() the tracker properly"

            self.start_time = time.perf_counter()

        return time.perf_counter() - self.start_time
