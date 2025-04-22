from dataclasses import dataclass, make_dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown

from ..hub_utils import PushToHubMixin, classproperty
from ..trackers.energy import Efficiency, Energy
from ..trackers.latency import Latency, Throughput
from ..trackers.memory import Memory

CONSOLE = Console()
LOGGER = getLogger("report")


@dataclass
class TargetMeasurements:
    memory: Optional[Memory] = None
    latency: Optional[Latency] = None
    throughput: Optional[Throughput] = None
    energy: Optional[Energy] = None
    efficiency: Optional[Efficiency] = None

    def __post_init__(self):
        if self.memory is not None and isinstance(self.memory, dict):
            self.memory = Memory(**self.memory)
        if self.latency is not None and isinstance(self.latency, dict):
            self.latency = Latency(**self.latency)
        if self.throughput is not None and isinstance(self.throughput, dict):
            self.throughput = Throughput(**self.throughput)
        if self.energy is not None and isinstance(self.energy, dict):
            self.energy = Energy(**self.energy)
        if self.efficiency is not None and isinstance(self.efficiency, dict):
            self.efficiency = Efficiency(**self.efficiency)

    @staticmethod
    def aggregate_across_processes(measurements: List["TargetMeasurements"]) -> "TargetMeasurements":
        assert len(measurements) > 0, "No measurements to aggregate"

        m0 = measurements[0]

        memory = Memory.aggregate_across_processes([m.memory for m in measurements]) if m0.memory is not None else None
        latency = (
            Latency.aggregate_across_processes([m.latency for m in measurements]) if m0.latency is not None else None
        )
        throughput = (
            Throughput.aggregate_across_processes([m.throughput for m in measurements])
            if m0.throughput is not None
            else None
        )
        energy = Energy.aggregate_across_processes([m.energy for m in measurements]) if m0.energy is not None else None
        efficiency = (
            Efficiency.aggregate_across_processes([m.efficiency for m in measurements])
            if m0.efficiency is not None
            else None
        )

        return TargetMeasurements(
            memory=memory, latency=latency, throughput=throughput, energy=energy, efficiency=efficiency
        )

    def to_plain_text(self) -> str:
        plain_text = ""

        for key in ["memory", "latency", "throughput", "energy", "efficiency"]:
            measurement = getattr(self, key)
            if measurement is not None:
                plain_text += f"\t+ {key}:\n"
                plain_text += measurement.to_plain_text()

        return plain_text

    def log(self):
        for line in self.to_plain_text().split("\n"):
            if line:
                LOGGER.info(line)

    def to_markdown_text(self) -> str:
        markdown_text = ""

        for key in ["memory", "latency", "throughput", "energy", "efficiency"]:
            measurement = getattr(self, key)
            if measurement is not None:
                markdown_text += f"## {key}:\n\n"
                markdown_text += measurement.to_markdown_text()

        return markdown_text

    def print(self):
        CONSOLE.print(Markdown(self.to_markdown_text()))


@dataclass
class BenchmarkReport(PushToHubMixin):
    @classmethod
    def from_list(cls, targets: List[str]) -> "BenchmarkReport":
        return cls.from_dict(dict.fromkeys(targets))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkReport":
        return make_dataclass(cls_name=cls.__name__, fields=data.keys(), bases=(cls,))(**data)

    def __post_init__(self):
        for target in self.to_dict().keys():
            if getattr(self, target) is None:
                setattr(self, target, TargetMeasurements())
            elif isinstance(getattr(self, target), dict):
                setattr(self, target, TargetMeasurements(**getattr(self, target)))

    @classmethod
    def aggregate_across_processes(cls, reports: List["BenchmarkReport"]) -> "BenchmarkReport":
        aggregated_measurements = {}
        for target in reports[0].to_dict().keys():
            measurements = [getattr(report, target) for report in reports]
            aggregated_measurements[target] = TargetMeasurements.aggregate_across_processes(measurements)

        return cls.from_dict(aggregated_measurements)

    def to_plain_text(self) -> str:
        plain_text = ""

        for target in self.to_dict().keys():
            plain_text += f"+ {target}:\n"
            plain_text += getattr(self, target).to_plain_text()

        return plain_text

    def to_markdown_text(self) -> str:
        markdown_text = ""

        for target in self.to_dict().keys():
            markdown_text += f"# {target}:\n\n"
            markdown_text += getattr(self, target).to_markdown_text()

        return markdown_text

    def save_text(self, filename: str):
        with open(filename, mode="w") as f:
            f.write(self.to_plain_text())

    def save_markdown(self, filename: str):
        with open(filename, mode="w") as f:
            f.write(self.to_markdown_text())

    def log(self):
        for line in self.to_plain_text().split("\n"):
            if line:
                LOGGER.info(line)

    def print(self):
        CONSOLE.print(Markdown(self.to_markdown_text()))

    @classproperty
    def default_filename(self) -> str:
        return "benchmark_report.json"
