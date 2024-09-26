from dataclasses import dataclass, make_dataclass
from typing import Any, Dict, List, Optional

from ..hub_utils import PushToHubMixin, classproperty
from ..trackers.energy import Efficiency, Energy
from ..trackers.latency import Latency, Throughput
from ..trackers.memory import Memory


@dataclass
class BenchmarkMeasurements:
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
    def aggregate(measurements: List["BenchmarkMeasurements"]) -> "BenchmarkMeasurements":
        assert len(measurements) > 0, "No measurements to aggregate"

        m0 = measurements[0]

        memory = Memory.aggregate([m.memory for m in measurements]) if m0.memory is not None else None
        latency = Latency.aggregate([m.latency for m in measurements]) if m0.latency is not None else None
        throughput = Throughput.aggregate([m.throughput for m in measurements]) if m0.throughput is not None else None
        energy = Energy.aggregate([m.energy for m in measurements]) if m0.energy is not None else None
        efficiency = Efficiency.aggregate([m.efficiency for m in measurements]) if m0.efficiency is not None else None

        return BenchmarkMeasurements(
            memory=memory, latency=latency, throughput=throughput, energy=energy, efficiency=efficiency
        )

    def log(self, prefix: str = ""):
        if self.memory is not None:
            self.memory.log(prefix=prefix)
        if self.latency is not None:
            self.latency.log(prefix=prefix)
        if self.throughput is not None:
            self.throughput.log(prefix=prefix)
        if self.energy is not None:
            self.energy.log(prefix=prefix)
        if self.efficiency is not None:
            self.efficiency.log(prefix=prefix)

    def markdown(self, prefix: str = "") -> str:
        markdown = ""

        if self.memory is not None:
            markdown += self.memory.markdown(prefix=prefix)
        if self.latency is not None:
            markdown += self.latency.markdown(prefix=prefix)
        if self.throughput is not None:
            markdown += self.throughput.markdown(prefix=prefix)
        if self.energy is not None:
            markdown += self.energy.markdown(prefix=prefix)
        if self.efficiency is not None:
            markdown += self.efficiency.markdown(prefix=prefix)

        return markdown


@dataclass
class BenchmarkReport(PushToHubMixin):
    @classmethod
    def from_list(cls, targets: List[str]) -> "BenchmarkReport":
        return cls.from_dict({target: None for target in targets})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkReport":
        return make_dataclass(cls_name=cls.__name__, fields=data.keys(), bases=(cls,))(**data)

    def __post_init__(self):
        for target in self.to_dict().keys():
            if getattr(self, target) is None:
                setattr(self, target, BenchmarkMeasurements())
            elif isinstance(getattr(self, target), dict):
                setattr(self, target, BenchmarkMeasurements(**getattr(self, target)))

    @classmethod
    def aggregate(cls, reports: List["BenchmarkReport"]) -> "BenchmarkReport":
        aggregated_measurements = {}
        for target in reports[0].to_dict().keys():
            measurements = [getattr(report, target) for report in reports]
            aggregated_measurements[target] = BenchmarkMeasurements.aggregate(measurements)

        return cls.from_dict(aggregated_measurements)

    def log(self):
        for target in self.to_dict().keys():
            measurements: BenchmarkMeasurements = getattr(self, target)
            measurements.log(prefix=target)

    def markdown(self):
        markdown = ""

        for target in self.to_dict().keys():
            measurements: BenchmarkMeasurements = getattr(self, target)
            markdown += measurements.markdown(prefix=target)

        return markdown

    @classproperty
    def default_filename(self) -> str:
        return "benchmark_report.json"
