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

        return BenchmarkMeasurements(memory, latency, throughput, energy, efficiency)


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

    def log_memory(self):
        for target in self.to_dict().keys():
            measurements: BenchmarkMeasurements = getattr(self, target)
            if measurements.memory is not None:
                measurements.memory.log(prefix=target)

    def log_latency(self):
        for target in self.to_dict().keys():
            measurements: BenchmarkMeasurements = getattr(self, target)
            if measurements.latency is not None:
                measurements.latency.log(prefix=target)

    def log_throughput(self):
        for target in self.to_dict().keys():
            measurements: BenchmarkMeasurements = getattr(self, target)
            if measurements.throughput is not None:
                measurements.throughput.log(prefix=target)

    def log_energy(self):
        for target in self.to_dict().keys():
            measurements: BenchmarkMeasurements = getattr(self, target)
            if measurements.energy is not None:
                measurements.energy.log(prefix=target)

    def log_efficiency(self):
        for target in self.to_dict().keys():
            measurements: BenchmarkMeasurements = getattr(self, target)
            if measurements.efficiency is not None:
                measurements.efficiency.log(prefix=target)

    def log(self):
        for target in self.to_dict().keys():
            measurements: BenchmarkMeasurements = getattr(self, target)
            if measurements.memory is not None:
                measurements.memory.log(prefix=target)
            if measurements.latency is not None:
                measurements.latency.log(prefix=target)
            if measurements.throughput is not None:
                measurements.throughput.log(prefix=target)
            if measurements.energy is not None:
                measurements.energy.log(prefix=target)
            if measurements.efficiency is not None:
                measurements.efficiency.log(prefix=target)

    @classmethod
    def aggregate(cls, reports: List["BenchmarkReport"]) -> "BenchmarkReport":
        aggregated_measurements = {}
        for target in reports[0].to_dict().keys():
            measurements = [getattr(report, target) for report in reports]
            aggregated_measurements[target] = BenchmarkMeasurements.aggregate(measurements)

        return cls.from_dict(aggregated_measurements)

    @classproperty
    def default_filename(self) -> str:
        return "benchmark_report.json"
