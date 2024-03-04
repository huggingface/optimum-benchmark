from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional

from ..hub_utils import PushToHubMixin
from ..trackers.energy import Efficiency, Energy
from ..trackers.latency import Latency, Throughput
from ..trackers.memory import Memory

LOGGER = getLogger("report")


@dataclass
class BenchmarkMeasurements:
    memory: Optional[Memory] = None
    latency: Optional[Latency] = None
    throughput: Optional[Throughput] = None
    energy: Optional[Energy] = None
    efficiency: Optional[Efficiency] = None

    @staticmethod
    def aggregate(benchmark_measurements: List["BenchmarkMeasurements"]) -> "BenchmarkMeasurements":
        memory = (
            Memory.aggregate([m.memory for m in benchmark_measurements])
            if benchmark_measurements[0].memory is not None
            else None
        )
        latency = (
            Latency.aggregate([m.latency for m in benchmark_measurements])
            if benchmark_measurements[0].latency is not None
            else None
        )
        throughput = (
            Throughput.aggregate([m.throughput for m in benchmark_measurements if m.throughput is not None])
            if benchmark_measurements[0].throughput is not None
            else None
        )
        energy = (
            Energy.aggregate([m.energy for m in benchmark_measurements if m.energy is not None])
            if benchmark_measurements[0].energy is not None
            else None
        )
        efficiency = (
            Efficiency.aggregate([m.efficiency for m in benchmark_measurements if m.efficiency is not None])
            if benchmark_measurements[0].efficiency is not None
            else None
        )

        return BenchmarkMeasurements(
            memory=memory, latency=latency, throughput=throughput, energy=energy, efficiency=efficiency
        )


@dataclass
class BenchmarkReport(PushToHubMixin):
    def log_memory(self):
        for target in self.to_dict().keys():
            benchmark_measurements: BenchmarkMeasurements = getattr(self, target)
            if benchmark_measurements.memory is not None:
                benchmark_measurements.memory.log(prefix=target)

    def log_latency(self):
        for target in self.to_dict().keys():
            benchmark_measurements: BenchmarkMeasurements = getattr(self, target)
            if benchmark_measurements.latency is not None:
                benchmark_measurements.latency.log(prefix=target)

    def log_throughput(self):
        for target in self.to_dict().keys():
            benchmark_measurements: BenchmarkMeasurements = getattr(self, target)
            if benchmark_measurements.throughput is not None:
                benchmark_measurements.throughput.log(prefix=target)

    def log_energy(self):
        for target in self.to_dict().keys():
            benchmark_measurements: BenchmarkMeasurements = getattr(self, target)
            if benchmark_measurements.energy is not None:
                benchmark_measurements.energy.log(prefix=target)

    def log_efficiency(self):
        for target in self.to_dict().keys():
            benchmark_measurements: BenchmarkMeasurements = getattr(self, target)
            if benchmark_measurements.efficiency is not None:
                benchmark_measurements.efficiency.log(prefix=target)

    def log(self):
        for target in self.to_dict().keys():
            benchmark_measurements: BenchmarkMeasurements = getattr(self, target)
            if benchmark_measurements.memory is not None:
                benchmark_measurements.memory.log(prefix=target)
            if benchmark_measurements.latency is not None:
                benchmark_measurements.latency.log(prefix=target)
            if benchmark_measurements.throughput is not None:
                benchmark_measurements.throughput.log(prefix=target)
            if benchmark_measurements.energy is not None:
                benchmark_measurements.energy.log(prefix=target)
            if benchmark_measurements.efficiency is not None:
                benchmark_measurements.efficiency.log(prefix=target)

    @classmethod
    def aggregate(cls, reports: List["BenchmarkReport"]) -> "BenchmarkReport":
        aggregated_measurements = {}
        for target in reports[0].to_dict().keys():
            benchmark_measurements = [getattr(report, target) for report in reports]
            aggregated_measurements[target] = BenchmarkMeasurements.aggregate(benchmark_measurements)

        return cls(**aggregated_measurements)

    @property
    def file_name(self) -> str:
        return "benchmark_report.json"
