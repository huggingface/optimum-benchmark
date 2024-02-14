from dataclasses import dataclass
from logging import getLogger

from ..report import BenchmarkReport, BenchmarkMeasurements

LOGGER = getLogger("report")


@dataclass
class TrainingReport(BenchmarkReport):
    overall: BenchmarkMeasurements = BenchmarkMeasurements()
    warmup: BenchmarkMeasurements = BenchmarkMeasurements()
    train: BenchmarkMeasurements = BenchmarkMeasurements()
