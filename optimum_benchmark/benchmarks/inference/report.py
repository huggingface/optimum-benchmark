from dataclasses import dataclass
from logging import getLogger

from ..report import BenchmarkReport, BenchmarkMeasurements

LOGGER = getLogger("report")


@dataclass
class InferenceReport(BenchmarkReport):
    forward: BenchmarkMeasurements = BenchmarkMeasurements()


@dataclass
class ImageDiffusionReport(BenchmarkReport):
    call: BenchmarkMeasurements = BenchmarkMeasurements()


@dataclass
class TextGenerationReport(BenchmarkReport):
    prefill: BenchmarkMeasurements = BenchmarkMeasurements()
    decode: BenchmarkMeasurements = BenchmarkMeasurements()
