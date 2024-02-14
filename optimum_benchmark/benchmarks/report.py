from dataclasses import dataclass, asdict
from typing import Optional, Union, List
from logging import getLogger
from json import dump
import os

from ..trackers.latency import Latency, Throughput
from ..trackers.energy import Energy, Efficiency
from ..trackers.memory import MaxMemory

from transformers.configuration_utils import PushToHubMixin
from flatten_dict import flatten
import pandas as pd

LOGGER = getLogger("report")


@dataclass
class BenchmarkMeasurements:
    max_memory: Optional[MaxMemory] = None
    latency: Optional[Latency] = None
    throughput: Optional[Throughput] = None
    energy: Optional[Energy] = None
    efficiency: Optional[Efficiency] = None

    @staticmethod
    def aggregate(measurements: List["BenchmarkMeasurements"]) -> "BenchmarkMeasurements":
        max_memory = MaxMemory.aggregate([m.max_memory for m in measurements if m.max_memory is not None])
        latency = Latency.aggregate([m.latency for m in measurements if m.latency is not None])
        throughput = Throughput.aggregate([m.throughput for m in measurements if m.throughput is not None])
        energy = Energy.aggregate([m.energy for m in measurements if m.energy is not None])
        efficiency = Efficiency.aggregate([m.efficiency for m in measurements if m.efficiency is not None])

        return BenchmarkMeasurements(
            max_memory=max_memory,
            latency=latency,
            throughput=throughput,
            energy=energy,
            efficiency=efficiency,
        )


@dataclass
class BenchmarkReport(PushToHubMixin):
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            kwargs["token"] = use_auth_token

        config_file_name = config_file_name if config_file_name is not None else "benchmark_report.json"

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        output_config_file = os.path.join(save_directory, config_file_name)
        self.to_json(output_config_file)

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_flat_dict(self) -> dict:
        report_dict = self.to_dict()
        return flatten(report_dict, reducer="dot")

    def to_json(self, path: str, flat: bool = False) -> None:
        if flat:
            with open(path, "w") as f:
                dump(self.to_flat_dict(), f, indent=4)
        else:
            with open(path, "w") as f:
                dump(self.to_dict(), f, indent=4)

    def to_dataframe(self) -> pd.DataFrame:
        flat_report_dict = self.to_flat_dict()
        return pd.DataFrame(flat_report_dict, index=[0])

    def to_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def log_max_memory(self):
        for target in self.to_dict().keys():
            benchmark_measurements: BenchmarkMeasurements = getattr(self, target)
            if benchmark_measurements.max_memory is not None:
                benchmark_measurements.max_memory.log(prefix=target)

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

    def log_all(self):
        self.log_max_memory()
        self.log_latency()
        self.log_throughput()
        self.log_energy()
        self.log_efficiency()

    @classmethod
    def aggregate(cls, reports: List["BenchmarkReport"]) -> "BenchmarkReport":
        aggregated_report = cls()
        for target in aggregated_report.to_dict().keys():
            measurements = [getattr(report, target) for report in reports]
            aggregated_measurements = BenchmarkMeasurements.aggregate(measurements)
            setattr(aggregated_report, target, aggregated_measurements)

        return aggregated_report
