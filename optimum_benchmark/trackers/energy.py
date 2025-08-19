import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import List, Literal, Optional, Union

from rich.console import Console
from rich.markdown import Markdown

from ..import_utils import is_codecarbon_available, is_torch_available

if is_torch_available():
    import torch

if is_codecarbon_available():
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    from codecarbon.output import EmissionsData

CONSOLE = Console()
LOGGER = getLogger("energy")

POWER_UNIT = "W"
ENERGY_UNIT = "kWh"
POWER_CONSUMPTION_SAMPLING_RATE = 1  # in seconds

Energy_Unit_Literal = Literal["kWh"]
Efficiency_Unit_Literal = Literal["samples/kWh", "tokens/kWh", "images/kWh"]


@dataclass
class Energy:
    unit: Energy_Unit_Literal

    cpu: float
    ram: float
    gpu: float
    total: float

    def __sub__(self, other: "Energy") -> "Energy":
        """Enables subtraction of two Energy instances using the '-' operator."""

        if self.unit != other.unit:
            raise ValueError("Energy units must match to perform subtraction")

        return Energy(
            unit=self.unit,
            cpu=self.cpu - other.cpu,
            gpu=self.gpu - other.gpu,
            ram=self.ram - other.ram,
            total=self.total - other.total,
        )

    def __truediv__(self, scalar: float) -> "Energy":
        return Energy(
            unit=self.unit,
            cpu=self.cpu / scalar,
            gpu=self.gpu / scalar,
            ram=self.ram / scalar,
            total=self.total / scalar,
        )

    @staticmethod
    def aggregate_across_processes(energies: List[Optional["Energy"]]) -> Optional["Energy"]:
        if len(energies) == 0:
            raise ValueError("No energy measurements to aggregate")
        elif any(energy is None for energy in energies):
            raise ValueError("Some energy measurements are missing")

        # since measurements are machine-level, we just take the average
        total = sum(energy.total for energy in energies) / len(energies)
        cpu = sum(energy.cpu for energy in energies) / len(energies)
        gpu = sum(energy.gpu for energy in energies) / len(energies)
        ram = sum(energy.ram for energy in energies) / len(energies)
        unit = energies[0].unit

        return Energy(cpu=cpu, gpu=gpu, ram=ram, total=total, unit=unit)

    def to_plain_text(self) -> str:
        plain_text = ""
        plain_text += "\t\t+ cpu: {cpu:f} ({unit})\n"
        plain_text += "\t\t+ gpu: {gpu:f} ({unit})\n"
        plain_text += "\t\t+ ram: {ram:f} ({unit})\n"
        plain_text += "\t\t+ total: {total:f} ({unit})\n"
        return plain_text.format(**asdict(self))

    def log(self):
        for line in self.to_plain_text().split("\n"):
            if line:
                LOGGER.info(line)

    def to_markdown_text(self) -> str:
        markdown_text = ""
        markdown_text += "| metric     |     value |   unit |\n"
        markdown_text += "| :--------- | --------: | -----: |\n"
        markdown_text += "| cpu        |   {cpu:f} | {unit} |\n"
        markdown_text += "| gpu        |   {gpu:f} | {unit} |\n"
        markdown_text += "| ram        |   {ram:f} | {unit} |\n"
        markdown_text += "| total      | {total:f} | {unit} |\n"
        return markdown_text.format(**asdict(self))

    def print(self):
        CONSOLE.print(Markdown(self.to_markdown_text()))


@dataclass
class Efficiency:
    unit: Efficiency_Unit_Literal

    value: float

    @staticmethod
    def aggregate_across_processes(efficiencies: List[Optional["Efficiency"]]) -> Optional["Efficiency"]:
        if len(efficiencies) == 0:
            raise ValueError("No efficiency measurements to aggregate")
        elif any(efficiency is None for efficiency in efficiencies):
            raise ValueError("Some efficiency measurements are None")

        # since measurements are machine-level, we just take the average
        value = sum(efficiency.value for efficiency in efficiencies) / len(efficiencies)
        unit = efficiencies[0].unit

        return Efficiency(value=value, unit=unit)

    @staticmethod
    def from_energy(energy: "Energy", volume: int, unit: str) -> "Efficiency":
        return Efficiency(value=volume / energy.total if energy.total > 0 else 0, unit=unit)

    def to_plain_text(self) -> str:
        plain_text = ""
        plain_text += "\t\t+ efficiency: {value:f} ({unit})\n"
        return plain_text.format(**asdict(self))

    def log(self):
        for line in self.to_plain_text().split("\n"):
            if line:
                LOGGER.info(line)

    def to_markdown_text(self) -> str:
        markdown_text = ""
        markdown_text += "| metric     |     value |   unit |\n"
        markdown_text += "| :--------- | --------: | -----: |\n"
        markdown_text += "| efficiency | {value:f} | {unit} |\n"
        return markdown_text.format(**asdict(self))

    def print(self):
        CONSOLE.print(Markdown(self.to_markdown_text()))


class EnergyTracker:
    def __init__(self, device: str, backend: str, device_ids: Optional[Union[str, int, List[int]]] = None):
        self.device = device
        self.backend = backend
        self.device_ids = device_ids

        self.is_gpu = self.device == "cuda"
        self.is_pytorch_cuda = (self.backend, self.device) == ("pytorch", "cuda")

        LOGGER.info("\t\t+ Tracking RAM and CPU energy consumption")

        if self.is_gpu:
            if isinstance(self.device_ids, str):
                self.device_ids = list(map(int, self.device_ids.split(",")))
            elif isinstance(self.device_ids, int):
                self.device_ids = [self.device_ids]
            elif isinstance(self.device_ids, list):
                self.device_ids = self.device_ids
            elif self.device_ids is None:
                raise ValueError("GPU device IDs must be provided for energy tracking on GPUs")
            else:
                raise ValueError("GPU device IDs must be a string, an integer, or a list of integers")

            LOGGER.info(f"\t\t+ Tracking GPU energy consumption on devices {self.device_ids}")

        if not is_codecarbon_available():
            raise ValueError(
                "The library codecarbon is required to run energy benchmark, but is not installed. "
                "Please install it through `pip install codecarbon`."
            )

        try:
            self.emission_tracker = EmissionsTracker(
                log_level="warning",
                # tracking_mode="process" only tries to track memory consumption of current process
                # but computes cpu and gpu energy consumption based on the machine-level tracking
                tracking_mode="machine",
                gpu_ids=self.device_ids,
                # allow multiple trackers to run in the same machine (e.g., for distributed inference/training)
                # and also for testing purposes (we run many benchmarks in parallel)
                # https://github.com/mlco2/codecarbon/pull/562 added this feature
                # but it doesn't explain why one tracker is better than multiple
                allow_multiple_runs=True,
                output_file="codecarbon.csv",
                measure_power_secs=POWER_CONSUMPTION_SAMPLING_RATE,
            )
        except Exception:
            LOGGER.warning("\t\t+ Falling back to Offline Emissions Tracker")

            if os.environ.get("COUNTRY_ISO_CODE", None) is None:
                LOGGER.warning(
                    "\t\t+ Offline Emissions Tracker requires COUNTRY_ISO_CODE to be set. "
                    "We will set it to USA but the carbon footprint might be inaccurate."
                )

            self.emission_tracker = OfflineEmissionsTracker(
                log_level="warning",
                # tracking_mode="process" only tries to track memory consumption of current process
                # but computes cpu and gpu energy consumption based on the machine-level tracking
                tracking_mode="machine",
                gpu_ids=self.device_ids,
                # allow multiple trackers to run in the same machine (e.g., for distributed inference/training)
                # and also for testing purposes (we run many benchmarks in parallel)
                # https://github.com/mlco2/codecarbon/pull/562 added this feature
                # but it doesn't explain why one tracker is better than multiple
                allow_multiple_runs=True,
                output_file="codecarbon.csv",
                measure_power_secs=POWER_CONSUMPTION_SAMPLING_RATE,
                country_iso_code=os.environ.get("COUNTRY_ISO_CODE", "USA"),
            )

        self.total_energy: Optional[float] = None
        self.cpu_energy: Optional[float] = None
        self.gpu_energy: Optional[float] = None
        self.ram_energy: Optional[float] = None

    def reset(self):
        self.total_energy = None
        self.cpu_energy = None
        self.gpu_energy = None
        self.ram_energy = None

    @contextmanager
    def track(self, task_name: str = "task"):
        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

        self.emission_tracker.start_task(task_name=task_name)

        yield

        if self.is_pytorch_cuda:
            torch.cuda.synchronize()

        emission_data: EmissionsData = self.emission_tracker.stop_task()

        if emission_data is None:
            raise ValueError(
                "Energy tracking failed. Please ensure that the codecarbon library is properly installed and configured."
            )

        with open(f"{task_name}_codecarbon.json", "w") as f:
            LOGGER.info(f"\t\t+ Saving codecarbon emission data to {task_name}_codecarbon.json")
            json.dump(asdict(emission_data), f, indent=4)

        self.total_energy = emission_data.energy_consumed
        self.cpu_energy = emission_data.cpu_energy
        self.gpu_energy = emission_data.gpu_energy
        self.ram_energy = emission_data.ram_energy

    def get_energy(self) -> Energy:
        assert self.total_energy is not None, "Energy must be tracked before calling this method"

        return Energy(
            unit=ENERGY_UNIT, cpu=self.cpu_energy, gpu=self.gpu_energy, ram=self.ram_energy, total=self.total_energy
        )
