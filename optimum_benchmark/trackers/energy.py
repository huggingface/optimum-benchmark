import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from json import dump
from logging import getLogger
from typing import List, Literal, Optional

from ..import_utils import is_codecarbon_available, is_torch_available, is_torch_distributed_available
from ..system_utils import get_gpu_device_ids

if is_torch_available():
    import torch

if is_torch_distributed_available():
    import torch.distributed

if is_codecarbon_available():
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    from codecarbon.output import EmissionsData

LOGGER = getLogger("energy")

POWER_UNIT = "W"
ENERGY_UNIT = "kWh"
Energy_Unit_Literal = Literal["kWh"]
Efficiency_Unit_Literal = Literal["samples/kWh", "tokens/kWh", "images/kWh"]

POWER_CONSUMPTION_SAMPLING_RATE = 1  # in seconds


@dataclass
class Energy:
    unit: Energy_Unit_Literal

    cpu: float
    ram: float
    gpu: float
    total: float

    @staticmethod
    def aggregate(energies: List["Energy"]) -> "Energy":
        if len(energies) == 0 or all(energy is None for energy in energies):
            return None
        elif any(energy is None for energy in energies):
            raise ValueError("Some energy measurements are missing")

        # since measurements are machine-level, we just take the average
        cpu = sum(energy.cpu for energy in energies) / len(energies)
        gpu = sum(energy.gpu for energy in energies) / len(energies)
        ram = sum(energy.ram for energy in energies) / len(energies)
        total = sum(energy.total for energy in energies) / len(energies)

        return Energy(cpu=cpu, gpu=gpu, ram=ram, total=total, unit=ENERGY_UNIT)

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} energy consumption:")
        LOGGER.info(f"\t\t\t+ CPU: {self.cpu:f} ({self.unit})")
        LOGGER.info(f"\t\t\t+ GPU: {self.gpu:f} ({self.unit})")
        LOGGER.info(f"\t\t\t+ RAM: {self.ram:f} ({self.unit})")
        LOGGER.info(f"\t\t\t+ total: {self.total:f} ({self.unit})")

    def __sub__(self, other: "Energy") -> "Energy":
        """Enables subtraction of two Energy instances using the '-' operator."""

        if self.unit != other.unit:
            raise ValueError("Energy units must match to perform subtraction")

        return Energy(
            cpu=self.cpu - other.cpu,
            gpu=self.gpu - other.gpu,
            ram=self.ram - other.ram,
            total=self.total - other.total,
            unit=self.unit,
        )

    def __truediv__(self, scalar: float) -> "Energy":
        return Energy(
            cpu=self.cpu / scalar,
            gpu=self.gpu / scalar,
            ram=self.ram / scalar,
            total=self.total / scalar,
            unit=self.unit,
        )


@dataclass
class Efficiency:
    unit: Efficiency_Unit_Literal

    value: float

    @staticmethod
    def aggregate(efficiencies: List["Efficiency"]) -> "Efficiency":
        if len(efficiencies) == 0:
            raise ValueError("No efficiency measurements to aggregate")
        elif any(efficiency is None for efficiency in efficiencies):
            raise ValueError("Some efficiency measurements are None")

        unit = efficiencies[0].unit
        value = sum(efficiency.value for efficiency in efficiencies) / len(efficiencies)

        return Efficiency(value=value, unit=unit)

    @staticmethod
    def from_energy(energy: "Energy", volume: int, unit: str) -> "Efficiency":
        return Efficiency(value=volume / energy.total if energy.total > 0 else 0, unit=unit)

    def log(self, prefix: str = "method"):
        LOGGER.info(f"\t\t+ {prefix} energy efficiency: {self.value:f} ({self.unit})")


class EnergyTracker:
    def __init__(self, backend: str, device: str, device_ids: Optional[str] = None):
        self.device = device
        self.backend = backend
        self.device_ids = device_ids
        self.is_asynchronous = backend == "pytorch" and device == "cuda"
        self.is_distributed = is_torch_distributed_available() and torch.distributed.is_initialized()

        if self.device == "cuda":
            if self.device_ids is None:
                LOGGER.warning("\t+ `device=cuda` but `device_ids` not provided. Using all available CUDA devices.")
                self.device_ids = get_gpu_device_ids()

            self.device_ids = list(map(int, self.device_ids.split(",")))
            LOGGER.info(f"\t+ Tracking GPU energy on devices {self.device_ids}")

        if not is_codecarbon_available():
            raise ValueError(
                "The library codecarbon is required to run energy benchmark, but is not installed. "
                "Please install it through `pip install codecarbon`."
            )

        try:
            # TODO: use pynvml and amdsmi directly to get the GPU power consumption
            self.emission_tracker = EmissionsTracker(
                log_level="warning",  # "info" for more verbosity
                tracking_mode="machine",  # "machine" for machine-level tracking
                gpu_ids=self.device_ids,
                output_file="codecarbon.csv",
                measure_power_secs=POWER_CONSUMPTION_SAMPLING_RATE,
            )
        except Exception as e:
            LOGGER.warning("\t+ Failed to initialize Online Emissions Tracker:, %s", e)
            LOGGER.warning("\t+ Falling back to Offline Emissions Tracker")
            if os.environ.get("COUNTRY_ISO_CODE", None) is None:
                LOGGER.warning(
                    "\t+ Offline Emissions Tracker requires COUNTRY_ISO_CODE to be set. "
                    "We will set it to USA but the carbon footprint might be inaccurate."
                )

            self.emission_tracker = OfflineEmissionsTracker(
                log_level="warning",  # "info" for more verbosity
                tracking_mode="machine",  # "machine" for machine-level tracking
                gpu_ids=self.device_ids,
                measure_power_secs=POWER_CONSUMPTION_SAMPLING_RATE,
                country_iso_code=os.environ.get("COUNTRY_ISO_CODE", "USA"),
            )

        self.cpu_energy = None
        self.gpu_energy = None
        self.ram_energy = None
        self.total_energy = None

    @contextmanager
    def track(self, file_prefix: str = "task"):
        if self.is_distributed:
            torch.distributed.barrier()

        if self.is_asynchronous:
            torch.cuda.synchronize()

        self.emission_tracker.start_task()

        yield

        if self.is_distributed:
            torch.distributed.barrier()

        if self.is_asynchronous:
            torch.cuda.synchronize()

        emission_data: EmissionsData = self.emission_tracker.stop_task()

        with open(f"{file_prefix}_codecarbon.json", "w") as f:
            LOGGER.info(f"\t+ Saving codecarbon emission data to {file_prefix}_codecarbon.json")
            dump(asdict(emission_data), f, indent=4)

        self.cpu_energy = emission_data.cpu_energy
        self.gpu_energy = emission_data.gpu_energy
        self.ram_energy = emission_data.ram_energy
        self.total_energy = emission_data.energy_consumed

    def get_energy(self) -> Energy:
        return Energy(
            unit=ENERGY_UNIT, cpu=self.cpu_energy, gpu=self.gpu_energy, ram=self.ram_energy, total=self.total_energy
        )
