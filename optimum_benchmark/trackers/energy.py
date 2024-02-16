import os
from logging import getLogger
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional, Literal, List

from ..env_utils import get_cuda_device_ids
from ..import_utils import is_codecarbon_available

if is_codecarbon_available():
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker  # type: ignore

LOGGER = getLogger("energy")

ENERGY_UNIT = "kWh"
Energy_Unit_Literal = Literal["kWh"]
Efficiency_Unit_Literal = Literal["samples/kWh", "tokens/kWh", "images/kWh"]


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

        cpu = sum(energy.cpu for energy in energies)
        gpu = sum(energy.gpu for energy in energies)
        ram = sum(energy.ram for energy in energies)
        total = sum(energy.total for energy in energies)

        return Energy(cpu=cpu, gpu=gpu, ram=ram, total=total, unit=ENERGY_UNIT)

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} CPU energy: {self.cpu:f} ({self.unit})")
        LOGGER.info(f"\t\t+ {prefix} GPU energy: {self.gpu:f} ({self.unit})")
        LOGGER.info(f"\t\t+ {prefix} RAM energy: {self.ram:f} ({self.unit})")
        LOGGER.info(f"\t\t+ {prefix} total energy: {self.total:f} ({self.unit})")


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

    def log(self, prefix: str = "forward"):
        LOGGER.info(f"\t\t+ {prefix} efficiency: {self.value:f} ({self.unit})")


class EnergyTracker:
    def __init__(self, device: str, device_ids: Optional[str] = None):
        self.device = device
        self.device_ids = device_ids

        self.cpu_energy: float = 0
        self.gpu_energy: float = 0
        self.ram_energy: float = 0
        self.total_energy: float = 0

        if self.device == "cuda":
            if self.device_ids is None:
                LOGGER.warning("\t+ `device=cuda` but `device_ids` not provided. Using all available CUDA devices.")
                self.device_ids = get_cuda_device_ids()

            self.device_ids = list(map(int, self.device_ids.split(",")))
            LOGGER.info(f"\t+ Tracking GPU energy on devices {self.device_ids}")

    def reset(self):
        self.cpu_energy = 0
        self.gpu_energy = 0
        self.ram_energy = 0
        self.total_energy = 0

    @contextmanager
    def track(self, interval=1, file_prefix="method"):
        if not is_codecarbon_available():
            raise ValueError(
                "The library codecarbon is required to run energy benchmark, but is not installed. "
                "Please install it through `pip install codecarbon`."
            )

        try:
            # TODO: use pynvml and amdsmi directly to get the GPU power consumption
            self.emission_tracker = EmissionsTracker(
                log_level="error",  # "info" for more verbosity
                tracking_mode="process",  # "machine" for machine-level tracking
                gpu_ids=self.device_ids,
                measure_power_secs=interval,
                output_file=f"{file_prefix}_codecarbon.csv",
            )
        except Exception as e:
            LOGGER.warning("\t+ Failed to initialize Online Emissions Tracker:, %s", e)
            LOGGER.warning("\t+ Falling back to Offline Emissions Tracker")
            if os.environ.get("COUNTRY_ISO_CODE", None) is None:
                LOGGER.warning(
                    "\t+ Offline Emissions Tracker requires COUNTRY_ISO_CODE to be set. "
                    "We will set it to FRA but the carbon footprint will be inaccurate."
                )

            self.emission_tracker = OfflineEmissionsTracker(
                log_level="error",
                tracking_mode="process",
                gpu_ids=self.device_ids,
                measure_power_secs=interval,
                output_file=f"{file_prefix}_codecarbon.csv",
                country_iso_code=os.environ.get("COUNTRY_ISO_CODE", "FRA"),
            )

        self.emission_tracker.start()
        yield
        self.emission_tracker.stop()

        self.cpu_energy = self.emission_tracker._total_cpu_energy.kWh
        self.gpu_energy = self.emission_tracker._total_gpu_energy.kWh
        self.ram_energy = self.emission_tracker._total_ram_energy.kWh
        self.total_energy = self.emission_tracker._total_energy.kWh

    def get_elapsed_time(self) -> float:
        return self.emission_tracker._last_measured_time - self.emission_tracker._start_time

    def get_energy(self) -> Energy:
        return Energy(
            unit=ENERGY_UNIT,
            cpu=self.cpu_energy,
            gpu=self.gpu_energy,
            ram=self.ram_energy,
            total=self.total_energy,
        )
