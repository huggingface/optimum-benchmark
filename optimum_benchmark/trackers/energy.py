import os
from logging import getLogger
from contextlib import contextmanager
from typing import Optional, Dict

from ..env_utils import get_cuda_device_ids
from ..import_utils import is_codecarbon_available

if is_codecarbon_available():
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker


LOGGER = getLogger("energy")


class EnergyTracker:
    def __init__(self, device: str, device_ids: Optional[str] = None):
        self.device = device

        self.cpu_energy: float = 0
        self.gpu_energy: float = 0
        self.ram_energy: float = 0
        self.total_energy: float = 0

        if self.device == "cuda":
            if device_ids is None:
                LOGGER.warning("\t+ `device=cuda` but `device_ids` not provided. Using all available CUDA devices.")
                self.device_ids = list(map(int, get_cuda_device_ids().split(",")))
            else:
                self.device_ids = list(map(int, device_ids.split(",")))
        else:
            self.device_ids = []

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

    def get_energies_dict(self) -> Dict[str, float]:
        return {
            "cpu_energy(kHh)": self.cpu_energy,
            "gpu_energy(kHh)": self.gpu_energy,
            "ram_energy(kHh)": self.ram_energy,
            "total(kHh)": self.total_energy,
        }
