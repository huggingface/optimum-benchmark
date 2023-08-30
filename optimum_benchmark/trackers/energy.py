from contextlib import contextmanager
from logging import getLogger
from typing import Dict

from codecarbon import EmissionsTracker

LOGGER = getLogger("latency_tracker")


class EnergyTracker:
    def __init__(self):
        self.emissions: float = 0

    @contextmanager
    def track(self, interval=1, file_prefix=""):
        self.emission_tracker = EmissionsTracker(
            measure_power_secs=interval, output_file=f"{file_prefix}_emissions.csv"
        )
        self.emission_tracker.start()
        yield
        self.emission_tracker.stop()

    def get_energies(self) -> Dict[str, float]:
        return {
            "total_energy": self.emission_tracker._total_energy.kWh,
            "cpu_energy": self.emission_tracker._total_cpu_energy.kWh,
            "gpu_energy": self.emission_tracker._total_gpu_energy.kWh,
            "ram_energy": self.emission_tracker._total_ram_energy.kWh,
        }

    def get_emissions(self) -> float:
        return self.emission_tracker._emissions

    def get_elapsed_time(self) -> float:
        return self.emission_tracker._last_measured_time - self.emission_tracker._start_time
