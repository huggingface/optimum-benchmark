import os
from contextlib import contextmanager
from logging import getLogger

from codecarbon import EmissionsTracker

LOGGER = getLogger("latency_tracker")


class EnergyTracker:
    def __init__(self):
        self.total_energy: float = 0
        self.total_emissions: float = 0

    @contextmanager
    def track(self, interval=1, file_prefix=""):
        self.emission_tracker = EmissionsTracker(
            measure_power_secs=interval,
            output_file=f"{file_prefix}_codecarbon.csv",
            gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES", None),
        )
        self.emission_tracker.start()
        yield
        self.emission_tracker.stop()
        self.total_energy = self.emission_tracker._total_energy.kWh
        self.total_emissions = self.emission_tracker.final_emissions

    def get_total_energy(self) -> float:
        return self.total_energy

    def get_total_emissions(self) -> float:
        return self.total_emissions

    def get_elapsed_time(self) -> float:
        return self.emission_tracker._last_measured_time - self.emission_tracker._start_time
