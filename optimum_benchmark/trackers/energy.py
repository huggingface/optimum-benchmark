import os
from contextlib import contextmanager
from logging import getLogger

from codecarbon import EmissionsTracker, OfflineEmissionsTracker

LOGGER = getLogger("latency_tracker")


class EnergyTracker:
    def __init__(self):
        self.total_energy: float = 0
        self.total_emissions: float = 0

    @contextmanager
    def track(self, interval=1, file_prefix=""):
        try:
            self.emission_tracker = EmissionsTracker(
                log_level="error",  # "info" for more verbosity
                tracking_mode="process",  # "machine" for machine-level tracking
                measure_power_secs=interval,
                output_file=f"{file_prefix}_codecarbon.csv",
                gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES", None),
            )
        except Exception as e:
            LOGGER.warning(f"Failed to initialize Online Emissions Tracker: {e}")
            LOGGER.info("Falling back to Offline Emissions Tracker")
            country_iso_code = os.environ.get("COUNTRY_ISO_CODE", None)
            if country_iso_code is None:
                raise ValueError(
                    "COUNTRY_ISO_CODE environment variable must be set when using Offline Emissions Tracker"
                )
            self.emission_tracker = OfflineEmissionsTracker(
                log_level="error",
                tracking_mode="process",
                measure_power_secs=interval,
                country_iso_code=country_iso_code,
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
