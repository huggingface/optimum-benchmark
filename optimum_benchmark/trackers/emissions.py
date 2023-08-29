from contextlib import contextmanager
from logging import getLogger

from codecarbon import EmissionsTracker as CodeCarbonEmissionsTracker

LOGGER = getLogger("latency_tracker")


class EmissionsTracker:
    def __init__(self):
        self.emissions: float = 0.0

    @contextmanager
    def track(self, interval=1, file_prefix=""):
        emission_tracker = CodeCarbonEmissionsTracker(
            measure_power_secs=interval,
            output_file=f"{file_prefix}_emissions.csv",
        )
        emission_tracker.start()
        yield
        self.emissions = emission_tracker.stop()

    def get_emissions(self):
        return self.emissions
