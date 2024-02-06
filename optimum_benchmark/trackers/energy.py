import os
from logging import getLogger
from typing import List, Optional
from contextlib import contextmanager

from ..env_utils import is_nvidia_system, is_rocm_system
from ..import_utils import (
    is_py3nvml_available,
    is_pyrsmi_available,
    is_codecarbon_available,
)

if is_codecarbon_available():
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker

if is_nvidia_system():
    if is_py3nvml_available():
        import py3nvml.py3nvml as nvml
    else:
        raise ValueError(
            "The library py3nvml is required to run energy benchmark on NVIDIA GPUs, but is not installed. "
            "Please install it through `pip install py3nvml`."
        )

if is_rocm_system():
    if is_pyrsmi_available():
        # TODO: use amdsmi instead of pyrsmi
        from pyrsmi import rocml
    else:
        raise ValueError(
            "The library pyrsmi is required to run energy benchmark on ROCm-powered GPUs, but is not installed. "
            "Please install it through `pip install pyrsmi@git+https://github.com/RadeonOpenCompute/pyrsmi.git."
        )


LOGGER = getLogger("energy")


class EnergyTracker:
    def __init__(self, device_ids: Optional[List[int]] = None):
        self.device_ids = device_ids

        self.total_energy: float = 0
        self.total_emissions: float = 0

        if self.device_ids is None:
            self.device_ids = infer_cuda_device_ids()

    @contextmanager
    def track(self, interval=1, file_prefix=""):
        if not is_codecarbon_available():
            raise ValueError(
                "The library codecarbon is required to run energy benchmark, but is not installed. "
                "Please install it through `pip install codecarbon`."
            )

        try:
            self.emission_tracker = EmissionsTracker(
                log_level="error",  # "info" for more verbosity
                tracking_mode="process",  # "machine" for machine-level tracking
                gpu_ids=self.device_ids,
                measure_power_secs=interval,
                output_file=f"{file_prefix}_codecarbon.csv",
            )
        except Exception as e:
            LOGGER.warning(f"Failed to initialize Online Emissions Tracker: {e}")
            LOGGER.warning("Falling back to Offline Emissions Tracker")
            if os.environ.get("COUNTRY_ISO_CODE", None) is None:
                LOGGER.warning(
                    "Offline Emissions Tracker requires COUNTRY_ISO_CODE to be set. "
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
        self.total_energy = self.emission_tracker._total_energy.kWh
        self.total_emissions = self.emission_tracker.final_emissions

    def get_total_energy(self) -> float:
        return self.total_energy

    def get_total_emissions(self) -> float:
        return self.total_emissions

    def get_elapsed_time(self) -> float:
        return self.emission_tracker._last_measured_time - self.emission_tracker._start_time


def infer_cuda_device_ids() -> List[int]:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None:
        cuda_device_ids = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
    else:
        if is_nvidia_system() and is_py3nvml_available():
            nvml.nvmlInit()
            cuda_device_ids = list(range(nvml.nvmlDeviceGetCount()))
            nvml.nvmlShutdown()
        elif is_rocm_system() and is_pyrsmi_available():
            rocml.smi_initialize()
            cuda_device_ids = list(range(rocml.smi_get_device_count()))
            rocml.smi_shutdown()
        else:
            raise ValueError("Only NVIDIA and AMD ROCm GPUs are supported for CUDA energy tracking.")

    return cuda_device_ids
