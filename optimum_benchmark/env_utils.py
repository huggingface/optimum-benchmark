import os
import re
import platform
import subprocess
import importlib.util
from typing import Optional, List

from .import_utils import is_pynvml_available, is_amdsmi_available

import psutil


def is_nvidia_system():
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except Exception:
        return False


def is_rocm_system():
    try:
        subprocess.check_output("rocm-smi")
        return True
    except Exception:
        return False


def bytes_to_mega_bytes(bytes: int) -> float:
    # MB, not MiB
    return bytes / 1e6


def get_cpu() -> Optional[str]:
    if platform.system() == "Windows":
        return platform.processor()

    elif platform.system() == "Darwin":
        command = "sysctl -n machdep.cpu.brand_string"
        return str(subprocess.check_output(command, shell=True).decode().strip())

    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
        return "Could not find device name"

    else:
        raise ValueError(f"Unknown system '{platform.system()}'")


def get_cpu_ram_mb():
    return bytes_to_mega_bytes(psutil.virtual_memory().total)


def get_gpus():
    if is_nvidia_system():
        if not is_pynvml_available():
            raise ValueError(
                "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
            )
        import pynvml as nvml

        gpus = []
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            gpus.append(nvml.nvmlDeviceGetName(handle))
        nvml.nvmlShutdown()
    elif is_rocm_system():
        if not is_amdsmi_available():
            raise ValueError(
                "The library amdsmi is required to run memory benchmark on AMD GPUs, but is not installed. "
                "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
            )
        import amdsmi as rocml

        gpus = []
        rocml.amdsmi_init()
        devices_handles = rocml.amdsmi_get_processor_handles()
        for device_handle in devices_handles:
            gpus.append(rocml.amdsmi_get_gpu_vendor_name(device_handle))

        rocml.amdsmi_shut_down()
    else:
        gpus = []

    return gpus


def get_gpu_vram_mb() -> List[int]:
    if is_nvidia_system():
        if not is_pynvml_available():
            raise ValueError(
                "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
            )
        import pynvml as nvml

        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        vrams = [nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(i)).total for i in range(device_count)]
        nvml.nvmlShutdown()
    elif is_rocm_system():
        if not is_amdsmi_available():
            raise ValueError(
                "The library amdsmi is required to run memory benchmark on AMD GPUs, but is not installed. "
                "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
            )

        import amdsmi as rocml

        rocml.amdsmi_init()
        device_handles = rocml.amdsmi_get_processor_handles()
        vrams = [rocml.amdsmi_get_gpu_memory_total(device_handle) for device_handle in device_handles]
        rocml.amdsmi_shut_down()
    else:
        vrams = []

    return sum(vrams)


def get_cuda_device_ids() -> str:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        if is_nvidia_system():
            if not is_pynvml_available():
                raise ValueError(
                    "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                    "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
                )
            import pynvml as nvml

            nvml.nvmlInit()
            device_ids = list(range(nvml.nvmlDeviceGetCount()))
            nvml.nvmlShutdown()
        elif is_rocm_system():
            if not is_amdsmi_available():
                raise ValueError(
                    "The library amdsmi is required to run memory benchmark on AMD GPUs, but is not installed. "
                    "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
                )
            import amdsmi as rocml

            rocml.amdsmi_init()
            device_ids = len(rocml.amdsmi_get_processor_handles())
            rocml.amdsmi_shut_down()
        else:
            raise ValueError("No NVIDIA or ROCm GPUs found.")

    return ",".join(str(i) for i in device_ids)


def get_git_revision_hash(package_name: str) -> Optional[str]:
    """
    Returns the git commit SHA of a package installed from a git repository.
    """

    try:
        path = importlib.util.find_spec(package_name).origin
    except Exception:
        return None

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
    except Exception:
        return None

    return git_hash
