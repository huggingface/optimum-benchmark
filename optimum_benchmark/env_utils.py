import re
import platform
import subprocess
import importlib.util
from typing import Optional

import psutil

from .import_utils import is_py3nvml_available, is_pyrsmi_available


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


def bytes_to_mega_bytes(bytes: int) -> int:
    # MB, not MiB
    # Reference: https://en.wikipedia.org/wiki/Byte#Multiple-byte_units
    return int(bytes * 1e-6)


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
        if not is_py3nvml_available():
            raise ValueError(
                "The library py3nvml is required to collect information on NVIDIA GPUs, but is not installed. "
                "Please install it through `pip install py3nvml`."
            )
        import py3nvml.py3nvml as nvml

        gpus = []
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            gpus.append(nvml.nvmlDeviceGetName(handle))
        nvml.nvmlShutdown()
    elif is_rocm_system():
        if not is_pyrsmi_available():
            raise ValueError(
                "The library pyrsmi is required to collect information on ROCm-powered GPUs, but is not installed. "
                "Please install it following the instructions https://github.com/RadeonOpenCompute/pyrsmi."
            )
        from pyrsmi import rocml

        rocml.smi_initialize()

        device_count = rocml.smi_get_device_count()

        gpus = [rocml.smi_get_device_name(index) for index in range(device_count)]
        rocml.smi_shutdown()
    else:
        gpus = []

    return gpus


def get_git_revision_hash(package_name: str, path: Optional[str] = None) -> Optional[str]:
    """
    Returns the git commit SHA of a package installed from a git repository.
    """

    if path is None:
        try:
            path = importlib.util.find_spec(package_name).origin
        except Exception:
            return None

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
    except Exception:
        git_hash = None

    return git_hash
