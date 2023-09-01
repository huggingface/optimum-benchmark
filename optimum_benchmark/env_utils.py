import platform
import re
import subprocess
from logging import getLogger
from typing import Optional

import psutil

from .import_utils import is_py3nvml_available

LOGGER = getLogger("utils")


def bytes_to_mega_bytes(bytes: int) -> int:
    # Reference: https://en.wikipedia.org/wiki/Byte#Multiple-byte_units
    return int(bytes * 1e-6)


def get_cpu() -> Optional[str]:
    if platform.system() == "Windows":
        return platform.processor()

    elif platform.system() == "Darwin":
        command = "sysctl -n machdep.cpu.brand_string"
        return str(subprocess.check_output(command).strip())

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
    if is_py3nvml_available():
        import py3nvml.py3nvml as nvml

        gpus = []
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            gpus.append(nvml.nvmlDeviceGetName(handle))
        nvml.nvmlShutdown()
    else:
        gpus = ["py3nvml not available"]

    return gpus
