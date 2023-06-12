import re
import os
import torch
import random
import psutil
import platform
import subprocess
import numpy as np
import py3nvml.py3nvml as nvml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bytes_to_mega_bytes(bytes: int) -> int:
    return bytes >> 20


def get_device_name(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            return "CUDA not available"
        else:
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = nvml.nvmlDeviceGetName(handle)
            nvml.nvmlShutdown()
            return gpu_name

    elif device == "cpu":
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":
            os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
            command = "sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1)

    else:
        raise ValueError(f"Unknown device '{device}'")


def get_total_memory(device: str):
    if device == "cuda":
        if not torch.cuda.is_available():
            return "CUDA not available"
        else:
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            total_gpu_memory = bytes_to_mega_bytes(nvml.nvmlDeviceGetMemoryInfo(handle).total)  # type: ignore
            nvml.nvmlShutdown()
            return total_gpu_memory

    elif device == "cpu":
        total_cpu_ram_memory = bytes_to_mega_bytes(psutil.virtual_memory().total)
        return total_cpu_ram_memory

    else:
        raise ValueError(f"Unknown device '{device}'")


def get_used_memory(device: str):
    if device == "cuda":
        if not torch.cuda.is_available():
            return "CUDA not available"
        else:
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            used_gpu_memory = bytes_to_mega_bytes(nvml.nvmlDeviceGetMemoryInfo(handle).used)  # type: ignore
            nvml.nvmlShutdown()
            return used_gpu_memory

    elif device == "cpu":
        used_cpu_ram_memory = bytes_to_mega_bytes(psutil.virtual_memory().used)
        return used_cpu_ram_memory

    else:
        raise ValueError(f"Unknown device '{device}'")
