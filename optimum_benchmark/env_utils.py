import os
import re
import platform
import subprocess
from typing import Optional, List

from .import_utils import is_pynvml_available, is_amdsmi_available, torch_version

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


if is_nvidia_system():
    if is_pynvml_available():
        import pynvml as pynvml

if is_rocm_system():
    if is_amdsmi_available():
        import amdsmi as amdsmi


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
    return psutil.virtual_memory().total / 1e6


def get_gpus():
    if is_nvidia_system():
        if not is_pynvml_available():
            raise ValueError(
                "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
            )

        gpus = []
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpus.append(pynvml.nvmlDeviceGetName(handle))
        pynvml.nvmlShutdown()
    elif is_rocm_system():
        if not is_amdsmi_available():
            raise ValueError(
                "The library amdsmi is required to run memory benchmark on AMD GPUs, but is not installed. "
                "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
            )

        gpus = []
        amdsmi.amdsmi_init()
        rocm_version = torch_version().split("rocm")[-1]

        if rocm_version >= "5.7":
            devices_handles = amdsmi.amdsmi_get_processor_handles()
            for device_handle in devices_handles:
                gpus.append(amdsmi.amdsmi_get_gpu_vendor_name(device_handle))
        else:
            devices_handles = amdsmi.amdsmi_get_device_handles()
            for device_handle in devices_handles:
                gpus.append(amdsmi.amdsmi_dev_get_vendor_name(device_handle))

        amdsmi.amdsmi_shut_down()
    else:
        raise ValueError("No NVIDIA or ROCm GPUs found.")

    return gpus


def get_gpu_vram_mb() -> List[int]:
    if is_nvidia_system():
        if not is_pynvml_available():
            raise ValueError(
                "The library pynvml is required to run memory benchmark on NVIDIA GPUs, but is not installed. "
                "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
            )

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        vrams = [
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total for i in range(device_count)
        ]
        pynvml.nvmlShutdown()
    elif is_rocm_system():
        if not is_amdsmi_available():
            raise ValueError(
                "The library amdsmi is required to run memory benchmark on AMD GPUs, but is not installed. "
                "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
            )

        amdsmi.amdsmi_init()
        rocm_version = torch_version().split("rocm")[-1]

        if rocm_version >= "5.7":
            device_handles = amdsmi.amdsmi_get_processor_handles()
            vrams = [
                amdsmi.amdsmi_get_gpu_memory_total(device_handle, mem_type=amdsmi.AmdSmiMemoryType.VRAM)
                for device_handle in device_handles
            ]
        else:
            device_handles = amdsmi.amdsmi_get_device_handles()
            vrams = [
                amdsmi.amdsmi_dev_get_memory_total(device_handle, mem_type=amdsmi.AmdSmiMemoryType.VRAM)
                for device_handle in device_handles
            ]

        amdsmi.amdsmi_shut_down()

    else:
        raise ValueError("No NVIDIA or ROCm GPUs found.")

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

            pynvml.nvmlInit()
            device_ids = list(range(pynvml.nvmlDeviceGetCount()))
            pynvml.nvmlShutdown()
        elif is_rocm_system():
            if not is_amdsmi_available():
                raise ValueError(
                    "The library amdsmi is required to run memory benchmark on AMD GPUs, but is not installed. "
                    "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
                )

            amdsmi.amdsmi_init()
            rocm_version = torch_version().split("rocm")[-1]

            if rocm_version >= "5.7":
                device_ids = list(range(len(amdsmi.amdsmi_get_processor_handles())))
            else:
                device_ids = list(range(len(amdsmi.amdsmi_get_device_handles())))

            amdsmi.amdsmi_shut_down()
        else:
            raise ValueError("No NVIDIA or ROCm GPUs found.")

        device_ids = ",".join(str(i) for i in device_ids)

    return device_ids


def get_system_info() -> dict:
    system_dict = {
        "cpu": get_cpu(),
        "cpu_count": os.cpu_count(),
        "cpu_ram_mb": get_cpu_ram_mb(),
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    if is_nvidia_system() or is_rocm_system():
        system_dict["gpu"] = get_gpus()
        system_dict["gpu_count"] = len(get_gpus())
        system_dict["gpu_vram_mb"] = get_gpu_vram_mb()

    return system_dict
