import psutil
import torch
import py3nvml.py3nvml as nvml


def bytes_to_mega_bytes(bytes) -> int:
    return bytes >> 20


def get_gpu_name():
    if not torch.cuda.is_available():
        return "CUDA not available"
    else:
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = nvml.nvmlDeviceGetName(handle)
        nvml.nvmlShutdown()
        return gpu_name


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
