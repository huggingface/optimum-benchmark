import py3nvml.py3nvml as nvml


def bytes_to_mega_bytes(bytes) -> int:
    return bytes >> 20


def get_gpu():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    gpu = nvml.nvmlDeviceGetName(handle)
    nvml.nvmlShutdown()
    return gpu


def get_gpu_ram_mb():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    gpu_ram_mb = bytes_to_mega_bytes(nvml.nvmlDeviceGetMemoryInfo(handle).total)  # type: ignore
    nvml.nvmlShutdown()
    return gpu_ram_mb
