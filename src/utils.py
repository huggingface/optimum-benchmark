import py3nvml.py3nvml as nvml


def bytes_to_mega_bytes(bytes) -> int:
    return bytes >> 20


def get_gpu_name():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = nvml.nvmlDeviceGetName(handle)
    nvml.nvmlShutdown()
    return gpu_name


def get_total_gpu_memory():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    total_gpu_memory = bytes_to_mega_bytes(nvml.nvmlDeviceGetMemoryInfo(handle).total)  # type: ignore
    nvml.nvmlShutdown()
    return total_gpu_memory


def get_used_gpu_memory():
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    gpu_ram_mb = bytes_to_mega_bytes(nvml.nvmlDeviceGetMemoryInfo(handle).used)  # type: ignore
    nvml.nvmlShutdown()
    return gpu_ram_mb
