import os
import signal
import time
from typing import Dict, List

from ..env_utils import is_nvidia_system, is_rocm_system
from ..import_utils import is_py3nvml_available, is_pyrsmi_available


def only_this_process_is_running_on_cuda_devices(cuda_devices: List[int], benchmark_pid: int) -> None:
    """
    Raises a RuntimeError if any process other than the benchmark process is running on the specified CUDA devices.
    """
    pids: Dict[int, set] = {}
    for device_id in cuda_devices:
        pids[device_id] = set()

    if is_nvidia_system():
        if not is_py3nvml_available():
            raise ValueError(
                "check_no_process_is_running_on_cuda_device requires py3nvml. "
                "Please install it with `pip install py3nvml`."
            )
        import py3nvml.py3nvml as nvml

        nvml.nvmlInit()
        for device_id in cuda_devices:
            device_handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
            pids[device_id] = set(nvml.nvmlDeviceGetComputeRunningProcesses(device_handle))
        nvml.nvmlShutdown()
    elif is_rocm_system():
        if not is_pyrsmi_available():
            raise ValueError(
                "check_no_process_is_running_on_cuda_device requires pyrsmi. "
                "Please follow the instructions at https://github.com/RadeonOpenCompute/amdsmi/tree/master"
            )
        import amdsmi as rocml

        rocml.amdsmi_init()
        devices_handles = rocml.amdsmi_get_device_handles()
        for device_id in cuda_devices:
            device_handle = devices_handles[device_id]
            processes_handles = rocml.amdsmi_get_process_list(device_handle)
            for process_handle in processes_handles:
                info = rocml.amdsmi_get_process_info(device_handle, process_handle)
                if info["memory_usage"]["vram_mem"] == 4096:
                    continue
                pids[device_id].add(info["pid"])
        rocml.amdsmi_shut_down()
    else:
        raise ValueError("check_no_process_is_running_on_cuda_device is only supported on NVIDIA and AMD GPUs.")

    all_pids = set()
    for device_id in cuda_devices:
        all_pids |= pids[device_id]
    other_pids = all_pids - {benchmark_pid}

    if len(other_pids) > 0:
        error_message = f"Expected only process {benchmark_pid} on device(s) {cuda_devices}, but found {other_pids}."

        # for pid in other_pids:
        #     error_message += f"\nProcess {pid} info: {get_pid_info(pid)}"

        raise RuntimeError(error_message)


def only_this_process_will_run_on_cuda_devices(cuda_devices: List[int], benchmark_pid: int) -> None:
    """
    Kills the benchmark process if any other process is running on the specified CUDA devices.
    """
    while True:
        try:
            only_this_process_is_running_on_cuda_devices(cuda_devices, benchmark_pid)
            time.sleep(0.1)
        except RuntimeError as exception:
            os.kill(benchmark_pid, signal.SIGTERM)
            raise exception


## we can report more information about the process to explain the source of the error
## but that might be dangerous in a CI context

# import psutil

# def get_pid_info(pid: int) -> Dict[str, str]:
#     """Returns a dictionary containing the process' information."""

#     process = psutil.Process(pid)

#     return {
#         "pid": pid,
#         "name": process.name(),
#         "username": process.username(),
#         "cmdline": " ".join(process.cmdline()),
#     }
