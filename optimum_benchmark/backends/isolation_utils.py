import os
import signal
import time
from typing import Dict, List

from ..env_utils import is_nvidia_system, is_rocm_system
from ..import_utils import is_amdsmi_available, is_py3nvml_available, torch_version


def check_cuda_isolation(devices_ids: List[int], isolated_pid: int) -> None:
    """
    Raises a RuntimeError if any process other than the benchmark process is running on the specified CUDA devices.
    """
    pids: Dict[int, set] = {}
    for device_id in devices_ids:
        pids[device_id] = set()

    if is_nvidia_system():
        if not is_py3nvml_available():
            raise ValueError(
                "check_no_process_is_running_on_cuda_device requires py3nvml. "
                "Please install it with `pip install py3nvml`."
            )
        import py3nvml.py3nvml as nvml

        nvml.nvmlInit()
        for device_id in devices_ids:
            device_handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
            device_processes = nvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
            for device_process in device_processes:
                pids[device_id].add(device_process.pid)

        nvml.nvmlShutdown()
    elif is_rocm_system():
        rocm_version = torch_version().split("rocm")[-1]

        if not is_amdsmi_available():
            raise ValueError(
                "check_no_process_is_running_on_cuda_device requires amdsmi. "
                "Please follow the instructions at https://github.com/RadeonOpenCompute/amdsmi/tree/master"
            )
        import amdsmi as smi

        smi.amdsmi_init()

        if rocm_version >= "5.7":
            # starting from rocm 5.7, the api seems to have changed names
            devices_handles = smi.amdsmi_get_processor_handles()
            for device_id in devices_ids:
                device_handle = devices_handles[device_id]
                processes_handles = smi.amdsmi_get_gpu_process_list(device_handle)
                for process_handle in processes_handles:
                    info = smi.amdsmi_get_gpu_process_info(device_handle, process_handle)
                    if info["memory_usage"]["vram_mem"] == 4096:
                        continue
                    pids[device_id].add(info["pid"])
        else:
            devices_handles = smi.amdsmi_get_device_handles()
            for device_id in devices_ids:
                device_handle = devices_handles[device_id]
                processes_handles = smi.amdsmi_get_process_list(device_handle)
                for process_handle in processes_handles:
                    info = smi.amdsmi_get_process_info(device_handle, process_handle)
                    if info["memory_usage"]["vram_mem"] == 4096:
                        continue
                    pids[device_id].add(info["pid"])

        smi.amdsmi_shut_down()
    else:
        raise ValueError("check_no_process_is_running_on_cuda_device is only supported on NVIDIA and AMD GPUs.")

    all_pids = set()
    for device_id in devices_ids:
        all_pids |= pids[device_id]
    other_pids = all_pids - {isolated_pid}

    if len(other_pids) > 0:
        error_message = f"Expected only process {isolated_pid} on device(s) {devices_ids}, but found {other_pids}."
        raise RuntimeError(error_message)


def check_cuda_continuous_isolation(devices_ids: List[int], isolated_pid: int) -> None:
    """
    Kills the benchmark process if any other process is running on the specified CUDA devices.
    """

    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    devices_ids = [int(device_id) for device_id in CUDA_VISIBLE_DEVICES.split(",")]

    while True:
        try:
            check_cuda_isolation(devices_ids, isolated_pid)
            time.sleep(0.1)
        except Exception as exception:
            os.kill(isolated_pid, signal.SIGTERM)
            raise exception
