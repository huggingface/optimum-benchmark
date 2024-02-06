import os
import time
import signal
from typing import Dict, Set
from logging import getLogger
from multiprocessing import Process
from contextlib import contextmanager

import psutil

from ..logging_utils import setup_logging
from ..env_utils import is_nvidia_system, is_rocm_system
from ..import_utils import is_amdsmi_available, is_py3nvml_available, torch_version


if is_py3nvml_available():
    import py3nvml.py3nvml as nvml  # type: ignore

if is_amdsmi_available():
    import amdsmi  # type: ignore

LOGGER = getLogger("isolation")


def get_nvidia_devices_pids() -> Dict[int, list]:
    devices_pids: Dict[int, list] = {}
    devices_ids = [int(device_id) for device_id in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

    if not is_py3nvml_available():
        raise ValueError("get_nvidia_device_pids requires py3nvml. Please install it with `pip install py3nvml`.")

    nvml.nvmlInit()

    for device_id in devices_ids:
        device_handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
        device_processes = nvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
        for device_process in device_processes:
            if device_id not in devices_pids:
                devices_pids[device_id] = []

            devices_pids[device_id].append(device_process.pid)

    nvml.nvmlShutdown()

    return devices_pids


def get_amd_devices_pids() -> Dict[int, list]:
    devices_pids: Dict[int, list] = {}
    rocm_version = torch_version().split("rocm")[-1]
    devices_ids = [int(device_id) for device_id in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

    if not is_amdsmi_available():
        raise ValueError(
            "get_amd_devices_pids requires amdsmi. "
            "Please follow the instructions at https://github.com/RadeonOpenCompute/amdsmi/tree/master"
        )

    amdsmi.amdsmi_init()

    if rocm_version >= "5.7":
        # starting from rocm 5.7, the api seems to have changed names
        devices_handles = amdsmi.amdsmi_get_processor_handles()
        for device_id in devices_ids:
            device_handle = devices_handles[device_id]
            try:
                # these functions fail a lot for no apparent reason
                processes_handles = amdsmi.amdsmi_get_gpu_process_list(device_handle)
            except Exception:
                continue

            for process_handle in processes_handles:
                try:
                    # these functions fail a lot for no apparent reason
                    info = amdsmi.amdsmi_get_gpu_process_info(device_handle, process_handle)
                except Exception:
                    continue

                if info["memory_usage"]["vram_mem"] == 4096:
                    continue

                if device_id not in devices_pids:
                    devices_pids[device_id] = []

                devices_pids[device_id].append(info["pid"])
    else:
        devices_handles = amdsmi.amdsmi_get_device_handles()
        for device_id in devices_ids:
            device_handle = devices_handles[device_id]
            try:
                # these functions might fail for no apparent reason
                processes_handles = amdsmi.amdsmi_get_process_list(device_handle)
            except Exception:
                continue

            for process_handle in processes_handles:
                try:
                    # these functions might fail for no apparent reason
                    info = amdsmi.amdsmi_get_process_info(device_handle, process_handle)
                except Exception:
                    continue

                if info["memory_usage"]["vram_mem"] == 4096:
                    continue

                if device_id not in devices_pids:
                    devices_pids[device_id] = []

                devices_pids[device_id].append(info["pid"])

    amdsmi.amdsmi_shut_down()

    return devices_pids


def get_pids_running_on_system_device() -> Set[int]:
    """Returns the set of pids running on the system device(s)."""

    if is_nvidia_system():
        devices_pids = get_nvidia_devices_pids()
    elif is_rocm_system():
        devices_pids = get_amd_devices_pids()
    else:
        raise ValueError("get_pids_running_on_system_device is only supported on NVIDIA and AMD GPUs")

    all_devices_pids = set(sum(devices_pids.values(), []))

    return all_devices_pids


def assert_system_devices_isolation(benchmark_pid: int) -> None:
    setup_logging("ERROR")

    isolation_pid = os.getpid()

    while psutil.pid_exists(benchmark_pid):
        child_processes = set()
        non_permitted_pids = set()

        all_devices_pids = get_pids_running_on_system_device()

        for pid in list(all_devices_pids):
            if pid == benchmark_pid or pid == isolation_pid:
                continue

            try:
                info = psutil.Process(pid)
                parent_pid = info.ppid()
            except Exception as e:
                LOGGER.error(f"Failed to get info for process {pid} with error {e}")
                parent_pid = None

            if parent_pid == benchmark_pid or parent_pid == isolation_pid:
                child_processes.add(pid)
            else:
                non_permitted_pids.add(pid)

        if len(non_permitted_pids) > 0:
            LOGGER.error(f"Found non-permitted process(es) running on system device(s): {non_permitted_pids}")
            for pid in child_processes:
                try:
                    LOGGER.error(f"Terminating child process {pid}")
                    os.kill(pid, signal.SIGTERM)
                except Exception as e:
                    LOGGER.error(f"Failed to terminate child process {pid} with error {e}")

            LOGGER.error(f"Terminating benchmark process {benchmark_pid}")
            os.kill(benchmark_pid, signal.SIGTERM)
            break

        time.sleep(1)


@contextmanager
def device_isolation(benchmark_pid: int, enabled: bool) -> None:
    if not enabled:
        yield
        return

    isolation_process = Process(
        target=assert_system_devices_isolation,
        kwargs={"benchmark_pid": benchmark_pid},
        daemon=True,
    )
    isolation_process.start()
    LOGGER.info(f"\t+ Launched device(s) isolation process {isolation_process.pid}.")

    yield

    LOGGER.info("\t+ Closing device(s) isolation process...")
    isolation_process.kill()
    isolation_process.join()
    isolation_process.close()
