import os
import signal
import time
from contextlib import contextmanager
from logging import getLogger
from multiprocessing import Process
from typing import Dict, Set

from ..import_utils import is_amdsmi_available, is_psutil_available, is_pynvml_available
from ..logging_utils import setup_logging
from ..system_utils import is_nvidia_system, is_rocm_system

if is_psutil_available():
    import psutil

if is_pynvml_available():
    import pynvml

if is_amdsmi_available():
    import amdsmi

LOGGER = getLogger("isolation")


def get_nvidia_devices_pids(device_ids) -> Dict[int, list]:
    if not is_pynvml_available():
        raise ValueError(
            "The library pynvml is required to get the pids running on NVIDIA GPUs, but is not installed. "
            "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
        )

    devices_pids: Dict[int, list] = {}
    devices_ids = [int(device_id) for device_id in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

    pynvml.nvmlInit()

    for device_id in devices_ids:
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        device_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
        for device_process in device_processes:
            if device_id not in devices_pids:
                devices_pids[device_id] = []

            devices_pids[device_id].append(device_process.pid)

    pynvml.nvmlShutdown()

    return devices_pids


def get_amd_devices_pids() -> Dict[int, list]:
    if not is_amdsmi_available():
        raise ValueError(
            "The library amdsmi is required to get the pids running on AMD GPUs, but is not installed. "
            "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
        )

    devices_pids: Dict[int, list] = {}
    devices_ids = [int(device_id) for device_id in os.environ["ROCR_VISIBLE_DEVICES"].split(",")]

    amdsmi.amdsmi_init()

    # starting from rocm 5.7, the api seems to have changed names
    processor_handles = amdsmi.amdsmi_get_processor_handles()
    for device_id in devices_ids:
        processor_handle = processor_handles[device_id]
        try:
            # these functions fail a lot for no apparent reason
            processes_handles = amdsmi.amdsmi_get_gpu_process_list(processor_handle)
        except Exception:
            continue

        for process_handle in processes_handles:
            try:
                # these functions fail a lot for no apparent reason
                info = amdsmi.amdsmi_get_gpu_process_info(processor_handle, process_handle)
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


def assert_system_devices_isolation(main_pid: int) -> None:
    setup_logging("ERROR")
    isolation_pid = os.getpid()

    while psutil.pid_exists(main_pid):
        permitted_pids = set()
        non_permitted_pids = set()

        all_devices_pids = get_pids_running_on_system_device()

        for pid in list(all_devices_pids):
            if pid == main_pid or pid == isolation_pid:
                continue

            try:
                info = psutil.Process(pid)
                parent_pid = info.ppid()
            except Exception:
                parent_pid = None

            if parent_pid == main_pid or parent_pid == isolation_pid:
                permitted_pids.add(pid)
            else:
                try:
                    info = psutil.Process(parent_pid)
                    parent_parent_pid = info.ppid()
                except Exception:
                    parent_parent_pid = None

                if parent_parent_pid == main_pid or parent_parent_pid == isolation_pid:
                    permitted_pids.add(pid)
                else:
                    non_permitted_pids.add(pid)

        if len(non_permitted_pids) > 0:
            LOGGER.error(f"Found non-permitted process(es) running on system device(s): {non_permitted_pids}")
            for pid in permitted_pids:
                try:
                    LOGGER.error(f"Interrupting child process {pid} of main process {main_pid}")
                    os.kill(pid, signal.SIGINT)
                except Exception as e:
                    LOGGER.error(f"Failed to terminate child process {pid} with error {e}")

            LOGGER.error(f"Interrupting main process {main_pid}...")
            os.kill(main_pid, signal.SIGINT)
            exit(1)

        time.sleep(1)


@contextmanager
def device_isolation(enabled: bool):
    if not enabled:
        yield
        return

    isolation_process = Process(target=assert_system_devices_isolation, kwargs={"main_pid": os.getpid()}, daemon=True)
    isolation_process.start()

    LOGGER.info(f"\t+ Launched device(s) isolation process {isolation_process.pid}.")

    yield

    LOGGER.info("\t+ Closing device(s) isolation process...")

    isolation_process.kill()
    isolation_process.join()
    isolation_process.close()
