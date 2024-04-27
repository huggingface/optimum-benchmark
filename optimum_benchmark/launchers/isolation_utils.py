import multiprocessing as mp
import os
import signal
import time
from contextlib import contextmanager
from logging import getLogger
from typing import Set

from ..import_utils import is_amdsmi_available, is_psutil_available, is_pynvml_available
from ..logging_utils import setup_logging
from ..system_utils import is_nvidia_system, is_rocm_system

if is_psutil_available():
    import psutil

if is_pynvml_available():
    import pynvml

if is_amdsmi_available():
    import amdsmi  # type: ignore


LOGGER = getLogger("device-isolation")


def isolation_error_signal_handler(signum, frame):
    LOGGER.error(f"Process {os.getpid()} received an isolation signal with an `error` action. Exiting...")
    raise RuntimeError("Received an isolation signal with an `error` action")


signal.signal(signal.SIGUSR1, isolation_error_signal_handler)


def get_nvidia_devices_pids(device_ids: str) -> Set[int]:
    if not is_pynvml_available():
        raise ValueError(
            "The library pynvml is required to get the pids running on NVIDIA GPUs, but is not installed. "
            "Please install the official and NVIDIA maintained PyNVML library through `pip install nvidia-ml-py`."
        )

    pynvml.nvmlInit()

    devices_pids = set()
    devices_ids = list(map(int, device_ids.split(",")))

    for device_id in devices_ids:
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        device_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
        for device_process in device_processes:
            devices_pids.add(device_process.pid)

    pynvml.nvmlShutdown()

    return devices_pids


def get_amd_devices_pids(device_ids: str) -> Set[int]:
    if not is_amdsmi_available():
        raise ValueError(
            "The library amdsmi is required to get the pids running on AMD GPUs, but is not installed. "
            "Please install the official and AMD maintained amdsmi library from https://github.com/ROCm/amdsmi."
        )

    amdsmi.amdsmi_init()

    devices_pids = set()
    devices_ids = list(map(int, device_ids.split(",")))

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
                # not sure why these processes are always present
                continue

            devices_pids.add(info["pid"])

    amdsmi.amdsmi_shut_down()

    return devices_pids


def get_pids_running_on_system_devices(device_ids: str) -> Set[int]:
    """Returns the set of pids running on the system device(s)."""
    if is_nvidia_system():
        devices_pids = get_nvidia_devices_pids(device_ids)
    elif is_rocm_system():
        devices_pids = get_amd_devices_pids(device_ids)
    else:
        raise ValueError("get_pids_running_on_system_device is only supported on NVIDIA and AMD GPUs")

    return devices_pids


def get_process_children_pids(pid: int) -> Set[int]:
    """Returns the set of pids of the children of the given process."""
    process = psutil.Process(pid)
    children = process.children(recursive=True)
    children_pids = {child.pid for child in children}

    return children_pids


def assert_system_devices_isolation(isolated_pids: set, device_ids: str, action: str):
    setup_logging("INFO", prefix="DEVICE-ISOLATION")

    assert action in ["warn", "error", "kill"], f"Unsupported action `{action}`"

    while any(psutil.pid_exists(pid) for pid in isolated_pids):
        devices_pids = get_pids_running_on_system_devices(device_ids=device_ids)
        devices_pids = {pid for pid in devices_pids if psutil.pid_exists(pid)}

        permitted_pids = set()
        for pid in isolated_pids:
            permitted_pids.add(pid)
            permitted_pids |= get_process_children_pids(pid)
        permitted_pids = {pid for pid in permitted_pids if psutil.pid_exists(pid)}

        non_permitted_pids = devices_pids - permitted_pids

        if len(non_permitted_pids) > 0:
            LOGGER.info(f"Found non-permitted process(es) running on system device(s): {non_permitted_pids}")

            for pid in isolated_pids:
                if action == "warn":
                    LOGGER.warn(f"Process {pid} is not isolated and is running on system device(s)")
                elif action == "error":
                    os.kill(pid, signal.SIGUSR1)
                elif action == "kill":
                    os.kill(pid, signal.SIGKILL)

            LOGGER.warn("Exiting the isolation process...")
            exit(1)

        time.sleep(1)


@contextmanager
def device_isolation(enable: bool, action: str, isolated_pids: set):
    if not enable:
        yield
        return

    if is_nvidia_system():
        device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    elif is_rocm_system():
        device_ids = os.environ.get("ROCR_VISIBLE_DEVICES", None)
    else:
        raise ValueError("Device isolation is only supported on NVIDIA and AMD GPUs")

    if device_ids is None:
        raise ValueError(
            "Device isolation requires CUDA_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES to be set but none were found."
        )

    LOGGER.info(f"\t+ Isolated device(s) [{device_ids}]")
    LOGGER.info(f"\t+ Isolated process(es) [{isolated_pids}]")
    LOGGER.info(f"\t+ Sending an action signal `{action}` in case of non-isolation")

    isolation_process = mp.Process(
        target=assert_system_devices_isolation,
        kwargs={
            "isolated_pids": isolated_pids,
            "device_ids": device_ids,
            "action": action,
        },
        daemon=True,  # so that the process is killed when the main process is killed
    )
    isolation_process.start()
    LOGGER.info(f"\t+ Started device(s) isolation process {isolation_process.pid}")

    yield

    if isolation_process.is_alive():
        LOGGER.info("\t+ Closing device(s) isolation process...")
        isolation_process.kill()
        isolation_process.join()
        isolation_process.close()
