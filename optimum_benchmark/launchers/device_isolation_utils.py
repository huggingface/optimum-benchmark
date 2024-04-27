import multiprocessing as mp
import os
import signal
import time
from contextlib import contextmanager
from logging import getLogger
from typing import Optional, Set

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


class DeviceIsolationError(Exception):
    pass


def isolation_error_signal_handler(signum, frame):
    raise DeviceIsolationError("Received an error signal from the device isolation process")


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


def assert_device_isolation(action: str, pid: int, device_ids: str):
    setup_logging("INFO", prefix="DEVICE-ISOLATION-PROCESS")

    assert action in ["warn", "error", "kill"], f"Unsupported action `{action}`"

    while psutil.pid_exists(pid):
        device_pids = get_pids_running_on_system_devices(device_ids=device_ids)
        device_pids = {p for p in device_pids if psutil.pid_exists(pid)}

        permitted_pids = {pid} | get_process_children_pids(pid)
        permitted_pids = {p for p in permitted_pids if psutil.pid_exists(pid)}

        foreign_pids = device_pids - permitted_pids

        if len(foreign_pids) > 0:
            LOGGER.warn(
                f"Found foreign process(es) [{foreign_pids}] running on the isolated device(s) [{device_ids}], "
                f"other than the isolated process [{pid}] (and its children)."
            )

            if action == "warn":
                LOGGER.warn("Make sure no other process is running on the isolated device(s) while benchmarking.")
            elif action == "error":
                LOGGER.error("Signaling the isolated process to error out...")
                os.kill(pid, signal.SIGUSR1)
            elif action == "kill":
                LOGGER.error("Killing the isolated process...")
                os.kill(pid, signal.SIGKILL)

            LOGGER.warn("Exiting the isolation process...")
            exit(0)

        time.sleep(1)


@contextmanager
def device_isolation_context(enable: bool, action: Optional[str], pid: Optional[int], device_ids: Optional[str]):
    if not enable:
        yield
        return

    if action is None:
        raise ValueError("Device isolation requires the action to be specified")
    elif action not in ["warn", "error", "kill"]:
        raise ValueError(f"Unsupported action `{action}`")

    if pid is None:
        raise ValueError("Device isolation requires the pid of the isolated process")

    if device_ids is None:
        if is_nvidia_system():
            device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        elif is_rocm_system():
            device_ids = os.environ.get("ROCR_VISIBLE_DEVICES", None)

        if device_ids is None:
            raise ValueError(
                "Device isolation requires the device_ids of the isolated device(s) to be specified. "
                "Or for the environment variable `CUDA_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES` to be set."
            )

    if not (is_nvidia_system() or is_rocm_system()):
        raise ValueError("Device isolation is only supported on NVIDIA and AMD GPUs")

    device_isolation_process = mp.Process(
        target=assert_device_isolation, kwargs={"action": action, "pid": pid, "device_ids": device_ids}, daemon=True
    )
    device_isolation_process.start()

    LOGGER.info(
        f"\t+ Started device(s) isolation process [{device_isolation_process.pid}], monitoring "
        f"the isolated process [{pid}], running on device(s) [{device_ids}], with action [{action}]."
    )

    yield

    device_isolation_process.terminate()
    device_isolation_process.join(timeout=1)

    if device_isolation_process.is_alive():
        LOGGER.warn("The isolation process did not terminate gracefully. Killing it forcefully...")
        device_isolation_process.kill()
        device_isolation_process.join(timeout=1)

    device_isolation_process.close()
