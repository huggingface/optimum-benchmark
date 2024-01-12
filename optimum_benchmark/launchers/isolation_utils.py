# import logging.config
import logging.config
import os
import signal
import time
from contextlib import contextmanager
from logging import getLogger
from multiprocessing import Process
from typing import Dict, Optional, Set

from omegaconf import OmegaConf

# from omegaconf import OmegaConf
from ..env_utils import is_nvidia_system, is_rocm_system
from ..import_utils import (
    is_amdsmi_available,
    is_py3nvml_available,
    is_torch_distributed_available,
    torch_version,
)

if is_torch_distributed_available():
    from torch.distributed import FileStore

if is_py3nvml_available():
    import py3nvml.py3nvml as nvml

if is_amdsmi_available():
    import amdsmi  # type: ignore

LOGGER = getLogger("isolation")


def get_nvidia_devices_pids() -> Dict[int, set]:
    devices_pids: Dict[int, set] = {}
    devices_ids = [int(device_id) for device_id in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

    if not is_py3nvml_available():
        raise ValueError(
            "check_no_process_is_running_on_cuda_device requires py3nvml. "
            "Please install it with `pip install py3nvml`."
        )

    nvml.nvmlInit()
    for device_id in devices_ids:
        device_handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
        device_processes = nvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
        for device_process in device_processes:
            if device_id not in devices_pids:
                devices_pids[device_id] = []
            else:
                devices_pids[device_id].append(device_process.pid)

    nvml.nvmlShutdown()

    return devices_pids


def get_amd_devices_pids() -> None:
    devices_pids: Dict[int, list] = {}
    rocm_version = torch_version().split("rocm")[-1]
    devices_ids = [int(device_id) for device_id in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

    if not is_amdsmi_available():
        raise ValueError(
            "check_no_process_is_running_on_cuda_device requires amdsmi. "
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
                else:
                    devices_pids[device_id].append(info["pid"])
    else:
        devices_handles = amdsmi.amdsmi_get_device_handles()
        for device_id in devices_ids:
            device_handle = devices_handles[device_id]
            try:
                # these functions fail a lot for no apparent reason
                processes_handles = amdsmi.amdsmi_get_process_list(device_handle)
            except Exception:
                continue

            for process_handle in processes_handles:
                try:
                    # these functions fail a lot for no apparent reason
                    info = amdsmi.amdsmi_get_process_info(device_handle, process_handle)
                except Exception:
                    continue

                if info["memory_usage"]["vram_mem"] == 4096:
                    continue

                if device_id not in devices_pids:
                    devices_pids[device_id] = []
                else:
                    devices_pids[device_id].append(info["pid"])

    amdsmi.amdsmi_shut_down()

    return devices_pids


def get_pids_running_on_system_devices() -> Set[int]:
    """
    Returns the set of pids running on the system devices
    """

    if is_nvidia_system():
        devices_pids = get_nvidia_devices_pids()
    elif is_rocm_system():
        devices_pids = get_amd_devices_pids()
    else:
        raise ValueError("get_pids_running_on_system_devices is only supported on NVIDIA and AMD GPUs")

    all_devices_pids = set(sum(devices_pids.values(), []))

    return all_devices_pids


def assert_system_devices_isolation(permitted_pids: Set[int], world_size: Optional[int] = None) -> None:
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    if os.getpid() not in permitted_pids:
        permitted_pids.add(os.getpid())

    if world_size is not None:
        # add all pids in tcp store to the permitted pids
        STORE = FileStore("torchrun_filestore")
        perimitted_workers_names = [f"rank_{rank}" for rank in range(world_size)]
        STORE.wait(perimitted_workers_names)
        perimitted_workers_pids = {int(STORE.get(name)) for name in perimitted_workers_names}
        permitted_pids.update(perimitted_workers_pids)

    while True:
        all_devices_pids = get_pids_running_on_system_devices()
        non_permitted_pids = all_devices_pids - permitted_pids

        if len(non_permitted_pids) > 0:
            LOGGER.error(f"Found non-permitted process(es) running on system device(s): {non_permitted_pids}")
            for pid in permitted_pids:
                if pid == os.getpid():
                    continue
                try:
                    LOGGER.error(f"Killing isolated process {pid}")
                    os.kill(pid, signal.SIGTERM)
                except Exception as e:
                    LOGGER.error(f"Failed to kill isolated process {pid} with error {e}")

            LOGGER.error("Exiting isolation process")
            exit()
        else:
            time.sleep(1)


@contextmanager
def devices_isolation(enabled: bool, permitted_pids: Set[int], world_size: Optional[int] = None) -> None:
    if not enabled:
        yield
    else:
        isolation_process = Process(
            target=assert_system_devices_isolation,
            kwargs={"permitted_pids": permitted_pids, "world_size": world_size},
            daemon=True,
        )
        isolation_process.start()
        LOGGER.info(f"\t+ Launched device(s) isolation process {isolation_process.pid}.")

        yield

        LOGGER.info("\t+ Closing device(s) isolation process...")
        isolation_process.kill()
        isolation_process.join()
        isolation_process.close()
