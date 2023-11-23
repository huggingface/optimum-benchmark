import logging.config
import os
import signal
import time
from logging import getLogger
from typing import Dict, List

from omegaconf import OmegaConf

from ..env_utils import is_nvidia_system, is_rocm_system
from ..import_utils import is_amdsmi_available, is_py3nvml_available, torch_version

LOGGER = getLogger("isolation")


def check_cuda_isolation(isolated_devices: List[int], isolated_pid: int) -> None:
    """
    Raises a RuntimeError if any process other than the benchmark process is running on the specified CUDA devices.
    """
    pids: Dict[int, set] = {}
    for device_id in isolated_devices:
        pids[device_id] = set()

    if is_nvidia_system():
        if not is_py3nvml_available():
            raise ValueError(
                "check_no_process_is_running_on_cuda_device requires py3nvml. "
                "Please install it with `pip install py3nvml`."
            )
        import py3nvml.py3nvml as nvml

        nvml.nvmlInit()
        for device_id in isolated_devices:
            device_handle = nvml.nvmlDeviceGetHandleByIndex(device_id)
            device_processes = nvml.nvmlDeviceGetComputeRunningProcesses(device_handle)
            for device_process in device_processes:
                if device_process.pid == os.getpid():
                    continue
                if device_process.pid != isolated_pid:
                    LOGGER.warning(f"Found unexpected process {device_process.pid} on device {device_id}.")
                    LOGGER.warning(f"Process info: {device_process}")

                pids[device_id].add(device_process.pid)

        nvml.nvmlShutdown()
    elif is_rocm_system():
        rocm_version = torch_version().split("rocm")[-1]

        if not is_amdsmi_available():
            raise ValueError(
                "check_no_process_is_running_on_cuda_device requires amdsmi. "
                "Please follow the instructions at https://github.com/RadeonOpenCompute/amdsmi/tree/master"
            )
        import amdsmi

        amdsmi.amdsmi_init()

        if rocm_version >= "5.7":
            # starting from rocm 5.7, the api seems to have changed names
            devices_handles = amdsmi.amdsmi_get_processor_handles()
            for device_id in isolated_devices:
                device_handle = devices_handles[device_id]
                processes_handles = amdsmi.amdsmi_get_gpu_process_list(device_handle)
                for process_handle in processes_handles:
                    info = amdsmi.amdsmi_get_gpu_process_info(device_handle, process_handle)
                    if info["memory_usage"]["vram_mem"] == 4096:
                        continue
                    if info["pid"] == os.getpid():
                        continue
                    if info["pid"] != isolated_pid:
                        LOGGER.warning(f"Found unexpected process {info['pid']} on device {device_id}.")
                        LOGGER.warning(f"Process info: {info}")

                    pids[device_id].add(info["pid"])
        else:
            devices_handles = amdsmi.amdsmi_get_device_handles()
            for device_id in isolated_devices:
                device_handle = devices_handles[device_id]
                processes_handles = amdsmi.amdsmi_get_process_list(device_handle)
                for process_handle in processes_handles:
                    info = amdsmi.amdsmi_get_process_info(device_handle, process_handle)
                    if info["memory_usage"]["vram_mem"] == 4096:
                        continue
                    if info["pid"] == os.getpid():
                        continue
                    if info["pid"] != isolated_pid:
                        LOGGER.warning(f"Found unexpected process {info['pid']} on device {device_id}.")
                        LOGGER.warning(f"Process info: {info}")

                    pids[device_id].add(info["pid"])

        amdsmi.amdsmi_shut_down()
    else:
        raise ValueError("check_no_process_is_running_on_cuda_device is only supported on NVIDIA and AMD GPUs.")

    all_pids = set()
    for device_id in isolated_devices:
        all_pids |= pids[device_id]
    other_pids = all_pids - {isolated_pid}

    if len(other_pids) > 0:
        error_message = (
            f"Expected only process {isolated_pid} on device(s) {isolated_devices}, but found {other_pids}."
        )
        raise RuntimeError(error_message)


def check_cuda_continuous_isolation(isolated_pid: int) -> None:
    """
    Kills the benchmark process if any other process is running on the specified CUDA devices.
    """

    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")) == 1:
        isolated_devices = [int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))]
    elif os.environ.get("LOCAL_RANK", None) is not None:
        local_rank = int(os.environ["LOCAL_RANK"])
        available_devices = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
        isolated_devices = [available_devices[local_rank]]
    else:
        isolated_devices = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))

    LOGGER.info(f"Continuously checking only process {isolated_pid} is running on device(s) {isolated_devices}")
    print(f"Continuously checking only process {isolated_pid} is running on device(s) {isolated_devices}")
    while True:
        try:
            check_cuda_isolation(isolated_devices, isolated_pid)
            time.sleep(0.1)
        except Exception as e:
            LOGGER.error(f"Error while checking CUDA isolation: {e}")
            os.kill(isolated_pid, signal.SIGTERM)  # graceful kill, will trigger the backend cleanup
            exit(1)
