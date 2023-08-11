import re
import os
import time
import torch
import signal
import random
import importlib
import psutil
import platform
import subprocess
import numpy as np
from typing import Optional, TYPE_CHECKING
from logging import getLogger
from omegaconf import DictConfig
    

LOGGER = getLogger("utils")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bytes_to_mega_bytes(bytes: int) -> int:
    return bytes >> 20


def get_cpu() -> Optional[str]:
    if platform.system() == "Windows":
        return platform.processor()

    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return str(subprocess.check_output(command).strip())

    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
        return "Could not find device name"

    else:
        raise ValueError(f"Unknown system '{platform.system()}'")


def get_cpu_ram_mb():
    return bytes_to_mega_bytes(psutil.virtual_memory().total)


def check_no_process_is_running_on_cuda_device(device: torch.device) -> None:
    """
    Raises a RuntimeError if any process is running on the given cuda device.
    """

    cuda_device_id = (
        device.index if device.index is not None else torch.cuda.current_device()
    )

    # get list of all PIDs running on nvidia devices
    pids = [
        int(pid)
        for pid in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
        )
        .decode()
        .strip()
        .split("\n")
        if pid != ""
    ]

    # get list of PIDs running on cuda device_id
    pids_on_device_id = set(
        [
            pid
            for pid in pids
            if subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-compute-apps=pid,used_memory",
                    f"--format=csv,noheader,nounits",
                    f"--id={cuda_device_id}",
                ]
            )
            .decode()
            .startswith(f"{pid},")
        ]
    )

    if len(pids_on_device_id) > 0:
        raise RuntimeError(
            f"Expected no processes on device {cuda_device_id}, "
            f"found {len(pids_on_device_id)} processes "
            f"with PIDs {pids_on_device_id}."
        )


def check_only_this_process_is_running_on_cuda_device(
    device: torch.device, pid
) -> None:
    """
    Raises a RuntimeError if at any point in time, there is a process running
    on the given cuda device that is not the current process.
    """

    cuda_device_id = (
        device.index if device.index is not None else torch.cuda.current_device()
    )

    while True:
        # get list of all PIDs running on nvidia devices
        pids = [
            int(pid)
            for pid in subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
            )
            .decode()
            .strip()
            .split("\n")
            if pid != ""
        ]

        # get list of PIDs running on cuda device_id
        pids_on_device_id = set(
            [
                pid
                for pid in pids
                if subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--query-compute-apps=pid,used_memory",
                        f"--format=csv,noheader,nounits",
                        f"--id={cuda_device_id}",
                    ]
                )
                .decode()
                .startswith(f"{pid},")
            ]
        )

        # check if there is a process running on device_id that is not the current process
        if len(pids_on_device_id) > 1:
            os.kill(pid, signal.SIGTERM)
            raise RuntimeError(
                f"Expected only process {pid} on device {cuda_device_id}, "
                f"found {len(pids_on_device_id)} processes "
                f"with PIDs {pids_on_device_id}."
            )

        # sleep for 1 second
        time.sleep(1)


def infer_device_id(device: str) -> int:
    """
    Infer the device id from the given device string.
    """

    if device == "cuda":
        return torch.cuda.current_device()
    elif torch.device(device).type == "cuda":
        return torch.device(device).index
    elif torch.device(device).type == "cpu":
        return -1
    else:
        raise ValueError(f"Unknown device '{device}'")


_NAME_TO_IMPORTPATH = {
    "pytorch": "optimum_benchmark.backends.pytorch",
    "openvino": "optimum_benchmark.backends.openvino",
    "neural_compressor": "optimum_benchmark.backends.neural_compressor",
    "onnxruntime": "optimum_benchmark.backends.onnxruntime",
    "inference": "optimum_benchmark.benchmarks.inference",
    "training": "optimum_benchmark.benchmarks.training",
}

_NAME_TO_CLASS_NAME = {
    "pytorch": "PyTorchConfig",
    "openvino": "OVConfig",
    "neural_compressor": "INCConfig",
    "onnxruntime": "ORTConfig",
    "inference": "InferenceConfig",
    "training": "TrainingConfig",
}

def name_to_dataclass(name: str):
    # We use a map name to import path to avoid importing everything here, especially every backend, to avoid to install all backends to run
    # optimum-benchmark.
    module = importlib.import_module(_NAME_TO_IMPORTPATH[name])
    dataclass_class = getattr(module, _NAME_TO_CLASS_NAME[name])
    return dataclass_class

def remap_to_correct_metadata(experiment: DictConfig):
    for key, value in experiment.items():
        if isinstance(value, DictConfig) and hasattr(value, "name"):
            experiment[key]._metadata.object_type = name_to_dataclass(experiment[key].name)
    return experiment
