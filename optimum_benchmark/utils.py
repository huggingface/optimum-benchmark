from typing import Optional, List
from logging import getLogger
import subprocess
import platform
import random
import signal
import time
import re
import os

import numpy as np
import psutil

LOGGER = getLogger("utils")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def bytes_to_mega_bytes(bytes: int) -> int:
    # Reference: https://en.wikipedia.org/wiki/Byte#Multiple-byte_units
    return int(bytes * 1e-6)


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


def check_no_process_is_running_on_cuda_device(device_ids: List[int]) -> None:
    """
    Raises a RuntimeError if any process is running on the given cuda device.
    """

    for device_id in device_ids:
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
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                        f"--id={device_id}",
                    ]
                )
                .decode()
                .startswith(f"{pid},")
            ]
        )

        # TODO: It would be safer to run each run of a sweep in a subprocess.
        # Although we can trust PyTorch to clear GPU memory when asked,
        # it is not a safe assumption to make for all backends.
        if len(pids_on_device_id) > 1 or (
            len(pids_on_device_id) == 1 and os.getpid() not in pids_on_device_id
        ):
            raise RuntimeError(
                f"Expected no processes on device {device_id}, "
                f"found {len(pids_on_device_id)} processes "
                f"with PIDs {pids_on_device_id}."
            )


def check_only_this_process_is_running_on_cuda_device(
    device_ids: List[int], pid
) -> None:
    """
    Raises a RuntimeError if at any point in time, there is a process running
    on the given cuda device that is not the current process.
    """

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

        for device_id in device_ids:
            # get list of PIDs running on cuda device_id
            pids_on_device_id = set(
                [
                    pid
                    for pid in pids
                    if subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits",
                            f"--id={device_id}",
                        ]
                    )
                    .decode()
                    .startswith(f"{pid},")
                ]
            )

            # check if there is a process running on
            # device_id that is not the current process
            if len(pids_on_device_id) > 1:
                os.kill(pid, signal.SIGTERM)
                raise RuntimeError(
                    f"Expected only process {pid} on device {device_id}, "
                    f"found {len(pids_on_device_id)} processes "
                    f"with PIDs {pids_on_device_id}."
                )

        # sleep for 1 second
        time.sleep(1)


DIFFUSION_TASKS = [
    "stable-diffusion",
    "stable-diffusion-xl",
]


TEXT_GENERATION_TASKS = [
    "text-generation",
    "text2text-generation",
    "image-to-text",
    "automatic-speech-recognition",
]

# let's leave this here for now, it's a good list of tasks supported by transformers
ALL_TASKS = [
    "conversational",
    "feature-extraction",
    "fill-mask",
    "text-generation",
    "text2text-generation",
    "text-classification",
    "token-classification",
    "multiple-choice",
    "object-detection",
    "question-answering",
    "image-classification",
    "image-segmentation",
    "mask-generation",
    "masked-im",
    "semantic-segmentation",
    "automatic-speech-recognition",
    "audio-classification",
    "audio-frame-classification",
    "audio-xvector",
    "image-to-text",
    "stable-diffusion",
    "stable-diffusion-xl",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
]
