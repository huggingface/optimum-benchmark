import re
import os
import torch
import random
import psutil
import platform
import subprocess
import numpy as np
from typing import Optional
import py3nvml.py3nvml as nvml

from logging import getLogger

LOGGER = getLogger("utils")

LLM_MODEL_TYPES = [
    "mpt",
    "codegen",
    "RefinedWeb",
    "gpt_bigcode",
    "opt",
    "gptj",
    "gpt_neox",
    "bloom",
    "xglm",
    "gpt2",
    "gpt_neo",
    "llama",
    "RefinedWebModel",
]


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
        all_info = subprocess.check_output(
            command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
        return "Could not find device name"

    else:
        raise ValueError(f"Unknown system '{platform.system()}'")


def get_cpu_ram_mb():
    return bytes_to_mega_bytes(psutil.virtual_memory().total)
