import re
import os
import torch
import random
import psutil
import platform
import subprocess
import numpy as np
from typing import Optional


from logging import getLogger

LOGGER = getLogger("utils")

LLM_MODEL_TYPES = [
    "mpt",
    "opt",
    "gptj",
    "xglm",
    "gpt2",
    "bloom",
    "llama",
    "gpt_neo",
    "codegen",
    "gpt_neox",
    "RefinedWeb",
    "gpt_bigcode",
    "RefinedWebModel",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


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
