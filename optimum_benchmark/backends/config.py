import os
from abc import ABC
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, Optional, TypeVar

from psutil import cpu_count

from ..system_utils import get_gpu_device_ids, is_nvidia_system, is_rocm_system
from ..task_utils import infer_library_from_model_name_or_path, infer_task_from_model_name_or_path

LOGGER = getLogger("backend")

HUB_KWARGS = {"revision": "main", "force_download": False, "local_files_only": False, "trust_remote_code": False}


@dataclass
class BackendConfig(ABC):
    name: str
    version: str
    _target_: str

    seed: int = 42

    model: Optional[str] = None
    device: Optional[str] = None
    device_ids: Optional[str] = None
    # yes we use a string here instead of a list
    # because it's easier to pass in a yaml or from cli
    # and it's consistent with GPU environment variables

    task: Optional[str] = None
    library: Optional[str] = None

    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    hub_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model is None:
            raise ValueError("`model` must be specified.")

        if self.task is None:
            self.task = infer_task_from_model_name_or_path(self.model)

        if self.device is None:
            self.device = "cuda" if is_nvidia_system() or is_rocm_system() else "cpu"
            LOGGER.warning(f"`device` is not specified, defaulting to {self.device} based on system configuration.")

        if self.device not in ["cuda", "cpu", "mps", "xla"]:
            raise ValueError(f"`device` must be either `cuda`, `cpu`, `mps` or `xla`, but got {self.device}")

        if ":" in self.device:
            # support pytorch device index notation
            self.device = self.device.split(":")[0]
            self.device_ids = self.device.split(":")[1]

        if self.device == "cuda":
            if self.device_ids is None:
                self.device_ids = get_gpu_device_ids()

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_ids

            if is_rocm_system():
                # https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html
                os.environ["GPU_DEVICE_ORDINAL"] = self.device_ids
                os.environ["HIP_VISIBLE_DEVICES"] = self.device_ids
                os.environ["ROCR_VISIBLE_DEVICES"] = self.device_ids

        if self.library is None:
            self.library = infer_library_from_model_name_or_path(self.model)

        if self.library not in ["transformers", "diffusers", "timm"]:
            raise ValueError(f"`library` must be either `transformers`, `diffusers` or `timm`, but got {self.library}")

        if self.inter_op_num_threads is not None:
            if not isinstance(self.inter_op_num_threads, int):
                raise ValueError(f"`inter_op_num_threads` must be an integer, but got {self.inter_op_num_threads}")
            if self.inter_op_num_threads == -1:
                self.inter_op_num_threads = cpu_count()

        if self.intra_op_num_threads is not None:
            if not isinstance(self.intra_op_num_threads, int):
                raise ValueError(f"`intra_op_num_threads` must be an integer, but got {self.intra_op_num_threads}")
            if self.intra_op_num_threads == -1:
                self.intra_op_num_threads = cpu_count()

        self.hub_kwargs = {**HUB_KWARGS, **self.hub_kwargs}


BackendConfigT = TypeVar("BackendConfigT", bound=BackendConfig)
