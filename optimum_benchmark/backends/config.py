import os
from abc import ABC
from logging import getLogger
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Dict, Any

from ..import_utils import is_psutil_available
from ..env_utils import get_cuda_device_ids, is_nvidia_system, is_rocm_system
from ..task_utils import infer_library_from_model_name_or_path, infer_task_from_model_name_or_path

if is_psutil_available():
    from psutil import cpu_count

LOGGER = getLogger("backend")

HUB_KWARGS = {
    "revision": "main",
    "force_download": False,
    "local_files_only": False,
    "trust_remote_code": False,
}


@dataclass
class BackendConfig(ABC):
    name: str
    version: str
    _target_: str

    seed: int = 42

    model: Optional[str] = None
    device: Optional[str] = None
    # yes we use a string here instead of a list
    # it's easier to pass in a yaml or from cli
    # also it's consistent with CUDA_VISIBLE_DEVICES
    device_ids: Optional[str] = None

    task: Optional[str] = None
    library: Optional[str] = None

    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None

    hub_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model is None:
            raise ValueError("`model` must be specified.")

        if self.device is None:
            self.device = "cuda" if is_nvidia_system() or is_rocm_system() else "cpu"

        if ":" in self.device:
            # using device index
            self.device = self.device.split(":")[0]
            self.device_ids = self.device.split(":")[1]

        if self.device == "cuda":
            if self.device_ids is None:
                self.device_ids = get_cuda_device_ids()

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device_ids
            # TODO: add rocm specific environment variables ?

        if self.device not in ["cuda", "cpu", "mps", "xla"]:
            raise ValueError(f"`device` must be either `cuda`, `cpu`, `mps` or `xla`, but got {self.device}")

        if self.task is None:
            self.task = infer_task_from_model_name_or_path(self.model)

        if self.library is None:
            self.library = infer_library_from_model_name_or_path(self.model)

        if self.inter_op_num_threads is not None:
            if self.inter_op_num_threads == -1:
                self.inter_op_num_threads = cpu_count()

        if self.intra_op_num_threads is not None:
            if self.intra_op_num_threads == -1:
                self.intra_op_num_threads = cpu_count()

        self.hub_kwargs = {**HUB_KWARGS, **self.hub_kwargs}


BackendConfigT = TypeVar("BackendConfigT", bound=BackendConfig)
