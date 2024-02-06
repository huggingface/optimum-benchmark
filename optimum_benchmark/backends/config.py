import os
from abc import ABC
from logging import getLogger
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Dict, Any

from psutil import cpu_count

from ..env_utils import get_gpus, is_nvidia_system, is_rocm_system
from ..task_utils import (
    infer_library_from_model_name_or_path,
    infer_task_from_model_name_or_path,
)

LOGGER = getLogger("backend")

HUB_KWARGS = {
    "revision": "main",
    "force_download": False,
    "local_files_only": False,
}


@dataclass
class BackendConfig(ABC):
    name: str
    version: str
    _target_: str

    seed: int = 42

    model: Optional[str] = None
    device: Optional[str] = None

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
            raise ValueError(
                f"Device was specified as {self.device} with a target index."
                "We recommend using the main cuda device (e.g. `cuda`) and "
                "specifying the target index in `CUDA_VISIBLE_DEVICES`."
            )

        if self.device not in ["cuda", "cpu", "mps", "xla"]:
            raise ValueError("`device` must be either `cuda`, `cpu`, `mps` or `xla`.")

        if self.device == "cuda" and len(get_gpus()) > 1:
            if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
                LOGGER.warning(
                    "Multiple GPUs detected but CUDA_VISIBLE_DEVICES is not set. "
                    "This means that code might allocate resources from the wrong GPUs. "
                    "For example, with `auto_device='auto'. `We recommend setting CUDA_VISIBLE_DEVICES "
                    "to isolate the GPUs that will be used for this experiment. `CUDA_VISIBLE_DEVICES` will "
                    "be set to `0` to ensure that only the first GPU is used. If you want to use multiple "
                    "GPUs, please set `CUDA_VISIBLE_DEVICES` to the desired GPU indices."
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            if os.environ.get("CUDA_DEVICE_ORDER", None) != "PCI_BUS_ID":
                LOGGER.warning(
                    "Multiple GPUs detected but CUDA_DEVICE_ORDER is not set to `PCI_BUS_ID`. "
                    "This means that code might allocate resources from the wrong GPUs even if "
                    "`CUDA_VISIBLE_DEVICES` is set. For example pytorch uses the `FASTEST_FIRST` "
                    "order by default, which is not guaranteed to be the same as nvidia-smi. `CUDA_DEVICE_ORDER` "
                    "will be set to `PCI_BUS_ID` to ensure that the GPUs are allocated in the same order as nvidia-smi. "
                )
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        elif self.device == "cuda" and len(get_gpus()) == 1:
            if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
