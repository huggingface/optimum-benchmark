import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from ...import_utils import py_txi_version
from ...system_utils import is_nvidia_system, is_rocm_system
from ...task_utils import TEXT_EMBEDDING_TASKS, TEXT_GENERATION_TASKS
from ..config import BackendConfig


@dataclass
class PyTXIConfig(BackendConfig):
    name: str = "py-txi"
    version: Optional[str] = py_txi_version()
    _target_: str = "optimum_benchmark.backends.py_txi.backend.PyTXIBackend"

    # optimum benchmark specific
    no_weights: bool = False

    # Image to use for the container
    image: Optional[str] = None
    # Shared memory size for the container
    shm_size: str = "1g"
    # List of custom devices to forward to the container e.g. ["/dev/kfd", "/dev/dri"] for ROCm
    devices: Optional[List[str]] = None
    # NVIDIA-docker GPU device options e.g. "all" (all) or "0,1,2,3" (ids) or 4 (count)
    gpus: Optional[Union[str, int]] = None
    # Things to forward to the container
    ports: Dict[str, Any] = field(
        default_factory=lambda: {"80/tcp": ("127.0.0.1", 0)},
        metadata={"help": "Dictionary of ports to expose from the container."},
    )
    volumes: Dict[str, Any] = field(
        default_factory=lambda: {HUGGINGFACE_HUB_CACHE: {"bind": "/data", "mode": "rw"}},
        metadata={"help": "Dictionary of volumes to mount inside the container."},
    )
    environment: List[str] = field(
        default_factory=lambda: ["HUGGING_FACE_HUB_TOKEN"],
        metadata={"help": "List of environment variables to forward to the container from the host."},
    )

    # Common options
    dtype: Optional[str] = None
    max_concurrent_requests: Optional[int] = None

    # TGI specific
    sharded: Optional[str] = None
    quantize: Optional[str] = None
    num_shard: Optional[int] = None
    speculate: Optional[int] = None
    cuda_graphs: Optional[bool] = None
    disable_custom_kernels: Optional[bool] = None
    trust_remote_code: Optional[bool] = None

    # TEI specific
    pooling: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        if self.task not in TEXT_GENERATION_TASKS + TEXT_EMBEDDING_TASKS:
            raise NotImplementedError(f"TXI does not support task {self.task}")

        # Device options
        if self.device_ids is not None and is_nvidia_system() and self.gpus is None:
            self.gpus = self.device_ids

        if self.device_ids is not None and is_rocm_system() and self.devices is None:
            ids = list(map(int, self.device_ids.split(",")))
            renderDs = [file for file in os.listdir("/dev/dri") if file.startswith("renderD")]
            self.devices = ["/dev/kfd"] + [f"/dev/dri/{renderDs[i]}" for i in ids]

        # Common options
        if self.max_concurrent_requests is None:
            if self.task in TEXT_GENERATION_TASKS:
                self.max_concurrent_requests = 128
            elif self.task in TEXT_EMBEDDING_TASKS:
                self.max_concurrent_requests = 512

        # TGI specific
        if self.task in TEXT_GENERATION_TASKS:
            if self.trust_remote_code is None:
                self.trust_remote_code = self.model_kwargs.get("trust_remote_code", False)
