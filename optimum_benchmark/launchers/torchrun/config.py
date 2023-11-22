import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict

from omegaconf import OmegaConf

from ..base import LauncherConfig

LOGGER = getLogger("torchrun")

OmegaConf.register_new_resolver("available_gpus", lambda: len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))


@dataclass
class TorchrunConfig(LauncherConfig):
    name: str = "torchrun"
    _target_: str = "optimum_benchmark.launchers.torchrun.launcher.TorchrunLauncher"

    min_nodes: int = 1
    max_nodes: int = 1
    nproc_per_node: int = "${available_gpus:}"
    run_id: str = "${experiment_name}"
    role: str = "benchmark_worker"
    monitor_interval: int = 30
    rdzv_endpoint: str = "localhost:29500"
    rdzv_backend: str = "static"
    rdzv_timeout: int = 900
    rdzv_configs: Dict[str, Any] = field(default_factory=lambda: {"rank": 0, "timeout": 900})
    max_restarts: int = 0
    start_method: str = "spawn"
    metrics_cfg: str = ""
    redirects: str = "0"
    tee: str = "0"
    local_addr: str = ""
    log_dir: str = ""

    def __post_init__(self) -> None:
        if self.start_method not in ["spawn", "fork"]:
            raise ValueError(f"start_method must be one of ['spawn', 'fork'], got {self.start_method}")

        if self.min_nodes != self.max_nodes:
            raise ValueError(
                f"min_nodes and max_nodes must be equal for a reproducible benchmark, got {self.min_nodes} and {self.max_nodes}"
            )

        if self.min_nodes != 1:
            LOGGER.info("For multi-node benchmarks, run the benchmark on each node separately.")
            LOGGER.info(f"Waiting for the other nodes to be avaialable at {self.rdzv_endpoint}...")
