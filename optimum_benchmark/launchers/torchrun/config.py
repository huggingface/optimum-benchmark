import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..config import LauncherConfig


@dataclass
class TorchrunConfig(LauncherConfig):
    name: str = "torchrun"
    _target_: str = "optimum_benchmark.launchers.torchrun.launcher.TorchrunLauncher"

    # Minimum amount of nodes that the user function will be launched on.
    # Elastic agent ensures that the user function start only when the min_nodes amount enters the rendezvous.
    min_nodes: int = 1
    # Maximum amount of nodes that the user function will be launched on.
    max_nodes: int = 1
    # On each node the elastic agent will launch this amount of workers that will execute user defined function.
    nproc_per_node: int = 2
    # User defined role of the worker (defaults to "trainer").
    role: str = "benchmarker"
    # The interval in seconds that is used by the elastic_agent as a period of monitoring workers.
    monitor_interval: int = 30
    # The name of the rdzv store.
    rdzv_id: str = str(uuid.uuid4())
    # rdzv_backend to use in the rendezvous (etcd).
    rdzv_backend: str = "c10d"
    # The endpoint of the rdzv sync. storage.
    rdzv_endpoint: str = "localhost:0"
    # Key, value pair that specifies rendezvous specific configuration.
    rdzv_configs: Dict[str, Any] = field(default_factory=lambda: {"rank": 0, "timeout": -1})
    # The timeout in seconds that is used by the elastic agent to wait for the workers to enter the rendezvous.
    rdzv_timeout: int = -1
    # The maximum amount of restarts that elastic agent will conduct on workers before failure.
    max_restarts: int = 0
    # The method is used by the elastic agent to start the workers (spawn, fork, forkserver).
    start_method: str = "spawn"
    # address of the local node if any. If not set, a lookup on the local machine's FQDN will be performed.
    local_addr: Optional[str] = None

    # The socket ifname
    socket_ifname: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        if self.start_method not in ["spawn", "fork"]:
            raise ValueError(f"start_method must be one of ['spawn', 'fork'], got {self.start_method}")

        if self.min_nodes != self.max_nodes:
            raise ValueError(
                f"min_nodes and max_nodes must be equal for a reproducible benchmark, got {self.min_nodes} and {self.max_nodes}"
            )
