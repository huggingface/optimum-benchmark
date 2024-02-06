import os
import multiprocessing as mp
from logging import getLogger
from typing import Callable, Dict, Any, List, Union

import torch.distributed
from torch.distributed import FileStore
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, launch_agent

from ..base import Launcher
from .config import TorchrunConfig
from ...logging_utils import setup_logging
from ..isolation_utils import device_isolation
from ...benchmarks.utils import consolidate_reports


LOGGER = getLogger("torchrun")


class TorchrunLauncher(Launcher[TorchrunConfig]):
    NAME = "torchrun"

    def __init__(self, config: TorchrunConfig):
        super().__init__(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            LOGGER.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}.")
            mp.set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable, *worker_args) -> Dict[str, Any]:
        launch_config = LaunchConfig(
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            nproc_per_node=self.config.nproc_per_node,
            role=self.config.role,
            monitor_interval=self.config.monitor_interval,
            run_id=self.config.rdzv_id,
            rdzv_endpoint=self.config.rdzv_endpoint,
            rdzv_backend=self.config.rdzv_backend,
            rdzv_configs=self.config.rdzv_configs,
            max_restarts=self.config.max_restarts,
            start_method=self.config.start_method,
            metrics_cfg=self.config.metrics_cfg,
            redirects=Std.from_str(self.config.redirects),
            tee=Std.from_str(self.config.tee),
            local_addr=self.config.local_addr,
            log_dir=self.config.log_dir,
        )
        current_log_level = getLogger().getEffectiveLevel()

        with device_isolation(enabled=self.config.device_isolation, benchmark_pid=os.getpid()):
            LOGGER.info(f"\t+ Launching torchrun agent with {self.config.nproc_per_node} workers processes")
            report: Union[Dict[str, Any], List[Dict[str, Any]]] = launch_agent(
                config=launch_config,
                entrypoint=entrypoint,
                args=(worker, current_log_level, *worker_args),
            )

        if isinstance(report, list):
            report = consolidate_reports(report)

        return report


@record
def entrypoint(fn, log_level, *args):
    """
    This a pickalable function that correctly sets up the logging configuration
    """
    if not torch.distributed.is_initialized():
        # initialize the process group if not already initialized
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

    rank = torch.distributed.get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if rank == 0:
        setup_logging(log_level)

    # TODO: use a tcp store instead
    store = FileStore("torchrun_filestore")
    store.set(f"rank_{rank}", str(os.getpid()))

    return fn(*args)
