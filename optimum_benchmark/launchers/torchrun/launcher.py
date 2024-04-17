import os
from logging import getLogger
from typing import Any, Callable, Dict, List

import torch.distributed
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, launch_agent

from ...benchmarks.report import BenchmarkReport
from ...logging_utils import setup_logging
from ..base import Launcher
from ..isolation_utils import device_isolation
from .config import TorchrunConfig

LOGGER = getLogger("torchrun")


class TorchrunLauncher(Launcher[TorchrunConfig]):
    NAME = "torchrun"

    def __init__(self, config: TorchrunConfig):
        super().__init__(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            LOGGER.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}.")
            mp.set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable, *worker_args) -> Dict[str, Any]:
        log_level = getLogger().getEffectiveLevel()
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

        ctx = mp.get_context(self.config.start_method)
        queue = ctx.Queue()
        lock = ctx.Lock()

        with device_isolation(
            isolated_pid=os.getpid(),
            enabled=self.config.device_isolation,
            action=self.config.device_isolation_action,
        ):
            LOGGER.info(f"\t+ Launching torchrun agent with {self.config.nproc_per_node} worker processes")
            launch_agent(
                entrypoint=entrypoint, args=(worker, queue, lock, log_level, *worker_args), config=launch_config
            )

        reports: List[BenchmarkReport] = []

        while not queue.empty():
            reports.append(queue.get())

        if len(reports) > 1:
            LOGGER.info(f"\t+ Merging benchmark reports from {len(reports)} workers")
            report = reports[0].aggregate(reports)
        elif len(reports) == 1:
            report = reports[0]
        else:
            raise ValueError("No benchmark report was returned by the workers")

        # Log the final report
        report.log()

        return report


@record
def entrypoint(worker, queue, lock, log_level, *worker_args):
    """
    This a pickalable function that correctly sets up the logging configuration
    """

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None
    setup_logging(level=log_level, prefix=f"RANK-{rank}") if rank == 0 else setup_logging(level="ERROR")

    torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.distributed.barrier()

    output = worker(*worker_args)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

    lock.acquire()
    queue.put(output)
    lock.release()
