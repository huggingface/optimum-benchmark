import os
from logging import getLogger
from typing import Any, Callable, Dict, List

import torch.distributed
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

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

        self.launch_config = LaunchConfig(
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            nproc_per_node=self.config.nproc_per_node,
            run_id=self.config.rdzv_id,
            role=self.config.role,
            rdzv_endpoint=self.config.rdzv_endpoint,
            rdzv_backend=self.config.rdzv_backend,
            rdzv_configs=self.config.rdzv_configs,
            rdzv_timeout=self.config.rdzv_timeout,
            max_restarts=self.config.max_restarts,
            monitor_interval=self.config.monitor_interval,
            start_method=self.config.start_method,
            local_addr=self.config.local_addr,
        )

    def launch(self, worker: Callable, *worker_args) -> Dict[str, Any]:
        ctx = mp.get_context(self.config.start_method)
        log_level = ctx.get_logger().getEffectiveLevel()
        queue = ctx.Queue()
        lock = ctx.Lock()

        with device_isolation(
            enable=self.config.device_isolation,
            action=self.config.device_isolation_action,
            isolated_pids={mp.current_process().pid},
        ):
            elastic_agent_launcher = elastic_launch(config=self.launch_config, entrypoint=entrypoint)
            _ = elastic_agent_launcher(worker, queue, lock, log_level, *worker_args)

        reports: List[BenchmarkReport] = []

        # gather reports from all workers
        while not queue.empty():
            reports.append(queue.get())

        if len(reports) > 0:
            LOGGER.info(f"\t+ Merging benchmark reports from {len(reports)} workers")
            report = BenchmarkReport.aggregate(reports)
        else:
            raise ValueError("No benchmark report was returned by the workers")

        # Log the final report
        report.log()

        return report


@record
def entrypoint(worker, queue, lock, log_level, *worker_args):
    """
    This a pickalable function that correctly sets up the logging configuration for the worker process,
    and puts the output of the worker function into a lock-protected queue.
    """

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None
    (setup_logging(level=log_level, prefix=f"RANK-{rank}") if rank == 0 else setup_logging(level="ERROR"))
    LOGGER.info(f"\t+ Running benchmark in isolated process with rank {rank} and PID {mp.current_process().pid}.")

    torch.distributed.init_process_group()
    torch.distributed.barrier()

    output = worker(*worker_args)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

    lock.acquire()
    queue.put(output)
    lock.release()
