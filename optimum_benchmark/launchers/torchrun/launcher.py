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

        process = mp.Process(
            target=target,
            args=(worker, queue, lock, log_level, *worker_args),
            kwargs={"start_method": self.config.start_method, "launch_config": self.launch_config},
            daemon=False,
        )
        process.start()

        with device_isolation(
            enable=self.config.device_isolation,
            action=self.config.device_isolation_action,
            isolated_pids={process.pid},
        ):
            process.join()

            if queue.empty() and process.exitcode != 0:
                LOGGER.error(f"\t+ Process exited with code {process.exitcode}.")
                raise ValueError("No benchmark report was returned by the workers")

            report: BenchmarkReport = queue.get()

        return report


def target(worker, queue, lock, log_level, *worker_args, start_method: str, launch_config: LaunchConfig):
    """
    This a pickalable function that correctly sets up the logging configuration for the worker process,
    and puts the output of the worker function into a lock-protected queue.
    """

    setup_logging(log_level, prefix="PROCESS")
    LOGGER.info(f"\t+ Running benchmark in isolated process with PID {mp.current_process().pid}.")

    sub_ctx = mp.get_context(start_method)
    sub_log_level = sub_ctx.get_logger().getEffectiveLevel()
    sub_queue = sub_ctx.Queue()
    sub_lock = sub_ctx.Lock()

    elastic_agent_launcher = elastic_launch(config=launch_config, entrypoint=entrypoint)
    _ = elastic_agent_launcher(worker, sub_queue, sub_lock, sub_log_level, *worker_args)

    # gather reports from all workers
    reports: List[BenchmarkReport] = []
    while not sub_queue.empty():
        reports.append(sub_queue.get())

    if len(reports) > 0:
        LOGGER.info(f"\t+ Merging benchmark reports from {len(reports)} workers")
        report = BenchmarkReport.aggregate(reports)
    else:
        raise ValueError("No benchmark report was returned by the workers")

    # Log the final report
    report.log()

    lock.acquire()
    queue.put(report)
    lock.release()


@record
def entrypoint(worker, queue, lock, log_level, *worker_args):
    """
    This a pickalable function that correctly sets up the logging configuration for the worker process,
    and puts the output of the worker function into a lock-protected queue.
    """

    rank = int(os.environ["RANK"])
    (setup_logging(level=log_level, prefix=f"RANK-{rank}") if rank == 0 else setup_logging(level="ERROR"))
    LOGGER.info(f"\t+ Running benchmark in isolated process with rank {rank} and PID {mp.current_process().pid}.")

    torch.cuda.set_device(rank) if torch.cuda.is_available() else None

    torch.distributed.init_process_group()
    torch.distributed.barrier()

    output = worker(*worker_args)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

    lock.acquire()
    queue.put(output)
    lock.release()
