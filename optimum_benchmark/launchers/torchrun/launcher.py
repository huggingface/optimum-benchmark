import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
import os
from logging import getLogger
from typing import Any, Callable, Dict

import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ...benchmarks.report import BenchmarkReport
from ...logging_utils import setup_logging
from ..base import Launcher
from ..device_isolation_utils import device_isolation_context
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

        isolated_process = mp.Process(
            target=target,
            args=(worker, queue, lock, log_level, *worker_args),
            kwargs={"launch_config": self.launch_config},
            daemon=False,
        )
        isolated_process.start()

        with device_isolation_context(
            enable=self.config.device_isolation, action=self.config.device_isolation_action, pid=isolated_process.pid
        ):
            isolated_process.join()

        if isolated_process.exitcode != 0:
            raise RuntimeError(f"Process exited with non-zero code {isolated_process.exitcode}.")
        elif queue.empty():
            raise RuntimeError("Queue is empty 😥")

        reports = []

        while not queue.empty():
            lock.acquire()
            reports.append(queue.get())
            lock.release()

        if len(reports) != self.config.nproc_per_node:
            raise RuntimeError(
                f"Number of gathered reports ({len(reports)}) does not match the number of processes ({self.config.nproc_per_node})."
            )

        report = BenchmarkReport.aggregate(reports)
        report.log()

        return report


def target(worker, queue: mp.Queue, lock: mp_sync.Lock, log_level, *worker_args, launch_config: LaunchConfig):
    os.environ["ISOLATED_PROCESS_PID"] = str(os.getpid())
    setup_logging(level=log_level, prefix="ISOLATED-PROCESS")
    LOGGER.info(f"Running benchmark in isolated process [{os.getpid()}].")

    elastic_agent_launcher = elastic_launch(config=launch_config, entrypoint=entrypoint)
    elastic_agent_launcher(worker, queue, lock, log_level, *worker_args)


@record
def entrypoint(worker, queue: mp.Queue, lock: mp_sync.Lock, log_level, *worker_args):
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        setup_logging(level=log_level, prefix=f"RANK-{rank}")
    elif os.environ.get("LOG_ALL_RANKS", "0") == "1":
        setup_logging(level=log_level, prefix=f"RANK-{rank}")
    else:
        setup_logging(level="ERROR", prefix=f"RANK-{rank}")

    if torch.cuda.is_available():
        LOGGER.info(f"\t+ Setting torch.distributed cuda device to {rank}.")
        torch.cuda.set_device(rank)

    LOGGER.info("Initializing torch.distributed process group.")
    torch.distributed.init_process_group()
    torch.distributed.barrier()

    output = worker(*worker_args)

    lock.acquire()
    queue.put(output)
    lock.release()

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    LOGGER.info("Destroyed torch.distributed process group.")
