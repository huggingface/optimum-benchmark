import os
from logging import getLogger
from typing import Callable, Dict, Any, List

from ..base import Launcher
from .config import TorchrunConfig
from ...benchmarks.report import BenchmarkReport
from ..isolation_utils import device_isolation
from ...logging_utils import setup_logging

import torch.distributed
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, launch_agent


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

        with device_isolation(enabled=self.config.device_isolation):
            LOGGER.info(f"\t+ Launching torchrun agent with {self.config.nproc_per_node} workers processes")
            launch_agent(
                entrypoint=entrypoint, args=(worker, queue, lock, log_level, *worker_args), config=launch_config
            )

        outputs: List[BenchmarkReport] = []
        while not queue.empty():
            outputs.append(queue.get())

        if len(outputs) > 1:
            LOGGER.info(f"\t+ Merging benchmark reports from {len(outputs)} workers")
            report = outputs[0].aggregate(outputs)
        elif len(outputs) == 1:
            report = outputs[0]
        else:
            raise ValueError("No benchmark report was returned by the workers")

        setup_logging(level=log_level)
        report.log()

        return report


@record
def entrypoint(worker, queue, lock, log_level, *worker_args):
    """
    This a pickalable function that correctly sets up the logging configuration
    """

    rank = int(os.environ.get("RANK", "0"))
    setup_logging(level=log_level, prefix=f"RANK-{rank}") if rank == 0 else None

    torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(rank % torch.cuda.device_count()) if torch.cuda.is_available() else None

    output = worker(*worker_args)

    lock.acquire()
    queue.put(output)
    lock.release()
