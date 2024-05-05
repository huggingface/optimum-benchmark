import multiprocessing as mp
import os
from logging import getLogger
from typing import Any, Callable, Union

import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ...logging_utils import setup_logging
from ...report import BenchmarkReport
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

    def launch(self, worker: Callable[..., BenchmarkReport], *worker_args: Any) -> BenchmarkReport:
        ctx = mp.get_context(self.config.start_method)
        log_level = ctx.get_logger().getEffectiveLevel()

        isolated_process = mp.Process(
            target=target,
            args=(worker, log_level, *worker_args),
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

        reports = []

        LOGGER.info("\t+ Gatehring reports from all ranks.")
        for rank in range(self.config.nproc_per_node):
            if not os.path.isfile(f"rank_{rank}_report.json"):
                raise RuntimeError(f"Could not find report from rank {rank}.")

            report = BenchmarkReport.from_json(f"rank_{rank}_report.json")
            reports.append(report)

        LOGGER.info("\t+ Aggregating reports from all ranks.")
        report = BenchmarkReport.aggregate(reports)
        report.log()

        return report


def target(
    worker: Callable[..., BenchmarkReport], log_level: Union[str, int], *worker_args: Any, launch_config: LaunchConfig
):
    isolated_process_pid = os.getpid()
    os.environ["ISOLATED_PROCESS_PID"] = str(isolated_process_pid)

    setup_logging(level=log_level, format_prefix="ISOLATED-PROCESS")
    LOGGER.info(f"Running benchmark in isolated process [{isolated_process_pid}].")

    elastic_agent_launcher = elastic_launch(config=launch_config, entrypoint=entrypoint)
    elastic_agent_launcher(worker, log_level, *worker_args)

    LOGGER.info("Exiting isolated process.")


@record
def entrypoint(worker: Callable[..., BenchmarkReport], log_level: Union[str, int], *worker_args: Any):
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0 or (os.environ.get("LOG_ALL_RANKS", "0") == "1"):
        setup_logging(level=log_level, format_prefix=f"RANK-{rank}")
    else:
        setup_logging(level="ERROR", format_prefix=f"RANK-{rank}")

    if torch.cuda.is_available():
        LOGGER.info(f"\t+ Setting torch.distributed cuda device to {rank}.")
        torch.cuda.set_device(rank)

    LOGGER.info("Initializing torch.distributed process group.")
    torch.distributed.init_process_group()

    report = worker(*worker_args)

    LOGGER.info(f"Saving report from rank {rank}.")
    report.save_json(f"rank_{rank}_report.json")

    LOGGER.info("Destroying torch.distributed process group.")
    torch.distributed.destroy_process_group()

    LOGGER.info("Exiting torchrun rank.")
