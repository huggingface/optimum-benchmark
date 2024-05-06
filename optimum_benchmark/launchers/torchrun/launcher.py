import multiprocessing as mp
import os
import signal
from logging import getLogger
from typing import Any, Callable, Union

import torch.distributed
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ...logging_utils import setup_logging
from ...report import BenchmarkReport
from ..base import Launcher
from ..device_isolation_utils import device_isolation_context
from .config import TorchrunConfig

LOGGER = getLogger("torchrun")


class ForcedZeroExit(SystemExit):
    code: int = 0


def forced_zero_exit_signal_handler(signum, frame):
    for p in mp.active_children():
        LOGGER.info(f"Sending a forced zero exit signal to process [{p.pid}].")
        os.kill(p.pid, signal.SIGUSR2)

    raise ForcedZeroExit


signal.signal(signal.SIGUSR2, forced_zero_exit_signal_handler)


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
            args=(log_level, worker, *worker_args),
            kwargs={"launch_config": self.launch_config},
            daemon=False,
        )
        isolated_process.start()

        with device_isolation_context(
            enable=self.config.device_isolation, action=self.config.device_isolation_action, pid=isolated_process.pid
        ):
            isolated_process.join()

        if all(os.path.isfile(f"benchmark_report_rank_{rank}.json") for rank in range(self.config.nproc_per_node)):
            LOGGER.info("\t+ Gatehring reports from all ranks.")
            reports = [
                BenchmarkReport.from_json(f"benchmark_report_rank_{rank}.json")
                for rank in range(self.config.nproc_per_node)
            ]
            LOGGER.info("\t+ Aggregating reports from all ranks.")
            report = BenchmarkReport.aggregate(reports)
            LOGGER.info("\t+ Logging aggregated report.")
            report.log()
        elif isolated_process.exitcode != 0:
            raise RuntimeError(f"Process exited with non-zero code {isolated_process.exitcode}.")
        else:
            raise RuntimeError("Could not retrieve report from isolated process.")

        return report


def target(
    log_level: Union[str, int], worker: Callable[..., BenchmarkReport], *worker_args, launch_config: LaunchConfig
):
    isolated_process_pid = os.getpid()
    os.environ["ISOLATED_PROCESS_PID"] = str(isolated_process_pid)

    setup_logging(level=log_level, format_prefix="ISOLATED-PROCESS")
    LOGGER.info(f"Running benchmark in isolated process [{isolated_process_pid}].")

    elastic_agent_launcher = elastic_launch(config=launch_config, entrypoint=entrypoint)
    try:
        elastic_agent_launcher(log_level, worker, *worker_args)
    except ForcedZeroExit:
        pass

    LOGGER.info("Exiting isolated process.")
    exit(0)


def entrypoint(log_level: Union[str, int], worker: Callable[..., BenchmarkReport], *worker_args):
    rank = int(os.environ.get("RANK", "0"))
    isolated_process_pid = int(os.environ["ISOLATED_PROCESS_PID"])

    if (rank == 0) or (os.environ.get("LOG_ALL_RANKS", "0") == "1"):
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
    report.save_json(f"benchmark_report_rank_{rank}.json")

    LOGGER.info("Waiting for all ranks to finish.")
    torch.distributed.barrier()

    LOGGER.info("Destroying torch.distributed process group.")
    torch.distributed.destroy_process_group()

    LOGGER.info("Sending a forced zero exit signal to the isolated process.")
    os.kill(isolated_process_pid, signal.SIGUSR2)
