import multiprocessing as mp
import os
import traceback
from logging import Logger
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import Any, Callable, List

import torch.distributed
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ...logging_utils import setup_logging
from ...report import BenchmarkReport
from ..base import Launcher
from .config import TorchrunConfig


class TorchrunLauncher(Launcher[TorchrunConfig]):
    NAME = "torchrun"

    def __init__(self, config: TorchrunConfig):
        super().__init__(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            self.logger.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}")
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

    def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
        ctx = mp.get_context(self.config.start_method)
        parent_connection, child_connection = ctx.Pipe()

        isolated_process = Process(
            target=target, args=(worker, worker_args, child_connection, self.launch_config, self.logger), daemon=False
        )
        isolated_process.start()
        self.logger.info(f"\t+ Started benchmark in isolated process [{isolated_process.pid}]")

        if self.config.device_isolation:
            self.start_device_isolation_process(pid=isolated_process.pid)

        parent_connection.send("START")
        isolated_process.join()

        if self.config.device_isolation:
            self.stop_device_isolation_process()

        if isolated_process.exitcode != 0:
            raise RuntimeError(f"Isolated process exited with non-zero code {isolated_process.exitcode}")

        if parent_connection.poll():
            response = parent_connection.recv()
        else:
            raise RuntimeError("Isolated process did not send any response")

        reports = []

        for output in response:
            if "traceback" in output:
                if "rank" in output:
                    self.logger.error(f"\t+ Received traceback from rank process [{output['rank']}]")
                    raise ChildProcessError(output["traceback"])
                else:
                    self.logger.error("\t+ Received traceback from isolated process")
                    raise ChildProcessError(output["traceback"])

            elif "report" in output:
                self.logger.info(f"\t+ Received report from rank process [{output['rank']}]")
                reports.append(BenchmarkReport.from_dict(output["report"]))

            else:
                raise RuntimeError(f"Received an unexpected response from isolated process: {output}")

        self.logger.info("\t+ Aggregating reports from all rank processes")
        report = BenchmarkReport.aggregate(reports)
        report.log()

        return report


def target(
    worker: Callable[..., BenchmarkReport],
    worker_args: List[Any],
    connection: Connection,
    config: LaunchConfig,
    logger: Logger,
):
    while True:
        if connection.poll():
            response = connection.recv()
            if response == "START":
                break

    isolated_process_pid = os.getpid()
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    os.environ["ISOLATED_PROCESS_PID"] = str(isolated_process_pid)
    setup_logging(level=log_level, to_file=log_to_file, prefix="ISOLATED-PROCESS")

    elastic_agent_launcher = elastic_launch(config=config, entrypoint=entrypoint)

    try:
        outputs = elastic_agent_launcher(worker, worker_args, logger)
    except Exception:
        logger.error("\t+ Sending traceback to main process")
        connection.send([{"traceback": traceback.format_exc()}])
    else:
        logger.info("\t+ Sending outputs to main process")
        connection.send(list(outputs.values()))
    finally:
        logger.info("\t+ Exiting isolated process")
        connection.close()
        exit(0)


def entrypoint(worker: Callable[..., BenchmarkReport], worker_args: List[Any], logger: Logger):
    rank = int(os.environ.get("RANK", "0"))
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    log_all_ranks = os.environ.get("LOG_ALL_RANKS", "0") == "1"

    if log_all_ranks or rank == 0:
        setup_logging(level=log_level, to_file=log_to_file, prefix=f"RANK-PROCESS-{rank}")
    else:
        setup_logging(level="ERROR", to_file=log_to_file, prefix=f"RANK-PROCESS-{rank}")

    if torch.cuda.is_available():
        logger.info(f"\t+ Setting torch.distributed cuda device to {rank}")
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)

    logger.info("\t+ Initializing torch.distributed process group")
    torch.distributed.init_process_group()

    try:
        report = worker(*worker_args)
    except Exception:
        logger.error("\t+ Benchmark failed with an exception")
        output = {"rank": rank, "traceback": traceback.format_exc()}
    else:
        logger.info("\t+ Benchmark completed successfully")
        output = {"rank": rank, "report": report.to_dict()}
    finally:
        logger.info("\t+ Destroying torch.distributed process group")
        torch.distributed.destroy_process_group()
        logger.info("\t+ Exiting rank process")
        return output
