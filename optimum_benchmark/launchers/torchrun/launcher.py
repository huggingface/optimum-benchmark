import multiprocessing as mp
import os
import signal
import time
import traceback
from logging import Logger
from multiprocessing import Process, Queue
from multiprocessing.connection import Connection
from typing import Any, Callable, List

import torch.distributed
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ...logging_utils import setup_logging
from ...report import BenchmarkReport
from ..base import Launcher
from .config import TorchrunConfig


class ForcedZeroExit(SystemExit):
    code: int = 0


def forced_zero_exit_signal_handler(signum, frame):
    for p in mp.active_children():
        mp.get_context().get_logger().info(f"Sending a forced zero exit signal to child process [{p.pid}]")
        os.kill(p.pid, signal.SIGUSR2)

    raise ForcedZeroExit


signal.signal(signal.SIGUSR2, forced_zero_exit_signal_handler)


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
        queue = ctx.Queue()

        isolated_process = Process(
            target=target,
            args=(worker, worker_args, child_connection, self.launch_config, self.logger, queue),
            daemon=False,
        )
        isolated_process.start()
        self.logger.info(f"\t+ Started benchmark in isolated process [{isolated_process.pid}]")

        if self.config.device_isolation:
            self.start_device_isolation_process(pid=isolated_process.pid)

        parent_connection.send("start")
        isolated_process.join()

        if self.config.device_isolation:
            self.stop_device_isolation_process()

        if isolated_process.exitcode != 0:
            raise RuntimeError(f"Isolated process exited with non-zero code [{isolated_process.exitcode}]")

        if not queue.empty() and queue.qsize() == self.config.nproc_per_node:
            outputs = [queue.get(block=False) for _ in range(queue.qsize())]
        elif queue.empty():
            raise RuntimeError("Queue is empty, something went wrong in the isolated process")
        else:
            raise RuntimeError(f"Queue size ({queue.qsize()}) != number of ranks ({self.config.nproc_per_node})")

        reports = []

        for output in outputs:
            if "traceback" in output:
                if output["rank"] is not None:
                    self.logger.error(f"\t+ Received traceback from rank [{output['rank']}]")
                    raise ChildProcessError(output["traceback"])
                else:
                    self.logger.error("\t+ Received traceback from isolated process")
                    raise ChildProcessError(output["traceback"])
            elif "report" in output:
                self.logger.info(f"\t+ Received report from rank [{output['rank']}]")
                reports.append(BenchmarkReport.from_dict(output["report"]))
            else:
                raise RuntimeError(f"Received an unexpected response from isolated process: {output}")

        self.logger.info("\t+ Aggregating reports from all ranks")
        report = BenchmarkReport.aggregate(reports)
        report.log()

        return report


def target(
    worker: Callable[..., BenchmarkReport],
    worker_args: List[Any],
    connection: Connection,
    config: LaunchConfig,
    logger: Logger,
    queue: Queue,
):
    while True:
        if connection.poll():
            response = connection.recv()
            if response == "start":
                break

    isolated_process_pid = os.getpid()
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    os.environ["ISOLATED_PROCESS_PID"] = str(isolated_process_pid)
    setup_logging(level=log_level, to_file=log_to_file, prefix="ISOLATED-PROCESS")

    try:
        elastic_agent_launcher = elastic_launch(config=config, entrypoint=entrypoint)
        elastic_agent_launcher(worker, worker_args, logger, queue)
    except ForcedZeroExit:
        pass
    except Exception:
        logger.error("\t+ Exception occurred in isolated process. Sending traceback to main process")
        queue.put({"traceback": traceback.format_exc(), "rank": None})
    finally:
        logger.info("\t+ Exiting isolated process")
        exit(0)


def entrypoint(
    worker: Callable[..., BenchmarkReport],
    worker_args: List[Any],
    logger: Logger,
    queue: Queue,
):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    isolated_process_pid = int(os.environ["ISOLATED_PROCESS_PID"])

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    log_all_ranks = os.environ.get("LOG_ALL_RANKS", "0") == "1"

    if log_all_ranks or rank == 0:
        setup_logging(level=log_level, to_file=log_to_file, prefix=f"RANK-{rank}")
    else:
        setup_logging(level="ERROR", to_file=log_to_file, prefix=f"RANK-{rank}")

    if torch.cuda.is_available():
        logger.info(f"\t+ Setting torch.distributed cuda device to {rank}")
        torch.cuda.set_device(rank)

    logger.info("\t+ Initializing torch.distributed process group")
    torch.distributed.init_process_group()

    try:
        report = worker(*worker_args)
    except Exception:
        logger.error("\t+ Putting traceback into the the Queue")
        queue.put({"traceback": traceback.format_exc(), "rank": rank})
    else:
        logger.info("\t+ Putting benchmark report into the Queue")
        queue.put({"report": report.to_dict(), "rank": rank})

    finally:
        if rank == 0:
            queue_size = queue.qsize()
            while queue_size < world_size:
                logger.info("\t+ Waiting for other ranks to put their output into the Queue.")
                logger.info(f"\t+ Queue size: {queue_size} / World size: {world_size}")
                queue_size = queue.qsize()
                time.sleep(1)

            logger.info("\t+ All ranks have put their output into the Queue.")
            logger.info(f"\t+ Queue size: {queue_size} / World size: {world_size}")

            logger.info("\t+ Sending a forced zero exit signal to the isolated process")
            os.kill(isolated_process_pid, signal.SIGUSR2)

        logger.info("\t+ Destroying torch.distributed process group")
        torch.distributed.destroy_process_group()
