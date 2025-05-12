import os
import traceback
from contextlib import ExitStack
from logging import Logger
from multiprocessing import Pipe, Process, get_start_method, set_start_method
from multiprocessing.connection import Connection
from typing import Any, Callable, List

import psutil
import torch.distributed
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from ...benchmark.report import BenchmarkReport
from ...logging_utils import setup_logging
from ...process_utils import receive_serializable, send_serializable, sync_with_child, sync_with_parent
from ..base import Launcher
from .config import TorchrunConfig


class TorchrunLauncher(Launcher[TorchrunConfig]):
    NAME = "torchrun"

    def __init__(self, config: TorchrunConfig):
        super().__init__(config)

        if get_start_method(allow_none=True) != self.config.start_method:
            self.logger.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}")
            set_start_method(self.config.start_method, force=True)

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
        parent_connection, child_connection = Pipe()
        main_process_pid = os.getpid()
        isolated_process = Process(
            target=target,
            args=(worker, worker_args, child_connection, main_process_pid, self.launch_config, self.logger),
            daemon=False,
        )

        with ExitStack() as stack:
            if self.config.numactl:
                stack.enter_context(self.numactl_executable())

            isolated_process.start()

            if isolated_process.is_alive():
                sync_with_child(parent_connection)
            else:
                raise RuntimeError("Could not synchronize with isolated process")

            if self.config.device_isolation:
                stack.enter_context(self.device_isolation(isolated_process.pid))

            if isolated_process.is_alive():
                sync_with_child(parent_connection)
            else:
                raise RuntimeError("Could not synchronize with isolated process")

            isolated_process.join()

        if isolated_process.exitcode != 0:
            raise RuntimeError(f"Isolated process exited with non-zero code {isolated_process.exitcode}")

        if parent_connection.poll():
            response_type = parent_connection.recv()
        else:
            raise RuntimeError("Isolated process did not send any response")

        if response_type == "traceback":
            self.logger.error("\t+ Received traceback from isolated process")
            traceback_str = receive_serializable(parent_connection, self.logger)
            raise ChildProcessError(traceback_str)
        elif response_type == "outputs":
            self.logger.info("\t+ Received outputs from isolated process")
            outputs = receive_serializable(parent_connection, self.logger)

            reports = []
            for rank, report_dict in outputs.items():
                if isinstance(report_dict, str):
                    self.logger.error(f"\t+ Received traceback from rank process {rank}")
                    raise ChildProcessError(report_dict)

                self.logger.info(f"\t+ Received report from rank process {rank}")
                report = BenchmarkReport.from_dict(report_dict)
                reports.append(report)

        self.logger.info("\t+ Aggregating reports from all rank processes")
        report = BenchmarkReport.aggregate_across_processes(reports)
        return report


def target(
    worker: Callable[..., BenchmarkReport],
    worker_args: List[Any],
    child_connection: Connection,
    main_process_pid: int,
    config: LaunchConfig,
    logger: Logger,
):
    main_process = psutil.Process(main_process_pid)

    if main_process.is_running():
        sync_with_parent(child_connection)
    else:
        raise RuntimeError("Could not synchronize with main process")

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    setup_logging(level=log_level, to_file=log_to_file, prefix="ISOLATED-PROCESS")

    if main_process.is_running():
        sync_with_parent(child_connection)
    else:
        raise RuntimeError("Could not synchronize with main process")

    try:
        elastic_agent_launcher = elastic_launch(config=config, entrypoint=entrypoint)
        outputs = elastic_agent_launcher(worker, worker_args, logger)
    except Exception:
        logger.error("\t+ Sending traceback to main process")
        child_connection.send("traceback")
        send_serializable(child_connection, traceback.format_exc(), logger)
    else:
        logger.info("\t+ Sending outputs to main process")
        child_connection.send("outputs")
        send_serializable(child_connection, outputs, logger)
    finally:
        logger.info("\t+ Exiting isolated process")
        child_connection.close()
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

    backend = None
    if hasattr(torch.mps, "is_available") and torch.mps.is_available():
        backend = "gloo"

    logger.info("\t+ Initializing torch.distributed process group")
    torch.distributed.init_process_group(backend=backend)

    try:
        report = worker(*worker_args)
    except Exception:
        logger.error("\t+ Benchmark failed with an exception")
        output = traceback.format_exc()
    else:
        logger.info("\t+ Benchmark completed successfully")
        output = report.to_dict()
    finally:
        logger.info("\t+ Destroying torch.distributed process group")
        torch.distributed.destroy_process_group()
        logger.info("\t+ Exiting rank process")
        return output
