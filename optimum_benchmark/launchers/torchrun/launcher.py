import os
import pickle
import tempfile
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
from ...process_utils import sync_with_child, sync_with_parent
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

            while isolated_process.is_alive() and not parent_connection.poll():
                pass

        if not isolated_process.is_alive() and isolated_process.exitcode is not None and isolated_process.exitcode != 0:
            raise RuntimeError(f"Isolated process exited with non-zero code {isolated_process.exitcode}")

        if parent_connection.poll():
            response = parent_connection.recv()
        else:
            raise RuntimeError("Isolated process did not send any response")

        if isinstance(response, str) and response.startswith(tempfile.gettempdir()):
            response = pickle.load(open(response, "rb"))

        if isinstance(response, str):
            self.logger.error("\t+ Received traceback from isolated process")
            raise ChildProcessError(response)
        elif isinstance(response, dict):
            self.logger.info("\t+ Received outputs from isolated process")
            reports = []
            for rank, report_dict in response.items():
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
    file_based_comm_threshold = int(os.environ.get("FILE_BASED_COMM_THRESHOLD", "1_000_000"))

    if main_process.is_running():
        sync_with_parent(child_connection)
    else:
        raise RuntimeError("Could not synchronize with main process")

    try:
        elastic_agent_launcher = elastic_launch(config=config, entrypoint=entrypoint)
        outputs = elastic_agent_launcher(worker, worker_args, logger)

    except Exception:
        logger.error("\t+ Sending traceback string to main process")
        str_traceback = traceback.format_exc()
        traceback_size = len(str_traceback)
        if traceback_size <= file_based_comm_threshold:
            logger.info(f"\t+ Sending traceback string directly ({traceback_size} bytes)")
            child_connection.send(str_traceback)
        else:
            logger.warning(f"\t+ Sending traceback string ({traceback_size} bytes) through file-based communication")
            temp_file_path = os.path.join(tempfile.gettempdir(), f"optimum_benchmark_{os.getpid()}.txt")
            with open(temp_file_path, "wb") as f:
                pickle.dump(str_traceback, f)
            child_connection.send(temp_file_path)

    else:
        logger.info("\t+ Sending outputs to main process")
        outputs_size = len(str(outputs))
        if outputs_size <= file_based_comm_threshold:
            logger.info(f"\t+ Sending outputs directly ({outputs_size} bytes)")
            child_connection.send(outputs)
        else:
            logger.warning(f"\t+ Sending outputs ({outputs_size} bytes) through file-based communication")
            temp_file_path = os.path.join(tempfile.gettempdir(), f"optimum_benchmark_{os.getpid()}.pkl")
            with open(temp_file_path, "wb") as f:
                pickle.dump(outputs, f)
            child_connection.send(temp_file_path)

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
