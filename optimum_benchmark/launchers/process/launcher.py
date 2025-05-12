import os
import traceback
from contextlib import ExitStack
from logging import Logger
from multiprocessing import Pipe, Process, get_start_method, set_start_method
from multiprocessing.connection import Connection
from typing import Any, Callable, List

import psutil

from ...benchmark.report import BenchmarkReport
from ...logging_utils import setup_logging
from ...process_utils import receive_serializable, send_serializable, sync_with_child, sync_with_parent
from ..base import Launcher
from .config import ProcessConfig


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self, config: ProcessConfig):
        super().__init__(config)

        if get_start_method(allow_none=True) != self.config.start_method:
            self.logger.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}")
            set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
        child_connection, parent_connection = Pipe()
        main_process_pid = os.getpid()
        isolated_process = Process(
            target=target, args=(worker, worker_args, child_connection, main_process_pid, self.logger), daemon=False
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

            while not parent_connection.poll():
                pass

        if isolated_process.exitcode is not None and isolated_process.exitcode != 0:
            raise RuntimeError(f"Isolated process exited with non-zero code {isolated_process.exitcode}")

        if parent_connection.poll():
            response_type = parent_connection.recv()
        else:
            raise RuntimeError("Received no response from isolated process")

        if response_type == "traceback":
            self.logger.error("\t+ Receiving traceback from isolated process")
            traceback_str = receive_serializable(parent_connection, self.logger)
            raise ChildProcessError(traceback_str)
        elif response_type == "report":
            self.logger.info("\t+ Receiving report from isolated process")
            report_dict = receive_serializable(parent_connection, self.logger)
            report = BenchmarkReport.from_dict(report_dict)
        else:
            raise RuntimeError(f"Received an unexpected response type from isolated process: {response_type}")

        return report


def target(
    worker: Callable[..., BenchmarkReport],
    worker_args: List[Any],
    child_connection: Connection,
    main_process_pid: int,
    logger: Logger,
) -> None:
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
        report = worker(*worker_args)
    except Exception:
        logger.error("\t+ Sending traceback string to main process")
        child_connection.send("traceback")
        send_serializable(child_connection, traceback.format_exc(), logger)
    else:
        logger.info("\t+ Sending report dictionary to main process")
        child_connection.send("report")
        send_serializable(child_connection, report.to_dict(), logger)
    finally:
        logger.info("\t+ Exiting isolated process")
        child_connection.close()
        exit(0)
