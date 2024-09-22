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
from ..base import Launcher
from .config import ProcessConfig


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self, config: ProcessConfig):
        super().__init__(config)

        if get_start_method(allow_none=True) != self.config.start_method:
            self.logger.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}")
            set_start_method(self.config.start_method, force=True)
            # creates the resource tracker with default executable
            self.logger.info("\t+ Warming up multiprocessing context")
            dummy_process = Process(target=dummy_target, daemon=False)
            dummy_process.start()
            dummy_process.join()
            dummy_process.close()

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

        with ExitStack() as stack:
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
            response = parent_connection.recv()
        else:
            raise RuntimeError("Received no response from isolated process")

        if "traceback" in response:
            self.logger.error("\t+ Received traceback from isolated process")
            raise ChildProcessError(response["traceback"])
        elif "exception" in response:
            self.logger.error("\t+ Received exception from isolated process")
            raise ChildProcessError(response["exception"])
        elif "report" in response:
            self.logger.info("\t+ Received report from isolated process")
            report = BenchmarkReport.from_dict(response["report"])
            report.log()
        else:
            raise RuntimeError(f"Received an unexpected response from isolated process: {response}")

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
        logger.error("\t+ Sending traceback to main process")
        child_connection.send({"traceback": traceback.format_exc()})
    else:
        logger.info("\t+ Sending report to main process")
        child_connection.send({"report": report.to_dict()})
    finally:
        logger.info("\t+ Exiting isolated process")
        child_connection.close()
        exit(0)


def sync_with_parent(child_connection: Connection) -> None:
    if child_connection.poll():
        response = child_connection.recv()
    else:
        raise RuntimeError("Received no response from main process")

    if response == "SYNC":
        return
    else:
        raise RuntimeError(f"Received an unexpected response from main process: {response}")


def sync_with_child(parent_connection: Connection) -> None:
    parent_connection.send("SYNC")
    parent_connection.recv()


def dummy_target() -> None:
    exit(0)
