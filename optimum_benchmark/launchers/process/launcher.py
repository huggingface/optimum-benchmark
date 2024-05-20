import os
import traceback
from contextlib import ExitStack
from logging import Logger
from multiprocessing import Pipe, Process, get_start_method, set_start_method
from multiprocessing.connection import Connection
from typing import Any, Callable, List

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
            self.logger.info("\t+ Warming up multiprocessing context")
            # creates the resource tracker with default executable
            dummy_process = Process()
            dummy_process.start()
            dummy_process.join()

    def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
        child_connection, parent_connection = Pipe()
        isolated_process = Process(
            target=target, args=(worker, worker_args, child_connection, self.logger), daemon=False
        )

        with ExitStack() as stack:
            if self.config.numactl:
                stack.enter_context(self.numactl_executable())

            self.logger.info("\t+ Starting isolated process")
            isolated_process.start()
            while True:
                if parent_connection.poll():
                    message = parent_connection.recv()
                    if message == "READY":
                        self.logger.info("\t+ Isolated process is ready")
                        break
                    else:
                        raise RuntimeError(f"Unexpected message from isolated process: {message}")

        with ExitStack() as stack:
            if self.config.device_isolation:
                stack.enter_context(self.device_isolation(isolated_process.pid))

            parent_connection.send("START")
            isolated_process.join()

        if isolated_process.exitcode != 0:
            raise RuntimeError(f"Isolated process exited with non-zero code {isolated_process.exitcode}")

        if parent_connection.poll():
            response = parent_connection.recv()

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
    connection: Connection,
    logger: Logger,
) -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    setup_logging(level=log_level, to_file=log_to_file, prefix="ISOLATED-PROCESS")

    connection.send("READY")

    while True:
        if connection.poll():
            message = connection.recv()
            if message == "START":
                logger.info("\t+ Starting benchmark in isolated process")
                break
            else:
                raise RuntimeError(f"Unexpected message from main process: {message}")

    try:
        report = worker(*worker_args)
    except Exception:
        logger.error("\t+ Sending traceback to main process")
        connection.send({"traceback": traceback.format_exc()})
    else:
        logger.info("\t+ Sending report to main process")
        connection.send({"report": report.to_dict()})
    finally:
        logger.info("\t+ Exiting isolated process")
        connection.close()
        exit(0)
