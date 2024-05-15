import multiprocessing as mp
import os
import traceback
from logging import Logger
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import Any, Callable, List

from ...logging_utils import setup_logging
from ...report import BenchmarkReport
from ..base import Launcher
from .config import ProcessConfig


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self, config: ProcessConfig):
        super().__init__(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            self.logger.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}")
            mp.set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable[..., BenchmarkReport], worker_args: List[Any]) -> BenchmarkReport:
        ctx = mp.get_context(self.config.start_method)
        child_connection, parent_connection = ctx.Pipe()

        isolated_process = Process(
            target=target, args=(worker, worker_args, child_connection, self.logger), daemon=False
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
