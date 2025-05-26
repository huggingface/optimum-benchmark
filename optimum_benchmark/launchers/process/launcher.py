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

from ...benchmark.report import BenchmarkReport
from ...logging_utils import setup_logging
from ...process_utils import sync_with_child, sync_with_parent
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
            self.logger.info("\t+ Received report dictionary from isolated process")
            report = BenchmarkReport.from_dict(response)
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
    file_based_comm_threshold = int(os.environ.get("FILE_BASED_COMM_THRESHOLD", "1_000_000"))

    if main_process.is_running():
        sync_with_parent(child_connection)
    else:
        raise RuntimeError("Could not synchronize with main process")

    try:
        report = worker(*worker_args)
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
        logger.info("\t+ Sending report dictionary to main process")
        report_dict = report.to_dict()
        report_size = len(str(report_dict))
        if report_size <= file_based_comm_threshold:
            logger.info(f"\t+ Sending report dictionary directly ({report_size} bytes)")
            child_connection.send(report_dict)
        else:
            logger.warning(f"\t+ Sending report dictionary ({report_size} bytes) through file-based communication")
        temp_file_path = os.path.join(tempfile.gettempdir(), f"optimum_benchmark_{os.getpid()}.pkl")
        with open(temp_file_path, "wb") as f:
            pickle.dump(report_dict, f)
        child_connection.send(temp_file_path)

    finally:
        logger.info("\t+ Exiting isolated process")
        child_connection.close()
        exit(0)
