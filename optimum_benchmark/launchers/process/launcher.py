import multiprocessing as mp
import multiprocessing.connection as mp_connection
import os
import traceback
from logging import getLogger
from typing import Any, Callable, Union

from ...logging_utils import setup_logging
from ...report import BenchmarkReport
from ..base import Launcher
from ..device_isolation_utils import device_isolation_context
from .config import ProcessConfig

LOGGER = getLogger("process")


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self, config: ProcessConfig):
        super().__init__(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            LOGGER.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}.")
            mp.set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable[..., BenchmarkReport], *worker_args: Any) -> BenchmarkReport:
        ctx = mp.get_context(self.config.start_method)
        log_level = ctx.get_logger().getEffectiveLevel()
        child_conn, parent_conn = ctx.Pipe()

        isolated_process = mp.Process(target=target, args=(log_level, child_conn, worker, *worker_args), daemon=False)
        isolated_process.start()

        with device_isolation_context(
            enable=self.config.device_isolation, action=self.config.device_isolation_action, pid=isolated_process.pid
        ):
            isolated_process.join()

        if isolated_process.exitcode != 0:
            raise RuntimeError(f"Isolated process exited with non-zero code {isolated_process.exitcode}")

        if parent_conn.poll():
            response = parent_conn.recv()
        else:
            raise RuntimeError("Isolated process did not send any response")

        if "exception" in response:
            LOGGER.error("Received exception from isolated process.")
            raise ChildProcessError(response["traceback"])
        elif "report" in response:
            LOGGER.info("Received benchmark report from isolated process.")
            report = BenchmarkReport.from_dict(response["report"])
        else:
            raise RuntimeError(f"Isolated process sent an unexpected response {str(response)}")

        return report


def target(
    log_level: Union[int, str],
    conn: mp_connection.Connection,
    worker: Callable[..., BenchmarkReport],
    *worker_args: Any,
) -> None:
    isolated_process_pid = os.getpid()
    os.environ["ISOLATED_PROCESS_PID"] = str(isolated_process_pid)

    setup_logging(level=log_level, format_prefix="ISOLATED-PROCESS")
    LOGGER.info(f"Running benchmark in isolated process [{isolated_process_pid}]...")

    try:
        report = worker(*worker_args)
    except Exception as exception:
        LOGGER.error("Benchmark failed with error. Sending exception and traceback to main process...")
        conn.send({"exception": exception, "traceback": traceback.format_exc()})
    else:
        LOGGER.info("Benchmark completed successfully. Sending report to main process...")
        conn.send({"report": report.to_dict()})
    finally:
        LOGGER.info("Exiting isolated process...")
        exit(0)
