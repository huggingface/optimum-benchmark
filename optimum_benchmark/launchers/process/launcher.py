import multiprocessing as mp
import os
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

    def launch(self, worker: Callable, *worker_args) -> BenchmarkReport:
        ctx = mp.get_context(self.config.start_method)
        log_level = ctx.get_logger().getEffectiveLevel()

        isolated_process = mp.Process(target=target, args=(worker, log_level, *worker_args), daemon=False)
        isolated_process.start()

        with device_isolation_context(
            enable=self.config.device_isolation, action=self.config.device_isolation_action, pid=isolated_process.pid
        ):
            while isolated_process.is_alive():
                isolated_process.join(timeout=1)
                if isolated_process.exitcode is not None:
                    break

        if isolated_process.exitcode != 0:
            raise RuntimeError(f"Process exited with non-zero code {isolated_process.exitcode}")

        if not os.path.exists("isolated_process_report.json"):
            raise RuntimeError("Could not find report from isolated process.")

        LOGGER.info("\t+ Loading report from isolated process.")
        report: BenchmarkReport = BenchmarkReport.from_json("isolated_process_report.json")
        report.log()

        return report


def target(worker: Callable[..., BenchmarkReport], log_level: Union[int, str], *worker_args: Any) -> None:
    isolated_process_pid = os.getpid()
    os.environ["ISOLATED_PROCESS_PID"] = str(isolated_process_pid)

    setup_logging(level=log_level, prefix="ISOLATED-PROCESS")
    LOGGER.info(f"Running benchmark in isolated process [{isolated_process_pid}].")

    report = worker(*worker_args)

    LOGGER.info("Saving report from isolated process.")
    report.save_json("isolated_process_report.json")

    LOGGER.info("Exiting isolated process.")
