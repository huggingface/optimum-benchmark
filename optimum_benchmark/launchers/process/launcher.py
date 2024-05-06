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

    def launch(self, worker: Callable[..., BenchmarkReport], *worker_args: Any) -> BenchmarkReport:
        ctx = mp.get_context(self.config.start_method)
        log_level = ctx.get_logger().getEffectiveLevel()
        queue = ctx.Queue()
        lock = ctx.Lock()

        isolated_process = mp.Process(
            target=target,
            args=(worker, *worker_args),
            kwargs={"log_level": log_level, "queue": queue, "lock": lock},
            daemon=False,
        )
        isolated_process.start()

        with device_isolation_context(
            enable=self.config.device_isolation, action=self.config.device_isolation_action, pid=isolated_process.pid
        ):
            isolated_process.join()

        if not queue.empty():
            LOGGER.info("Retrieving report from queue.")
            report = BenchmarkReport.from_dict(queue.get())
        elif isolated_process.exitcode != 0:
            raise RuntimeError(f"Process exited with non-zero code {isolated_process.exitcode}")
        else:
            raise RuntimeError("Could not retrieve report from isolated process.")

        return report


def target(
    worker: Callable[..., BenchmarkReport], *worker_args, log_level: Union[int, str], queue: mp.Queue, lock: mp.Lock
) -> None:
    isolated_process_pid = os.getpid()
    os.environ["ISOLATED_PROCESS_PID"] = str(isolated_process_pid)

    setup_logging(level=log_level, format_prefix="ISOLATED-PROCESS")
    LOGGER.info(f"Running benchmark in isolated process [{isolated_process_pid}].")

    report = worker(*worker_args)

    lock.acquire()
    LOGGER.info("Putting report in queue.")
    queue.put(report.to_dict())
    lock.release()

    LOGGER.info("Exiting isolated process.")
    exit(0)
