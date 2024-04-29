import multiprocessing as mp
import os
from logging import getLogger
from typing import Callable

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
        queue = ctx.Queue()
        lock = ctx.Lock()

        isolated_process = mp.Process(target=target, args=(worker, queue, lock, log_level, *worker_args), daemon=False)
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
        elif queue.empty():
            raise RuntimeError("No report was returned by the isolated process.")

        report: BenchmarkReport = queue.get(block=True)
        report.log()

        return report


def target(worker, queue: mp.Queue, lock: mp.Lock, log_level, *worker_args):
    os.environ["ISOLATED_PROCESS_PID"] = str(os.getpid())
    setup_logging(level=log_level, prefix="ISOLATED-PROCESS")
    LOGGER.info(f"Running benchmark in isolated process [{os.getpid()}].")

    report = worker(*worker_args)

    lock.acquire()
    queue.put(report, block=True)
    lock.release()
