import os
from typing import Callable
import multiprocessing as mp
from logging import getLogger
from multiprocessing import Process, Queue

from ..base import Launcher
from .config import ProcessConfig
from ...logging_utils import setup_logging
from ..isolation_utils import device_isolation
from ...benchmarks.report import BenchmarkReport


LOGGER = getLogger("process")


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self, config: ProcessConfig):
        super().__init__(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            LOGGER.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}.")
            mp.set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable, *worker_args) -> BenchmarkReport:
        # worker process can't be daemon since it might spawn its own processes
        queue = Queue()
        current_log_level = getLogger().getEffectiveLevel()
        worker_process = Process(
            daemon=False,
            target=target,
            args=(worker, queue, current_log_level, *worker_args),
        )
        worker_process.start()
        LOGGER.info(f"\t+ Launched worker process with PID {worker_process.pid}.")

        with device_isolation(enabled=self.config.device_isolation, benchmark_pid=os.getpid()):
            worker_process.join()

        if worker_process.exitcode != 0:
            LOGGER.error(f"\t+ Worker process exited with code {worker_process.exitcode}, forwarding...")
            exit(worker_process.exitcode)

        report = queue.get()

        return report


def target(fn, q, log_level, *args):
    """This a pickalable function that correctly sets up the logging configuration for the worker process."""

    setup_logging(log_level)

    out = fn(*args)

    q.put(out)
