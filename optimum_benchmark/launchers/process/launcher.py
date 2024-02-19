from typing import Callable
import multiprocessing as mp
from logging import getLogger
from multiprocessing import Process, Queue, Lock
from multiprocessing.synchronize import Lock as LockType

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
        lock = Lock()
        queue = Queue(1000)
        current_log_level = getLogger().getEffectiveLevel()
        worker_process = Process(
            target=target, args=(worker, queue, lock, current_log_level, *worker_args), daemon=False
        )

        with device_isolation(enabled=self.config.device_isolation):
            worker_process.start()
            LOGGER.info(f"\t+ Launched worker process with PID {worker_process.pid}.")
            worker_process.join()

        try:
            report = queue.get()
        except EOFError:
            LOGGER.error(f"\t+ Worker process exited with code {worker_process.exitcode}, forwarding...")
            exit(worker_process.exitcode)

        return report


def target(fn: Callable, queue: Queue, lock: LockType, log_level: str, *args):
    """
    This a pickalable function that correctly sets up the logging configuration for the worker process,
    and puts the output of the worker function into a lock-protected queue.
    """

    setup_logging(log_level)

    out = fn(*args)

    lock.acquire()
    queue.put(out)
    lock.release()
