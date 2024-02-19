from typing import Callable
from logging import getLogger

from ..base import Launcher
from .config import ProcessConfig
from ...logging_utils import setup_logging
from ..isolation_utils import device_isolation
from ...benchmarks.report import BenchmarkReport

import torch.multiprocessing as mp

LOGGER = getLogger("process")


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self, config: ProcessConfig):
        super().__init__(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            LOGGER.info(f"\t+ Setting multiprocessing start method to {self.config.start_method}.")
            mp.set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable, *worker_args) -> BenchmarkReport:
        log_level = getLogger().getEffectiveLevel()

        ctx = mp.get_context(self.config.start_method)
        queue = ctx.Queue()
        lock = ctx.Lock()

        with device_isolation(enabled=self.config.device_isolation):
            process_context = mp.start_processes(
                entrypoint,
                args=(worker, queue, lock, log_level, *worker_args),
                start_method=self.config.start_method,
                daemon=False,
                join=False,
                nprocs=1,
            )
            LOGGER.info(f"\t+ Launched worker process(es) with PID(s): {process_context.pids()}")
            while not process_context.join():
                pass

        try:
            report: BenchmarkReport = queue.get()
        except EOFError:
            raise RuntimeError("Worker process did not return a report")

        return report


def entrypoint(_, worker, queue, lock, log_level, *worker_args):
    """
    This a pickalable function that correctly sets up the logging configuration for the worker process,
    and puts the output of the worker function into a lock-protected queue.
    """

    setup_logging(log_level)

    worker_output = worker(*worker_args)

    lock.acquire()
    queue.put(worker_output)
    lock.release()
