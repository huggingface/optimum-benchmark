import os
from logging import getLogger
from typing import Callable

import torch.multiprocessing as mp

from ...benchmarks.report import BenchmarkReport
from ...logging_utils import setup_logging
from ..base import Launcher
from ..isolation_utils import device_isolation
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
        log_level = getLogger().getEffectiveLevel()

        ctx = mp.get_context(self.config.start_method)
        queue = ctx.Queue()
        lock = ctx.Lock()

        with device_isolation(
            isolated_pid=os.getpid(),
            enabled=self.config.device_isolation,
            action=self.config.device_isolation_action,
        ):
            process_context = mp.start_processes(
                entrypoint,
                args=(worker, queue, lock, log_level, *worker_args),
                start_method=self.config.start_method,
                daemon=False,
                join=False,
                nprocs=1,
            )
            LOGGER.info(f"\t+ Launched benchmark in isolated process {process_context.pids()[0]}.")
            while not process_context.join():
                pass

        report: BenchmarkReport = queue.get()

        return report


def entrypoint(i, worker, queue, lock, log_level, *worker_args):
    """
    This a pickalable function that correctly sets up the logging configuration for the worker process,
    and puts the output of the worker function into a lock-protected queue.
    """

    setup_logging(log_level, prefix=f"PROC-{i}")

    worker_output = worker(*worker_args)

    lock.acquire()
    queue.put(worker_output)
    lock.release()
