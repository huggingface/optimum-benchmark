import logging.config
import multiprocessing as mp
import os
from logging import getLogger
from multiprocessing import Process
from typing import Callable

from omegaconf import OmegaConf

from ..base import Launcher
from ..isolation_utils import devices_isolation
from .config import ProcessConfig

LOGGER = getLogger("process")


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: ProcessConfig) -> None:
        super().configure(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            LOGGER.info(f"Setting multiprocessing start method to {self.config.start_method}.")
            mp.set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable, *worker_args):
        worker_process = Process(target=target, args=(worker, *worker_args), daemon=True)
        worker_process.start()
        LOGGER.info(f"\t+ Launched worker process with PID {worker_process.pid}.")

        with devices_isolation(
            enabled=self.config.devices_isolation,
            permitted_pids={os.getpid(), worker_process.pid},
        ):
            worker_process.join()

        if worker_process.exitcode != 0:
            LOGGER.error(f"Worker process exited with code {worker_process.exitcode}, forwarding...")
            exit(worker_process.exitcode)


def target(fn, *args):
    """This a pickalable function that correctly sets up the logging configuration for the worker process."""
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    fn(*args)
