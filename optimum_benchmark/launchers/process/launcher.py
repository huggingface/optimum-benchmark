import logging.config
import multiprocessing as mp
from logging import getLogger
from multiprocessing import Process
from typing import Callable

from omegaconf import OmegaConf

from ..base import Launcher
from .config import ProcessConfig

LOGGER = getLogger("process")


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: ProcessConfig) -> None:
        super().configure(config)

    def launch(self, worker: Callable, *worker_args) -> None:
        # Set the multiprocessing start method if not already set
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method(self.config.start_method)

        # Execute in a separate process
        p = Process(
            target=target,
            args=(worker, *worker_args),
            daemon=True,
        )
        p.start()
        benchmark = p.join()

        # Exit with the same exit code as the child process
        if p.exitcode != 0:
            LOGGER.error(f"Child process exited with code {p.exitcode}")
            exit(p.exitcode)
        else:
            return benchmark


def target(fn, *args):
    """
    This a pickalable function that correctly sets up the logging configuration
    """
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    return fn(*args)
