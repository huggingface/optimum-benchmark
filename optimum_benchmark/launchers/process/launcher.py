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

    def launch(self, worker: Callable, *worker_args):
        # Set the multiprocessing start method if not already set
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method(self.config.start_method)

        # Create the process
        process = Process(
            target=target,
            args=(worker, *worker_args),
        )

        # Start the process
        process.start()
        # Wait to finish
        process.join()

        if process.exitcode is None:
            LOGGER.warning("Process did not terminate even after getting benchmark result")
        elif process.exitcode != 0:
            raise RuntimeError(f"Process exited with code {process.exitcode}")


def target(fn, *args):
    """
    This a pickalable function that correctly sets up the logging configuration
    """
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    fn(*args)
