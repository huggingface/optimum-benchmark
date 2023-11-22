import logging.config
import multiprocessing as mp
from logging import getLogger
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Callable

from omegaconf import OmegaConf

from ..base import Launcher
from .config import ProcessConfig

if TYPE_CHECKING:
    from ...benchmarks.base import Benchmark

LOGGER = getLogger("process")

# Create the Queue
QUEUE = Queue()


class ProcessLauncher(Launcher[ProcessConfig]):
    NAME = "process"

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: ProcessConfig) -> None:
        super().configure(config)

    def launch(self, worker: Callable, *worker_args) -> "Benchmark":
        # Set the multiprocessing start method if not already set
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method(self.config.start_method)

        # Create the process
        process = Process(
            target=target,
            args=(worker, *worker_args),
            daemon=True,
        )

        # Start the process
        process.start()

        # Wait for the process to finish
        process.join()

        if process.exitcode != 0:
            raise RuntimeError(f"Process exited with code {process.exitcode}")

        # Get the benchmark from the queue
        benchmark = QUEUE.get()

        return benchmark


def target(fn, *args):
    """
    This a pickalable function that correctly sets up the logging configuration
    """
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    # Run the function
    result = fn(*args)

    # Put the result in the queue
    QUEUE.put(result)
