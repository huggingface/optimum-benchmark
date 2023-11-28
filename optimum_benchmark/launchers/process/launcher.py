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
            # daemon=True
            # TODO: move isolation process to the launcher and make this daemon
            # which is currently not possible because daemon process cannot have children
        )

        process.start()
        LOGGER.info(f"\t+ Launched experiment in process with PID {process.pid}.")
        process.join()

        if process.exitcode is None:
            LOGGER.warning("\t+ Process did not exit even after getting benchmark result, terminating it.")
            process.terminate()
            process.join()

            if process.exitcode is None:
                LOGGER.error("\t+ Process did not exit even after being terminated, killing it.")
                process.kill()
                process.join()

        if process.exitcode != 0:
            exit_code = process.exitcode
            LOGGER.error(f"\t+ Process exited with code {exit_code}, closing it.")
            process.close()
            raise RuntimeError(f"Process exited with code {exit_code}")

        LOGGER.info("\t+ Process exited successfully, closing it.")
        process.close()


def target(fn, *args):
    """
    This a pickalable function that correctly sets up the logging configuration
    """
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    fn(*args)
