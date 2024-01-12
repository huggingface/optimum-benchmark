import logging.config
import multiprocessing as mp
import os
from logging import getLogger
from typing import Callable

from omegaconf import OmegaConf
from torch.distributed import FileStore
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, launch_agent

from ..base import Launcher
from ..isolation_utils import device_isolation
from .config import TorchrunConfig

LOGGER = getLogger("torchrun")


class TorchrunLauncher(Launcher[TorchrunConfig]):
    NAME = "torchrun"

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: TorchrunConfig) -> None:
        super().configure(config)

        if mp.get_start_method(allow_none=True) != self.config.start_method:
            LOGGER.info(f"Setting multiprocessing start method to {self.config.start_method}.")
            mp.set_start_method(self.config.start_method, force=True)

    def launch(self, worker: Callable, *worker_args):
        launch_config = LaunchConfig(
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            nproc_per_node=self.config.nproc_per_node,
            role=self.config.role,
            monitor_interval=self.config.monitor_interval,
            run_id=self.config.rdzv_id,
            rdzv_endpoint=self.config.rdzv_endpoint,
            rdzv_backend=self.config.rdzv_backend,
            rdzv_configs=self.config.rdzv_configs,
            max_restarts=self.config.max_restarts,
            start_method=self.config.start_method,
            metrics_cfg=self.config.metrics_cfg,
            redirects=Std.from_str(self.config.redirects),
            tee=Std.from_str(self.config.tee),
            local_addr=self.config.local_addr,
            log_dir=self.config.log_dir,
        )

        with device_isolation(
            enabled=self.config.device_isolation,
            permitted_pids={os.getpid()},
        ):
            LOGGER.info(f"\t+ Launching torchrun/torchelastic agent with {self.config.nproc_per_node} processes")
            launch_agent(
                entrypoint=entrypoint,
                args=(worker, *worker_args),
                config=launch_config,
            )


@record
def entrypoint(fn, *args):
    """
    This a pickalable function that correctly sets up the logging configuration
    """
    store = FileStore("torchrun_filestore")
    store.set(f"rank_{os.environ['RANK']}", str(os.getpid()))

    if os.environ["RANK"] == "0":
        hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    else:
        logging.disable(logging.CRITICAL)

    fn(*args)
