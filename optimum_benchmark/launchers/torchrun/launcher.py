import logging.config
import os
from logging import getLogger
from typing import Callable

from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, launch_agent

from ..base import Launcher
from .config import TorchrunConfig

LOGGER = getLogger("torchrun")


class TorchrunLauncher(Launcher[TorchrunConfig]):
    NAME = "torchrun"

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: "TorchrunConfig") -> None:
        super().configure(config)

        LOGGER.info(f"Running {self.config.nproc_per_node} processes per node")

    def launch(self, worker: Callable, *worker_args):
        launch_config = LaunchConfig(
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            nproc_per_node=self.config.nproc_per_node,
            run_id=self.config.run_id,
            role=self.config.role,
            monitor_interval=self.config.monitor_interval,
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

        benchmarks = launch_agent(
            entrypoint=entrypoint,
            args=(worker, *worker_args),
            config=launch_config,
        )

        return benchmarks[0]


@record
def entrypoint(fn, *args):
    """
    This a pickalable function that correctly sets up the logging configuration
    """

    if os.environ.get("LOCAL_RANK", None) == "0":
        hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    else:
        logging.disable(logging.CRITICAL)

    return fn(*args)
