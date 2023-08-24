from logging import getLogger
from typing import Optional
import logging.config
import os

import torch
from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing import Std

OmegaConf.register_new_resolver("device_count", lambda: torch.cuda.device_count())


DEFAULT_COMPILE_CONFIG = {
    "fullgraph": False,
    "dynamic": False,
    "backend": "inductor",
    "mode": None,
    "options": None,
    "disable": False,
}

# from https://github.com/pytorch/pytorch/blob/v2.0.0/torch/distributed/launcher/api.py#L29
# adjusted to the defaults of torch.distributed.run
# defined in https://github.com/pytorch/pytorch/blob/v2.0.0/torch/distributed/run.py#L770
# TODO: decide wrther to use torch.distributed.run arguments or the ones from
# torch.distributed.launcher.api
DEFAULT_DDP_CONFIG = {
    "min_nodes": 1,
    "max_nodes": 1,
    "run_id": "none",
    "nproc_per_node": "${device_count:}",
    "role": "default",
    "rdzv_endpoint": "127.0.0.1:29500",
    "rdzv_backend": "static",
    "rdzv_configs": {
        "timeout": 900,
        "rank": 0,
    },
    "max_restarts": 0,
    "monitor_interval": 5,
    "start_method": "spawn",
    "log_dir": None,
    "metrics_cfg": {},
    "local_addr": None,
    "redirects": Std.NONE,
    "tee": Std.NONE,
}


def randomize_weights(model):
    for param in model.parameters():
        if torch.cuda.is_available() and param.device.type == "cpu":
            # we take advantage of the fact that a cuda device
            # is available to use cuda kernels for randomization
            # this is slower than asynchronous randomization while
            # model is fully on gpu (because of data transfer) but
            # faster than randomization while model is on cpu
            param.data.cuda().normal_(mean=0.0, std=0.2).cpu()
        else:
            param.data.normal_(mean=0.0, std=0.2)


def get_worker_logger(
    name: Optional[str] = None,
    log_all: bool = False,
) -> logging.Logger:
    """
    PyTorch DDP subprocesses do not inherit from Hydra logger.
    Thus, we need to reconfigure the logger for the workers.
    """
    if os.environ["RANK"] == "0" or log_all:
        # TODO: also configure logging for other ranks
        hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
        logging.config.dictConfig(
            OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True)
        )

    return getLogger(name)
