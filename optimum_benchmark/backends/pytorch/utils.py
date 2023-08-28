import logging.config
import os
from logging import getLogger
from typing import Optional

import torch
from omegaconf import OmegaConf


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
    """PyTorch DDP subprocesses do not inherit from Hydra logger.
    Thus, we need to reconfigure the logger for the workers.
    """
    if os.environ["RANK"] == "0" or log_all:
        # TODO: also configure logging for other ranks
        hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    return getLogger(name)
