# TODO: this can be reformulated as a subclass of backend, from which pytorch and onnxruntime and any other backend

import logging.config
import os
from logging import getLogger
from typing import TYPE_CHECKING, Optional

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from transformers import TrainerState

from ..import_utils import is_torch_distributed_available

# from launchConfig in https://github.com/pytorch/pytorch/blob/v2.0.0/torch/distributed/launcher/api.py#L29 adjusted
# to defaults of torch.distributed.run in https://github.com/pytorch/pytorch/blob/v2.0.0/torch/distributed/run.py#L770
DDP_CONFIG = {
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
}


def get_worker_logger(name: Optional[str] = None, log_all: bool = False) -> logging.Logger:
    """
    PyTorch DDP subprocesses do not inherit from Hydra logger.
    Thus, we need to reconfigure the logger for the workers.
    """
    if os.environ["RANK"] == "0" or log_all:
        # TODO: also configure logging for other ranks
        hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))

    return getLogger(name)


def training_worker(args) -> "TrainerState":
    dataset_format = args[0]
    backend_logger = args[1]
    trainer_class = args[2]
    training_arguments_class = args[3]
    use_ddp = args[4]
    training_dataset = args[5]
    training_arguments = args[6]
    training_data_collator = args[7]
    training_callbacks = args[8]
    pretrained_model = args[9]

    if use_ddp:
        LOGGER_WORKER = get_worker_logger("pytorch-ddp-worker", log_all=False)
        env_variables = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_MAX_RESTARTS"]
        LOGGER_WORKER.info("Initializing DDP worker")
        for env_var in env_variables:
            LOGGER_WORKER.info(f"{env_var}: {os.environ.get(env_var)}")
    else:
        LOGGER_WORKER = backend_logger

    LOGGER_WORKER.info(f"\t+ Setting dataset format to `{dataset_format}`.")
    training_dataset.set_format(type=dataset_format, columns=list(training_dataset.features.keys()))
    LOGGER_WORKER.info("\t+ Wrapping training arguments with transformers.TrainingArguments")
    training_arguments = training_arguments_class(**training_arguments)
    LOGGER_WORKER.info("\t+ Wrapping model with transformers.Trainer")
    trainer = trainer_class(
        model=pretrained_model,
        args=training_arguments,
        callbacks=training_callbacks,
        train_dataset=training_dataset,
        data_collator=training_data_collator,
    )
    LOGGER_WORKER.info("\t+ Starting training")
    trainer.train()
    LOGGER_WORKER.info("\t+ Training finished successfully")
    return trainer.state


def record(func):
    if is_torch_distributed_available():
        from torch.distributed.elastic.multiprocessing.errors import record

        record(func)
    else:
        func()
