import torch
from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torch.distributed.elastic.multiprocessing import Std
from dataclasses import dataclass, field, MISSING
from logging import getLogger
import logging.config

from .base import TrainingBenchmark, TrainingConfig
from transformers import Trainer, TrainingArguments, PreTrainedModel

from omegaconf import OmegaConf
from typing import TYPE_CHECKING, Dict, Any, Optional
import os
import gc

if TYPE_CHECKING:
    from optimum_benchmark.backends.base import Backend

LOGGER = getLogger("training-pytorch")

OmegaConf.register_new_resolver("device_count", lambda: torch.cuda.device_count())

# Copied from https://github.com/pytorch/pytorch/blob/v2.0.0/torch/distributed/launcher/api.py#L29, adjusting to the defaults of torch.distributed.run
@dataclass
class PyTorchDDPLaunchConfig(LaunchConfig):
    min_nodes: int = 1
    max_nodes: int = 1
    nproc_per_node: int = "${device_count:}"
    run_id: str ="none"
    role: str = "default"
    rdzv_endpoint: str = "127.0.0.1:29500"
    rdzv_backend: str = "static"
    rdzv_configs: Dict[str, Any] = field(default_factory=lambda: {"timeout": 900, "rank": 0})
    max_restarts: int = 0
    monitor_interval: float = 5
    start_method: str = "spawn"
    log_dir: Optional[str] = None
    redirects: Std = Std.NONE
    tee: Std = Std.NONE
    metrics_cfg: Dict[str, str] = field(default_factory=dict)
    local_addr: Optional[str] = None

@dataclass
class PyTorchTrainingConfig(TrainingConfig):
    name: str = "training-pytorch"
    _target_: str = "optimum_benchmark.benchmarks.training.pytorch.PyTorchTrainingBenchmark"

    use_ddp: bool = True
    ddp_config: PyTorchDDPLaunchConfig = field(default_factory=PyTorchDDPLaunchConfig)

def get_logger(name: Optional[str] = None, log_all: bool = False):
    if os.environ["RANK"] == "0" or log_all:
        # TODO: also configure logging for other ranks
        hydra_conf = OmegaConf.load('.hydra/hydra.yaml')
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return getLogger(name)

def ddp_callable(args):
    pretrained_model = args[0]
    training_dataset = args[1]
    training_arguments = args[2]
    use_ddp = args[3]

    if use_ddp:
        LOGGER_WORKER = get_logger("training-pytorch-worker", log_all=False)

        env_variables = [
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_MAX_RESTARTS",
        ]
        for env_var in env_variables:
            LOGGER_WORKER.info(f"{env_var}: {os.environ.get(env_var)}")
    else:
        LOGGER_WORKER = LOGGER

    LOGGER_WORKER.info("\t+ Wrapping model with transformers.Trainer")
    training_arguments = TrainingArguments(**training_arguments)

    trainer = Trainer(
        model=pretrained_model,
        train_dataset=training_dataset,
        args=training_arguments,
    )
    
    LOGGER_WORKER.info("Training model")
    results = trainer.train().metrics

    result = {"train_samples_per_second": results["train_samples_per_second"], "train_runtime": results["train_runtime"]}
    return result


class PyTorchTrainingBenchmark(TrainingBenchmark):
    def run(self, backend: "Backend") -> None:
        LOGGER.info("Running training benchmark")

        # Converting from DictConfig to Dict is required to avoid a warning with DDP:
        # `[W CudaIPCTypes.cpp:15] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]`
        training_arguments_dict = OmegaConf.to_container(self.training_arguments, resolve=True)

        self.generate_dataset(backend)
        
        if self.config.use_ddp:
            # TODO: support multi-node training. Hydra is probably not the good infra for that though.
            if self.config.ddp_config.max_nodes != 1:
                raise ValueError("PyTorch DDP training benchmark currently supports only training on a single node.")
                        
            # TODO: The backend instance can not be passed here (cannot pickle 'weakref' object) so the nn.Module is passed directly.
            # It is not clear who is using weakref though.
            results = elastic_launch(
                config=self.config.ddp_config,
                entrypoint=ddp_callable,
            )((backend.pretrained_model, self.training_dataset, training_arguments_dict, True))
            
            # For DDP, we log only the stats from the first rank as transformers does. It could make sense to log for all ranks.
            self.training_throughput = results[0]["train_samples_per_second"]
            self.training_runtime = results[0]["train_runtime"]
        else:
            results = ddp_callable((backend.pretrained_model, self.training_dataset, training_arguments_dict, False))
        
            self.training_throughput = results["train_samples_per_second"]
            self.training_runtime = results["train_runtime"]
