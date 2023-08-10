from torch.distributed.launcher.api import elastic_launch
from torch.distributed.elastic.multiprocessing import Std
from dataclasses import dataclass, field


from .base import TrainingBenchmark, TrainingConfig

from typing import TYPE_CHECKING, Dict, Any, Optional, Union

if TYPE_CHECKING:
    from optimum_benchmark.backends.base import Backend

# Copied from https://github.com/pytorch/pytorch/blob/v2.0.0/torch/distributed/launcher/api.py#L29, adjusting to the defaults of torch.distributed.run
@dataclass
class PyTorchDDPLaunchConfig:
    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    run_id: str ="none"
    role = "default"
    rdzv_endpoint: str = "127.0.0.1:29500"
    rdzv_backend = "static"
    rdzv_configs: Dict[str, Any] = field(default_factory=lambda: {"timeout": 900})
    max_restarts: int = 0
    monitor_interval: float = 5
    start_method: str = "spawn"
    log_dir: Optional[str] = None
    redirects: Union[Std, Dict[int, Std]] = Std.NONE
    tee: Union[Std, Dict[int, Std]] = Std.NONE
    local_addr: Optional[str] = None

class PyTorchTrainingConfig(TrainingConfig):
    name: str = "training-pytorch"
    _target_: str = "optimum_benchmark.benchmarks.training.pytorch.PyTorchTrainingBenchmark"

    use_ddp: bool = True
    ddp_config: PyTorchDDPLaunchConfig = field(default_factory=PyTorchDDPLaunchConfig)

def ddp_callable(args):
    backend = args[0]
    training_dataset = args[1]
    backend.prepare_for_training(
        training_dataset=self.training_dataset,
        training_arguments=self.training_arguments,
    )

    results = backend.train().metrics

    result = {"train_samples_per_second": results["train_samples_per_second"], "train_runtime": results["train_runtime"]}
    return result


class PyTorchTrainingBenchmark(TrainingBenchmark):
    def run(self, backend: "Backend") -> None:
        LOGGER.info("Running training benchmark")

        if backend.task == "text-classification":
            self.training_dataset = Dataset.from_dict(
                {
                    "input_ids": torch.randint(
                        0,
                        backend.pretrained_config.vocab_size,
                        (
                            self.dataset_shapes.dataset_size,
                            self.dataset_shapes.sequence_length,
                        ),
                    ),
                    "labels": torch.randint(0, 1, (self.dataset_shapes.dataset_size,)),
                }
            )
            self.training_dataset.set_format(
                type="torch",
                columns=["input_ids", "labels"],
            )
        else:
            raise NotImplementedError(
                f"Training benchmark not implemented for task {backend.task}."
                "Please submit a PR to add support for this task."
            )
        
        if self.config.use_ddp:
            config = LaunchConfig(
                min_nodes=n_nodes,
                max_nodes=n_nodes,
                nproc_per_node=nproc_per_node
            )

            results = elastic_launch(
                config=config,
                entrypoint=ddp_callable,
            )((backend, self.training_dataset))
            
            self.training_throughput = results[0]["train_samples_per_second"]
            self.training_runtime = results[0]["train_runtime"]
        else:
            results = ddp_callable((backend, self.training_dataset))
        
            self.training_throughput = results["train_samples_per_second"]
            self.training_runtime = results["train_runtime"]
