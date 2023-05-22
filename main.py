from typing import Optional, Type
from logging import getLogger

import hydra
from omegaconf import OmegaConf
from hydra.utils import get_class
from hydra.core.config_store import ConfigStore
from optimum.exporters import TasksManager

from src.experiment import ExperimentConfig
from src.benchmark.base import Benchmark
from src.backend.base import Backend

from src.backend.pytorch import PyTorchConfig
from src.backend.onnxruntime import ORTConfig


# Register resolvers
OmegaConf.register_new_resolver(
    "extract_model_name", lambda model: model.split("/")[-1].replace("-", "_"))

OmegaConf.register_new_resolver(
    "onnxruntime_version", lambda: ORTConfig.version)
OmegaConf.register_new_resolver(
    "pytorch_version", lambda: PyTorchConfig.version)

OmegaConf.register_new_resolver(
    "is_profiling", lambda benchmark_name: benchmark_name == "profiling")
OmegaConf.register_new_resolver(
    "is_inference", lambda benchmark_name: benchmark_name == "inference")
OmegaConf.register_new_resolver(
    "is_gpu", lambda device: device in ["cuda", "gpu"])

OmegaConf.register_new_resolver(
    "infer_task", TasksManager.infer_task_from_model)
OmegaConf.register_new_resolver(
    "infer_provider", lambda device: f"{device.upper()}ExecutionProvider")

# Register configurations
cs = ConfigStore.instance()

# This is the default config that comes with
# an identifier and some environment variables
cs.store(name="experiment", node=ExperimentConfig)

LOGGER = getLogger(__name__)


@hydra.main(config_path="configs", config_name="bert", version_base=None)
def run_experiment(config: ExperimentConfig) -> Optional[float]:
    # Save the config
    OmegaConf.save(config, "config.yaml", resolve=True)

    # Allocate requested target backend
    backend_factory: Type[Backend] = get_class(config.backend._target_) # type: ignore
    backend: Backend = backend_factory(config.model, config.task, config.device)
    backend.configure(config.backend)

    # Allocate requested benchmark
    benchmark_factory: Type[Benchmark] = get_class(config.benchmark._target_) # type: ignore
    benchmark: Benchmark = benchmark_factory(config.model, config.task, config.device)
    benchmark.configure(config.benchmark)

    # Run the benchmark
    benchmark.run(backend)
    # Save the benchmark results
    benchmark.save()

    # get the optuna metric that will be minimized in a sweep
    if config.benchmark.name == "inference":
        return benchmark.results["Model latency mean (s)"][0]
    else:
        return None


if __name__ == "__main__":
    run_experiment()
