from typing import Optional, Type
from logging import getLogger

import hydra
from omegaconf import OmegaConf
from hydra.utils import get_class
from optimum.exporters import TasksManager
from hydra.core.config_store import ConfigStore

from src.experiment import ExperimentConfig

from src.benchmark.base import Benchmark
from src.benchmark.inference import InferenceConfig

from src.backend.base import Backend
from src.backend.pytorch import PyTorchConfig
from src.backend.onnxruntime import ORTConfig

# Register resolvers (maybe should be moved to a separate file)
OmegaConf.register_new_resolver(
    "clean_string", lambda model: model.split("/")[-1].replace("-", "_")
)
OmegaConf.register_new_resolver("onnxruntime_version", lambda: ORTConfig.version)
OmegaConf.register_new_resolver(
    "is_profiling", lambda benchmark_name: benchmark_name == "profiling"
)
OmegaConf.register_new_resolver(
    "is_inference", lambda benchmark_name: benchmark_name == "inference"
)
OmegaConf.register_new_resolver("is_gpu", lambda device: device in ["cuda", "gpu"])
OmegaConf.register_new_resolver("infer_task", TasksManager.infer_task_from_model)
OmegaConf.register_new_resolver(
    "infer_provider",
    lambda device: f"{device.upper()}ExecutionProvider",
)

# Register configurations (maybe should be moved to a separate file)
cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)
cs.store(group="backend", name="pytorch", node=PyTorchConfig)
cs.store(group="backend", name="onnxruntime", node=ORTConfig)
cs.store(group="benchmark", name="inference", node=InferenceConfig)

LOGGER = getLogger("optimum-benchmark")


@hydra.main(config_path="configs", version_base=None)
def run_experiment(config: ExperimentConfig) -> Optional[float]:
    # Save the config
    OmegaConf.save(config, "hydra_config.yaml", resolve=True)
    # Allocate requested benchmark
    benchmark_factory: Type[Benchmark] = get_class(config.benchmark._target_)  # type: ignore
    benchmark: Benchmark = benchmark_factory(config.model, config.task, config.device)
    benchmark.configure(config.benchmark)

    try:
        # Allocate requested backend
        backend_factory: Type[Backend] = get_class(config.backend._target_)  # type: ignore
        backend: Backend = backend_factory(config.model, config.task, config.device)
        backend.configure(config.backend)

        # Run the benchmark
        benchmark.run(backend)
        # clean backend
        backend.clean()
        # Save the benchmark results
        benchmark.save()

    # log error and traceback
    except Exception as e:
        LOGGER.error(e, exc_info=True)


if __name__ == "__main__":
    run_experiment()
