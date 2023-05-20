import hydra
from logging import getLogger
from omegaconf import OmegaConf
from hydra.utils import get_class
from typing import Optional, Type
from hydra.core.config_store import ConfigStore

from optimum.exporters import TasksManager

from src.experiment import ExperimentConfig
from src.benchmark.base import Benchmark

from src.backend.base import Backend
from src.backend.pytorch import PyTorchConfig
from src.backend.onnxruntime import ORTConfig


# Register resolvers
OmegaConf.register_new_resolver(
    "is_profiling", lambda benchmark_name: benchmark_name == 'profiling')
OmegaConf.register_new_resolver(
    'is_inference', lambda benchmark_name: benchmark_name == 'inference')

OmegaConf.register_new_resolver(
    "is_cuda", lambda device: device == 'cuda')
OmegaConf.register_new_resolver(
    "infer_task", TasksManager.infer_task_from_model)
OmegaConf.register_new_resolver(
    'infer_provider', lambda device: f'{device.upper()}ExecutionProvider')

OmegaConf.register_new_resolver(
    'pytorch_version', lambda: PyTorchConfig.version)
OmegaConf.register_new_resolver(
    'onnxruntime_version', lambda: ORTConfig.version)

# Register configurations
cs = ConfigStore.instance()

# This is the default config that comes with
# an identifier and some environment variables
cs.store(name="experiment", node=ExperimentConfig)

LOGGER = getLogger(__name__)


@hydra.main(config_path="configs", config_name="base_experiment", version_base=None)
def run_experiment(config: ExperimentConfig) -> Optional[float]:
    """
    Run the benchmark with the given configuration.
    """

    # Allocate requested target backend
    backend_factory: Type[Backend] = get_class(config.backend._target_)
    backend: Backend = backend_factory(
        config.model, config.task, config.device)
    backend.configure(config.backend)

    # Allocate requested benchmark
    benchmark_factory: Type[Benchmark] = get_class(config.benchmark._target_)
    benchmark: Benchmark = benchmark_factory(
        config.model, config.task, config.device)
    benchmark.configure(config.benchmark)

    # Run the benchmark
    benchmark.run(backend)

    # Save the benchmark results
    benchmark.save_results()

    # Save the resolved config
    OmegaConf.save(config, ".hydra/config.yaml", resolve=True)

    # get the optuna metric that will be minimized in a sweep
    if config.benchmark.name == 'inference':
        print(benchmark.inference_results['Model latency mean (s)'][0])
        return benchmark.inference_results['Model latency mean (s)'][0]
    else:
        return None


if __name__ == '__main__':
    run_experiment()
