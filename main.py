from omegaconf import OmegaConf
from logging import getLogger
from typing import Type
from json import dumps

import hydra
from hydra.utils import get_class
from hydra.core.config_store import ConfigStore

from src.backend.base import Backend
from src.backend.pytorch import PyTorchConfig, PyTorchOptimizationConfig
from src.backend.onnxruntime import ORTConfig, ORTOptimizationConfig
from src.benchmark.base import Benchmark
from src.benchmark.config import BenchmarkConfig

# Register resolvers
OmegaConf.register_new_resolver("pytorch_version", PyTorchConfig.version)
OmegaConf.register_new_resolver("onnxruntime_version", ORTConfig.version)

# Register configurations
cs = ConfigStore.instance()

# This is the default config that comes with
# an identifier and some environment variables
cs.store(name="benchmark", node=BenchmarkConfig)
cs.store(group="backend", name="pytorch_backend", node=PyTorchConfig)
cs.store(group="backend", name="onnxruntime_backend", node=ORTConfig)
cs.store(group="optimization", name="ort_optimization",
         node=ORTOptimizationConfig)
cs.store(group="optimization", name="pytorch_optimization",
         node=PyTorchOptimizationConfig)

LOGGER = getLogger(__name__)


@hydra.main(config_path="configs", config_name="default_benchmark", version_base=None)
def run(config: BenchmarkConfig) -> None:
    """
    Run the benchmark with the given configuration.
    """
    try:
        # Allocate requested target backend
        backend_factory: Type[Backend] = get_class(config.backend._target_)
        backend: Backend = backend_factory.allocate(config)
        # Run benchmark and reference
        benchmark, benchmark_outputs = backend.execute(config)
    except Exception as e:
        LOGGER.error(f"Exception occurred while running benchmark: {e}")
        benchmark = Benchmark()
    finally:
        # Complete stats
        benchmark.finalize(config.benchmark_duration)
        # Save the resolved config
        OmegaConf.save(config, ".hydra/config.yaml", resolve=True)
        # Save the benchmark results
        with open("stats.json", "w") as f:
            f.write(dumps(benchmark.stats_dict, indent=4))
        benchmark.details_df.to_csv("details.csv", index=False)


if __name__ == '__main__':
    run()
