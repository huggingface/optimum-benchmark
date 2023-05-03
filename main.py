from omegaconf import OmegaConf
from logging import getLogger
from typing import Type
from json import dumps

import hydra
from hydra.utils import get_class
from hydra.core.config_store import ConfigStore

from src.backends.base import Backend
from src.backends.pytorch import PyTorchConfig
from src.backends.onnxruntime import ORTConfig
from src.benchmark.config import BenchmarkConfig

# Register resolvers
OmegaConf.register_new_resolver("pytorch_version", PyTorchConfig.version)
OmegaConf.register_new_resolver("onnxruntime_version", ORTConfig.version)

# Register configurations
cs = ConfigStore.instance()
cs.store(name="base_benchmark", node=BenchmarkConfig)
cs.store(group="backends", name="pytorch_backend", node=PyTorchConfig)
cs.store(group="backends", name="onnxruntime_backend", node=ORTConfig)

LOGGER = getLogger(__name__)

@hydra.main(config_path="configs", config_name="benchmark", version_base=None)
def run(config: BenchmarkConfig) -> None:
    """
    Run the benchmark with the given configuration.
    """
    # Allocate requested target backend
    backend_factory: Type[Backend] = get_class(config.backend._target_)
    backend: Backend = backend_factory.allocate(config)

    # Run benchmark and reference
    benchmark, benchmark_outputs = backend.execute(config)

    # Save the resolved config
    OmegaConf.save(config, ".hydra/config.yaml", resolve=True)

    # Save the benchmark results
    stats_dict = benchmark.stats_dict
    with open("stats.json", "w") as f:
        f.write(dumps(stats_dict, indent=4))
    
    details_df = benchmark.details_df
    details_df.to_csv("details.csv", index=False)


if __name__ == '__main__':
    run()
