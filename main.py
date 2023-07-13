import hydra
from logging import getLogger
from omegaconf import OmegaConf
from hydra.utils import get_class
from typing import Optional, Type
from hydra.core.config_store import ConfigStore

from src.experiment import ExperimentConfig
from src.benchmark.base import Benchmark
from src.benchmark.inference import InferenceConfig
from src.backend.base import Backend
from src.backend.pytorch import PyTorchConfig
from src.backend.onnxruntime import ORTConfig
from src.backend.openvino import OVConfig

# Register configurations (maybe should be moved to a separate file)
cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)
cs.store(group="backend", name="pytorch", node=PyTorchConfig)
cs.store(group="backend", name="onnxruntime", node=ORTConfig)
cs.store(group="backend", name="openvino", node=OVConfig)

cs.store(group="benchmark", name="inference", node=InferenceConfig)

LOGGER = getLogger("main")


@hydra.main(config_path="configs", version_base=None)
def run_experiment(experiment: ExperimentConfig) -> Optional[float]:
    # Save the config
    OmegaConf.save(experiment, "hydra_config.yaml", resolve=True)

    # Allocate requested benchmark
    benchmark_factory: Type[Benchmark] = get_class(experiment.benchmark._target_)
    benchmark: Benchmark = benchmark_factory()
    benchmark.configure(experiment.benchmark)

    # Allocate requested backend
    backend_factory: Type[Backend] = get_class(experiment.backend._target_)
    backend: Backend = backend_factory(
        experiment.model,
        experiment.device,
        experiment.cache_kwargs,
    )

    try:
        backend.configure(experiment.backend)
        # Run the benchmark
        benchmark.run(backend)
        # Save the benchmark results
        benchmark.save()
    # log error and traceback
    except Exception as e:
        LOGGER.error(e, exc_info=True)

    # clean backend
    backend.clean()


if __name__ == "__main__":
    run_experiment()
