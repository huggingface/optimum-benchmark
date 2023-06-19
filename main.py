from typing import Optional, Type
from logging import getLogger

import hydra
from omegaconf import OmegaConf
from hydra.utils import get_class
from hydra.core.config_store import ConfigStore

from src.experiment import ExperimentConfig

from src.benchmark.base import Benchmark
from src.benchmark.inference import InferenceConfig

from src.backend.base import Backend
from src.backend.pytorch import PyTorchConfig
from src.backend.onnxruntime import ORTConfig

# Register configurations (maybe should be moved to a separate file)
cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)
cs.store(group="backend", name="pytorch", node=PyTorchConfig)
cs.store(group="backend", name="onnxruntime", node=ORTConfig)
cs.store(group="benchmark", name="inference", node=InferenceConfig)

LOGGER = getLogger("main")


@hydra.main(config_path="configs", version_base=None)
def run_experiment(experiment: ExperimentConfig) -> Optional[float]:
    # Save the config
    OmegaConf.save(experiment, "hydra_config.yaml", resolve=True)

    # Allocate requested benchmark
    benchmark_factory: Type[Benchmark] = get_class(experiment.benchmark._target_)
    benchmark: Benchmark = benchmark_factory(
        experiment.model,
        experiment.task,
        experiment.device,
        experiment.model_kwargs,
    )
    benchmark.configure(experiment.benchmark)

    try:
        # Allocate requested backend
        backend_factory: Type[Backend] = get_class(experiment.backend._target_)
        backend: Backend = backend_factory(
            experiment.model,
            experiment.task,
            experiment.device,
            experiment.model_kwargs,
        )
        backend.configure(experiment.backend)

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
