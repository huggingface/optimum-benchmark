from logging import getLogger
from typing import Type
from json import dumps

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

from src.input.base import InputGenerator
from src.input.text import TextConfig
from src.input.audio import AudioConfig

from optimum.exporters import TasksManager

# Register resolvers
OmegaConf.register_new_resolver(
    "for_gpu", lambda device: device == 'cuda')
OmegaConf.register_new_resolver(
    "infer_task", TasksManager.infer_task_from_model)
OmegaConf.register_new_resolver(
    'infer_provider', lambda device: f'{device.upper()}ExecutionProvider')
OmegaConf.register_new_resolver(
    'is_inference', lambda benchmark_name: benchmark_name == 'inference')
OmegaConf.register_new_resolver(
    'pytorch_version', lambda: PyTorchConfig.version)
OmegaConf.register_new_resolver(
    'onnxruntime_version', lambda: ORTConfig.version)

# Register configurations
cs = ConfigStore.instance()

# This is the default config that comes with
# an identifier and some environment variables
cs.store(name="experiment", node=ExperimentConfig)
cs.store(group="benchmark", name="inference_benchmark", node=InferenceConfig)

cs.store(group="backend", name="pytorch_backend", node=PyTorchConfig)
cs.store(group="backend", name="onnxruntime_backend", node=ORTConfig)

cs.store(group="input", name="text_input", node=TextConfig)
cs.store(group="input", name="audio_input", node=AudioConfig)


LOGGER = getLogger(__name__)


@hydra.main(config_path="configs", config_name="base_experiment", version_base=None)
def run_experiment(config: ExperimentConfig) -> None:
    """
    Run the benchmark with the given configuration.
    """

    # Allocate requested target backend
    backend_factory: Type[Backend] = get_class(config.backend._target_)
    backend: Backend = backend_factory(
        config.model, config.task, config.device)
    backend.configure(config.backend)

    # Allocate requested input generator
    input_factory: Type[InputGenerator] = get_class(config.input._target_)
    input_generator: InputGenerator = input_factory(
        config.model, config.task, config.device)
    input_generator.configure(config.input)

    # Allocate requested benchmark
    benchmark_factory: Type[Benchmark] = get_class(config.benchmark._target_)
    benchmark: Benchmark = benchmark_factory(
        config.model, config.task, config.device)
    benchmark.configure(config.benchmark)

    # Execute the benchmark
    benchmark.execute(backend, input_generator)

    # Save the resolved config
    OmegaConf.save(config, ".hydra/config.yaml", resolve=True)
    # Save the benchmark results
    with open("stats.json", "w") as f:
        f.write(dumps(benchmark.stats_dict, indent=4))
    benchmark.details_df.to_csv("details.csv", index=False)

    return benchmark.stats_dict['latency.mean']


if __name__ == '__main__':
    run_experiment()
