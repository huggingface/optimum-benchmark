import glob
import os
from logging import getLogger

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from . import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkReport,
    EnergyStarConfig,
    INCConfig,
    InferenceConfig,
    InlineConfig,
    LLMSwarmConfig,
    ORTConfig,
    OVConfig,
    ProcessConfig,
    PyTorchConfig,
    PyTXIConfig,
    TorchORTConfig,
    TorchrunConfig,
    TrainingConfig,
    TRTLLMConfig,
)
from .experiment import ExperimentConfig, launch  # deprecated
from .logging_utils import setup_logging

LOGGER = getLogger("hydra-cli")


class DeprecatedTrainingConfig(TrainingConfig):
    def __post_init__(self):
        if os.environ.get("BENCHMARK_INTERFACE", "API") == "CLI":
            LOGGER.warning("The `benchmark: training` schema is deprecated. Please use `scenario: training` instead.")

        super().__post_init__()


class DeprecatedInferenceConfig(InferenceConfig):
    def __post_init__(self):
        if os.environ.get("BENCHMARK_INTERFACE", "API") == "CLI":
            LOGGER.warning("The `benchmark: inference` schema is deprecated. Please use `scenario: inference` instead.")

        super().__post_init__()


class DeprecatedEnergyStarConfig(EnergyStarConfig):
    def __post_init__(self):
        if os.environ.get("BENCHMARK_INTERFACE", "API") == "CLI":
            LOGGER.warning(
                "The `benchmark: energy_star` schema is deprecated. Please use `scenario: energy_star` instead."
            )

        super().__post_init__()


# Register configurations
cs = ConfigStore.instance()
# benchmark configuration
cs.store(name="benchmark", node=BenchmarkConfig)
# backends configurations
cs.store(group="backend", name=OVConfig.name, node=OVConfig)
cs.store(group="backend", name=PyTorchConfig.name, node=PyTorchConfig)
cs.store(group="backend", name=ORTConfig.name, node=ORTConfig)
cs.store(group="backend", name=TorchORTConfig.name, node=TorchORTConfig)
cs.store(group="backend", name=TRTLLMConfig.name, node=TRTLLMConfig)
cs.store(group="backend", name=INCConfig.name, node=INCConfig)
cs.store(group="backend", name=PyTXIConfig.name, node=PyTXIConfig)
cs.store(group="backend", name=LLMSwarmConfig.name, node=LLMSwarmConfig)
# scenarios configurations
cs.store(group="scenario", name=TrainingConfig.name, node=TrainingConfig)
cs.store(group="scenario", name=InferenceConfig.name, node=InferenceConfig)
cs.store(group="scenario", name=EnergyStarConfig.name, node=EnergyStarConfig)
# launchers configurations
cs.store(group="launcher", name=InlineConfig.name, node=InlineConfig)
cs.store(group="launcher", name=ProcessConfig.name, node=ProcessConfig)
cs.store(group="launcher", name=TorchrunConfig.name, node=TorchrunConfig)

# deprecated
cs.store(group="benchmark", name=TrainingConfig.name, node=DeprecatedTrainingConfig)
cs.store(group="benchmark", name=InferenceConfig.name, node=DeprecatedInferenceConfig)
cs.store(group="benchmark", name=EnergyStarConfig.name, node=DeprecatedEnergyStarConfig)

LOGGING_SETUP_DONE = False


# optimum-benchmark
@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    os.environ["BENCHMARK_INTERFACE"] = "CLI"

    global LOGGING_SETUP_DONE

    if not LOGGING_SETUP_DONE:
        setup_logging(level="INFO", prefix="MAIN-PROCESS")
        LOGGING_SETUP_DONE = True
    else:
        setup_logging(level="INFO")

    if glob.glob("benchmark_report.json") and os.environ.get("OVERRIDE_BENCHMARKS", "0") != "1":
        LOGGER.warning(
            "Benchmark report already exists. If you want to override it, set the environment variable OVERRIDE_BENCHMARKS=1"
        )
        return

    # Instantiates the configuration with the right class and triggers its __post_init__
    config = OmegaConf.to_object(config)

    if isinstance(config, ExperimentConfig):
        experiment_config = config
        experiment_config.save_json("experiment_config.json")
        benchmark_report: BenchmarkReport = launch(experiment_config=experiment_config)
        benchmark_report.save_json("benchmark_report.json")

    elif isinstance(config, BenchmarkConfig):
        benchmark_config = config
        benchmark_config.save_json("benchmark_config.json")
        benchmark = Benchmark(config=benchmark_config)
        benchmark_report = benchmark.launch()
        benchmark_report.save_json("benchmark_report.json")
        # this one is new
        benchmark.save_json("benchmark.json")
