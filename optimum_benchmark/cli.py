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
    ExperimentConfig,
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
    VLLMConfig,
    launch,
)
from .logging_utils import setup_logging

LOGGER = getLogger("hydra-cli")


class DeprecatedTrainingConfig(TrainingConfig):
    def __post_init__(self):
        if os.environ.get("BENCHMARK_INTERFACE", "API") == "CLI":
            LOGGER.warning(
                "The `benchmark: training` in your defaults list is deprecated. Please use `scenario: training` instead."
            )

        super().__post_init__()


class DeprecatedInferenceConfig(InferenceConfig):
    def __post_init__(self):
        if os.environ.get("BENCHMARK_INTERFACE", "API") == "CLI":
            LOGGER.warning(
                "The `benchmark: inference` in your defaults list is deprecated. Please use `scenario: inference` instead."
            )

        super().__post_init__()


class DeprecatedEnergyStarConfig(EnergyStarConfig):
    def __post_init__(self):
        if os.environ.get("BENCHMARK_INTERFACE", "API") == "CLI":
            LOGGER.warning(
                "The `benchmark: energy_star` in your defaults list is deprecated. Please use `scenario: energy_star` instead."
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
cs.store(group="backend", name=VLLMConfig.name, node=VLLMConfig)
# scenarios configurations
cs.store(group="scenario", name=TrainingConfig.name, node=TrainingConfig)
cs.store(group="scenario", name=InferenceConfig.name, node=InferenceConfig)
cs.store(group="scenario", name=EnergyStarConfig.name, node=EnergyStarConfig)
# launchers configurations
cs.store(group="launcher", name=InlineConfig.name, node=InlineConfig)
cs.store(group="launcher", name=ProcessConfig.name, node=ProcessConfig)
cs.store(group="launcher", name=TorchrunConfig.name, node=TorchrunConfig)
# deprecated
cs.store(name="experiment", node=ExperimentConfig)
cs.store(group="benchmark", name=TrainingConfig.name, node=DeprecatedTrainingConfig)
cs.store(group="benchmark", name=InferenceConfig.name, node=DeprecatedInferenceConfig)
cs.store(group="benchmark", name=EnergyStarConfig.name, node=DeprecatedEnergyStarConfig)

LOGGING_SETUP_DONE = False


def setup_logging_once(*args, **kwargs):
    global LOGGING_SETUP_DONE
    if not LOGGING_SETUP_DONE:
        LOGGING_SETUP_DONE = True
        setup_logging(*args, **kwargs)


# optimum-benchmark
@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    global LOGGING_SETUP_DONE

    # TODO: remove when experiment schema is removed
    os.environ["BENCHMARK_INTERFACE"] = "CLI"

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    override_benchmarks = os.environ.get("OVERRIDE_BENCHMARKS", "0") == "1"

    setup_logging_once(level=log_level, to_file=log_to_file, prefix="MAIN-PROCESS")

    if glob.glob("benchmark_report.json") and not override_benchmarks:
        LOGGER.warning(
            "Benchmark was already conducted in the current directory. "
            "If you want to override it, set the environment variable OVERRIDE_BENCHMARKS=1 (in hydra.job.env_set)"
        )
        return

    # Instantiates the configuration with the right class and triggers its __post_init__
    config = OmegaConf.to_object(config)

    if isinstance(config, ExperimentConfig):
        # old api
        experiment_config = config
        experiment_config.save_json("experiment_config.json")
        benchmark_report: BenchmarkReport = launch(experiment_config=experiment_config)
        benchmark_report.save_json("benchmark_report.json")

    elif isinstance(config, BenchmarkConfig):
        # new api
        benchmark_config = config
        benchmark_config.save_json("benchmark_config.json")
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_report.save_json("benchmark_report.json")
        benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
        benchmark.save_json("benchmark.json")
