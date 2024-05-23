import glob
import os
from logging import getLogger

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from . import (
    Benchmark,
    BenchmarkConfig,
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
    VLLMConfig,
)
from .logging_utils import setup_logging

LOGGER = getLogger("hydra-cli")


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


# optimum-benchmark
@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_to_file = os.environ.get("LOG_TO_FILE", "1") == "1"
    override_benchmarks = os.environ.get("OVERRIDE_BENCHMARKS", "0") == "1"
    setup_logging(level=log_level, to_file=log_to_file, prefix="MAIN-PROCESS")

    if glob.glob("benchmark_report.json") and not override_benchmarks:
        LOGGER.warning(
            "Benchmark was already conducted in the current directory. "
            "If you want to override it, set the environment variable OVERRIDE_BENCHMARKS=1 (in hydra.job.env_set)"
        )
        return

    # Instantiates the configuration with the right class and triggers its __post_init__
    benchmark_config: BenchmarkConfig = OmegaConf.to_object(config)
    benchmark_config.save_json("benchmark_config.json")

    benchmark_report = Benchmark.launch(benchmark_config)
    benchmark_report.save_json("benchmark_report.json")

    benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
    benchmark.save_json("benchmark.json")
