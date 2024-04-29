import glob
import os
from logging import getLogger

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .backends.llm_swarm.config import LLMSwarmConfig
from .backends.neural_compressor.config import INCConfig
from .backends.onnxruntime.config import ORTConfig
from .backends.openvino.config import OVConfig
from .backends.py_txi.config import PyTXIConfig
from .backends.pytorch.config import PyTorchConfig
from .backends.tensorrt_llm.config import TRTLLMConfig
from .backends.torch_ort.config import TorchORTConfig
from .benchmark import BenchmarkConfig, launch
from .launchers.inline.config import InlineConfig
from .launchers.process.config import ProcessConfig
from .launchers.torchrun.config import TorchrunConfig
from .report import BenchmarkReport
from .scenarios.energy_star.config import EnergyStarConfig
from .scenarios.inference.config import InferenceConfig
from .scenarios.training.config import TrainingConfig

LOGGER = getLogger("cli")

# Register configurations
cs = ConfigStore.instance()
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


# optimum-benchmark
@hydra.main(version_base=None)
def benchmark_cli(benchmark_config: DictConfig) -> None:
    os.environ["BENCHMARK_INTERFACE"] = "CLI"

    if glob.glob("benchmark_report.json") and os.environ.get("OVERRIDE_BENCHMARKS", "0") != "1":
        LOGGER.warning(
            "Benchmark report already exists. If you want to override it, set the environment variable OVERRIDE_BENCHMARKS=1"
        )
        return

    # Instantiate the experiment configuration and trigger its __post_init__
    benchmark_config: BenchmarkConfig = OmegaConf.to_object(benchmark_config)
    benchmark_config.save_json("benchmark_config.json")

    benchmark_report: BenchmarkReport = launch(benchmark_config=benchmark_config)
    benchmark_report.save_json("benchmark_report.json")
