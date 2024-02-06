import os
import glob
import json
from logging import getLogger

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore

from .launchers.inline.config import InlineConfig
from .launchers.process.config import ProcessConfig
from .launchers.torchrun.config import TorchrunConfig

from .backends.openvino.config import OVConfig
from .backends.pytorch.config import PyTorchConfig
from .backends.onnxruntime.config import ORTConfig
from .backends.torch_ort.config import TorchORTConfig
from .backends.tensorrt_llm.config import TRTLLMConfig
from .backends.neural_compressor.config import INCConfig
from .backends.text_generation_inference.config import TGIConfig

from .experiment import launch, ExperimentConfig
from .benchmarks.training.config import TrainingConfig
from .benchmarks.inference.config import InferenceConfig


LOGGER = getLogger("cli")

# Register configurations
cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)
# backends configurations
cs.store(group="backend", name=OVConfig.name, node=OVConfig)
cs.store(group="backend", name=PyTorchConfig.name, node=PyTorchConfig)
cs.store(group="backend", name=ORTConfig.name, node=ORTConfig)
cs.store(group="backend", name=TorchORTConfig.name, node=TorchORTConfig)
cs.store(group="backend", name=TRTLLMConfig.name, node=TRTLLMConfig)
cs.store(group="backend", name=INCConfig.name, node=INCConfig)
cs.store(group="backend", name=TGIConfig.name, node=TGIConfig)
# benchmarks configurations
cs.store(group="benchmark", name=TrainingConfig.name, node=TrainingConfig)
cs.store(group="benchmark", name=InferenceConfig.name, node=InferenceConfig)
# launchers configurations
cs.store(group="launcher", name=InlineConfig.name, node=InlineConfig)
cs.store(group="launcher", name=ProcessConfig.name, node=ProcessConfig)
cs.store(group="launcher", name=TorchrunConfig.name, node=TorchrunConfig)


# optimum-benchmark
@hydra.main(version_base=None)
def benchmark_cli(experiment_config: DictConfig) -> None:
    if glob.glob("*.csv") and os.environ.get("OVERRIDE_BENCHMARKS", "0") != "1":
        LOGGER.warning(
            "Skipping benchmark because results already exist. "
            "Set OVERRIDE_BENCHMARKS=1 to override benchmark results."
        )
        return

    # fix backend until deprecated model and device are removed
    if experiment_config.task is not None:
        LOGGER.warning("`task` is deprecated in experiment. Use `backend.task` instead.")
        experiment_config.backend.task = experiment_config.task
    if experiment_config.model is not None:
        LOGGER.warning("`model` is deprecated in experiment. Use `backend.model` instead.")
        experiment_config.backend.model = experiment_config.model
    if experiment_config.device is not None:
        LOGGER.warning("`device` is deprecated in experiment. Use `backend.device` instead.")
        experiment_config.backend.device = experiment_config.device
    if experiment_config.library is not None:
        LOGGER.warning("`library` is deprecated in experiment. Use `backend.library` instead.")
        experiment_config.backend.library = experiment_config.library

    # Instantiate the experiment configuration and trigger its __post_init__
    experiment_config: ExperimentConfig = OmegaConf.to_object(experiment_config)
    OmegaConf.save(experiment_config, "experiment_config.yaml", resolve=True)

    benchmark_report = launch(experiment_config=experiment_config)

    json.dump(benchmark_report, open("benchmark_report.json", "w"), indent=4)
