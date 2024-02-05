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
cs.store(group="backend", name="neural-compressor", node=INCConfig)
cs.store(group="backend", name="text-generation-inference", node=TGIConfig)
# benchmarks configurations
cs.store(group="benchmark", name="training", node=TrainingConfig)
cs.store(group="benchmark", name="inference", node=InferenceConfig)
# launchers configurations
cs.store(group="launcher", name="inline", node=InlineConfig)
cs.store(group="launcher", name="process", node=ProcessConfig)
cs.store(group="launcher", name="torchrun", node=TorchrunConfig)


# optimum-benchmark
@hydra.main(version_base=None)
def benchmark_cli(experiment_config: DictConfig) -> None:
    if glob.glob("*.csv") and os.environ.get("OVERRIDE_BENCHMARKS", "0") != "1":
        LOGGER.warning(
            "Skipping benchmark because results already exist. "
            "Set OVERRIDE_BENCHMARKS=1 to override benchmark results."
        )
        return

    # Instantiate the experiment configuration and trigger its __post_init__
    experiment_config: ExperimentConfig = OmegaConf.to_object(experiment_config)
    OmegaConf.save(experiment_config, "experiment_config.yaml", resolve=True)

    # launch the experiment
    output = launch(experiment_config=experiment_config)

    # save the benchmark report
    json.dump(output, open("benchmark_report.json", "w"), indent=4)
