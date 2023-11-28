import os
import platform
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Type

from hydra.core.config_store import ConfigStore
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from .backends.neural_compressor.config import INCConfig
from .backends.onnxruntime.config import ORTConfig
from .backends.openvino.config import OVConfig
from .backends.pytorch.config import PyTorchConfig
from .backends.text_generation_inference.config import TGIConfig
from .benchmarks.inference.config import InferenceConfig
from .benchmarks.training.config import TrainingConfig
from .env_utils import get_cpu, get_cpu_ram_mb, get_git_revision_hash, get_gpus
from .import_utils import (
    accelerate_version,
    diffusers_version,
    optimum_version,
    transformers_version,
)
from .launchers.inline.config import InlineConfig
from .launchers.process.config import ProcessConfig
from .launchers.torchrun.config import TorchrunConfig
from .task_utils import infer_task_from_model_name_or_path

if TYPE_CHECKING:
    from .backends.base import Backend
    from .benchmarks.base import Benchmark
    from .launchers.base import Launcher


LOGGER = getLogger("experiment")

OmegaConf.register_new_resolver("infer_task", lambda model: infer_task_from_model_name_or_path(model))


@dataclass
class ExperimentConfig:
    # LAUNCHER CONFIGURATION
    launcher: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # BACKEND CONFIGURATION
    backend: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # BENCHMARK CONFIGURATION
    benchmark: Any  # https://github.com/facebookresearch/hydra/issues/1722#issuecomment-883568386

    # EXPERIMENT CONFIGURATION
    experiment_name: str = "experiment"
    # Model name or path (bert-base-uncased, google/vit-base-patch16-224, ...)
    model: str = "bert-base-uncased"
    # Task name (text-classification, image-classification, ...)
    task: str = "${infer_task:${model}}"
    # Device name or path (cpu, cuda, cuda:0, ...)
    device: str = "cuda"

    # ADDITIONAL MODEL CONFIGURATION: Model revision, use_auth_token, trust_remote_code
    hub_kwargs: Dict = field(
        default_factory=lambda: {
            # "token": None,
            "revision": "main",
            "cache_dir": None,
            "force_download": False,
            "local_files_only": False,
        }
    )

    # ENVIRONMENT CONFIGURATION
    environment: Dict = field(
        default_factory=lambda: {
            "optimum_version": optimum_version(),
            "optimum_commit": get_git_revision_hash(os.environ.get("OPTIMUM_PATH")),
            "transformers_version": transformers_version(),
            "transformers_commit": get_git_revision_hash(os.environ.get("TRANSFORMERS_PATH")),
            "accelerate_version": accelerate_version(),
            "accelerate_commit": get_git_revision_hash(os.environ.get("ACCELERATE_PATH")),
            "diffusers_version": diffusers_version(),
            "diffusers_commit": get_git_revision_hash(os.environ.get("DIFFUSERS_PATH")),
            "python_version": platform.python_version(),
            "system": platform.system(),
            "cpu": get_cpu(),
            "cpu_count": os.cpu_count(),
            "cpu_ram_mb": get_cpu_ram_mb(),
            "gpus": get_gpus(),
        }
    )

    def __post_init__(self) -> None:
        if self.device.startswith("cuda:"):
            raise ValueError(
                f"Device was specified as {self.device} with a target index."
                "We recommend using the main cuda device (`cuda`) and specifying the target index in `CUDA_VISIBLE_DEVICES`."
            )

        if self.device not in ["cuda", "cpu", "mps", "xla"]:
            raise ValueError("`device` must be either `cuda`, `cpu`, `mps` or `xla`.")

        if "cuda" in self.device and len(self.environment["gpus"]) > 1:
            if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
                LOGGER.warning(
                    "Multiple GPUs detected but CUDA_VISIBLE_DEVICES is not set. "
                    "This means that code might allocate resources from the wrong GPUs. "
                    "We recommend setting CUDA_VISIBLE_DEVICES to isolate the GPUs that will be used for this experiment. "
                    "`CUDA_VISIBLE_DEVICES` will be set to `0` to ensure that only the first GPU is used."
                    "If you want to use multiple GPUs, please set `CUDA_VISIBLE_DEVICES` to the desired GPU indices."
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            if os.environ.get("CUDA_DEVICE_ORDER", None) != "PCI_BUS_ID":
                LOGGER.warning(
                    "Multiple GPUs detected but CUDA_DEVICE_ORDER is not set to `PCI_BUS_ID`. "
                    "This means that code might allocate resources from the wrong GPUs even if `CUDA_VISIBLE_DEVICES` is set. "
                    "For example pytorch uses the `FASTEST_FIRST` order by default, which is not guaranteed to be the same as nvidia-smi. "
                    "`CUDA_DEVICE_ORDER` will be set to `PCI_BUS_ID` to ensure that the GPUs are allocated in the same order as nvidia-smi. "
                )
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# Register configurations
cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)
#
cs.store(group="backend", name="openvino", node=OVConfig)
cs.store(group="backend", name="pytorch", node=PyTorchConfig)
cs.store(group="backend", name="onnxruntime", node=ORTConfig)
cs.store(group="backend", name="neural-compressor", node=INCConfig)
cs.store(group="backend", name="text-generation-inference", node=TGIConfig)
#
cs.store(group="benchmark", name="inference", node=InferenceConfig)
cs.store(group="benchmark", name="training", node=TrainingConfig)
#
cs.store(group="launcher", name="inline", node=InlineConfig)
cs.store(group="launcher", name="process", node=ProcessConfig)
cs.store(group="launcher", name="torchrun", node=TorchrunConfig)


def run(experiment: "ExperimentConfig") -> "Benchmark":
    # Instantiate the experiment config to trigger __post_init__
    experiment: ExperimentConfig = OmegaConf.to_object(experiment)
    OmegaConf.save(experiment, "hydra_config.yaml", resolve=True)

    # Allocate requested backend
    backend_factory: Type["Backend"] = get_class(experiment.backend._target_)
    backend: "Backend" = backend_factory(
        task=experiment.task,
        model=experiment.model,
        device=experiment.device,
        hub_kwargs=experiment.hub_kwargs,
    )

    try:
        # Configure the backend
        backend.configure(experiment.backend)
    except Exception as e:
        LOGGER.error("Error during backend configuration: %s", e)
        backend.clean()
        raise e

    # Allocate requested benchmark
    benchmark_factory: Type["Benchmark"] = get_class(experiment.benchmark._target_)
    benchmark: "Benchmark" = benchmark_factory()

    try:
        # Configure the benchmark
        benchmark.configure(experiment.benchmark)
    except Exception as e:
        LOGGER.error("Error during benchmark configuration: %s", e)
        backend.clean()
        raise e

    # Run the benchmark
    try:
        benchmark.run(backend)
        benchmark.save()
        backend.clean()
    except Exception as e:
        LOGGER.error("Error during benchmark execution: %s", e)
        backend.clean()
        raise e


def run_with_launcher(experiment: DictConfig):
    # instead of emplimenting hydra/launcher plugins, we handle the launcher ourselves
    # this allows us to use spawn with torchrun, to gather outputs from parallel processes,
    # and to handle errors gracefully

    # Instantiate the experiment config to trigger __post_init__
    experiment.launcher = OmegaConf.to_object(experiment.launcher)

    launcher_factory: Type["Launcher"] = get_class(experiment.launcher._target_)
    launcher: "Launcher" = launcher_factory()

    try:
        launcher.configure(experiment.launcher)
    except Exception as e:
        LOGGER.error("Error during launcher configuration: %s", e)
        raise e

    try:
        launcher.launch(run, experiment)
    except Exception as e:
        LOGGER.error("Error during experiment execution: %s", e)
        raise e
