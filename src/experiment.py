from dataclasses import dataclass, MISSING
from logging import getLogger

from time import time
from platform import python_version
from transformers import __version__ as transformers_version
from optimum.version import __version__ as optimum_version

from src.backend.base import BackendConfig
from src.benchmark.inference import BenchmarkConfig


LOGGER = getLogger('experiment')


@dataclass
class ExperimentConfig:
    # MODEL CONFIGURATION
    # Name of the model to run (bert-base-uncased, ...)
    model: str = MISSING # type: ignore
    # Task on which the model is run (sequence-classification, ...)
    task: str = MISSING # type: ignore
    # Device on which the model is loaded and run (cpu, cuda, ...)
    device: str = MISSING # type: ignore

    # BACKEND CONFIGURATION
    # The backend to use for recording timing (pytorch, onnxruntime, ...)
    backend: BackendConfig = MISSING # type: ignore

    # BENCHMARK CONFIGURATION
    # The kind of benchmark to run (inference, training, ...)
    benchmark: BenchmarkConfig = MISSING # type: ignore

    # EXPERIMENT CONFIGURATION
    # Experiment name
    experiment_name: str = MISSING # type: ignore
    # Experiment identifier (timestamp)
    experiment_id: int = int(time())

    # ENVIRONMENT CONFIGURATION
    # Python interpreter version
    python_version: str = python_version()
    # Store the transformers version used during the benchmark
    transformers_version: str = transformers_version
    # # # Store the optimum version used during the benchmark
    optimum_version: str = optimum_version
