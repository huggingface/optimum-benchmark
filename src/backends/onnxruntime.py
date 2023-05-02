from typing import Set, Tuple
from dataclasses import dataclass
from logging import getLogger

from tqdm import trange

import torch
import onnxruntime
from transformers import AutoTokenizer
from optimum.exporters import TasksManager

from src.backends.base import Backend
from src.backends.config import BackendConfig

from src.benchmark.base import Benchmark
from src.benchmark.config import BenchmarkConfig

from src.utils import INPUT_GENERATORS

BACKEND_NAME = 'onnxruntime'

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class ORTConfig(BackendConfig):
    name: str = BACKEND_NAME

    device: str = "cpu"

    @staticmethod
    def version() -> str:
        return onnxruntime.__version__

    @staticmethod
    def supported_keys() -> Set[str]:
        return BackendConfig.supported_keys().union({'device'})


class ORTBackend(Backend[ORTConfig]):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        super().__init__(model)

        task = TasksManager.infer_task_from_model(model)
        model_class = TasksManager.get_model_class_for_task(task)
        self.pretrained_model = model_class.from_pretrained(model)
        self.input_names = AutoTokenizer.from_pretrained(model).model_input_names

        LOGGER.info(
            f"Allocated PyTorch Backend for model: {model}")

    @classmethod
    def allocate(cls, config: BenchmarkConfig):
        backend = cls(config.model)
        backend.configure(config.backend)

        return backend

    def configure(self, config: ORTConfig):
        LOGGER.info("Configuring OnnxRuntime Benchmark:")

        # Move model to device
        self.pretrained_model.to(config.device)
        LOGGER.info(f"\t+ Moved Module to device {config.device}")

    def execute(self, config: BenchmarkConfig) -> Tuple[Benchmark, torch.Tensor]:
        LOGGER.info("Running OnnxRuntime benchmark")
        benchmark = Benchmark()

        dummy_inputs = {
            input_name: INPUT_GENERATORS[input_name](config)
            for input_name in self.input_names
        }

        # Warmup
        outputs = []
        for _ in trange(config.warmup_runs, desc="Warming up"):
            output = self.pretrained_model(
                **dummy_inputs
            )
            outputs.append(output)

        # Run benchmark
        while sum(benchmark.latencies) < config.benchmark_duration:
            if config.backend.device == "cpu":
                with benchmark.track_cpu_latency():
                    self.pretrained_model(**dummy_inputs)
            elif config.backend.device == "cuda":
                with benchmark.track_cuda_latency():
                    self.pretrained_model(**dummy_inputs)
            else:
                raise ValueError(
                    f"Unsupported device type {config.backend.device}")

        benchmark.finalize(config.benchmark_duration)

        return benchmark, torch.stack(outputs)
