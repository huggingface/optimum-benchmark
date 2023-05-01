#  Copyright 2021 Hugging Face Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Dict, Set, Tuple
from dataclasses import dataclass
from logging import getLogger

from tqdm import trange

import onnxruntime
import numpy as np
from transformers import AutoTokenizer
from optimum.exporters import TasksManager
from optimum.onnxruntime import ORTModel

from backends.base import Backend
from backends.config import BackendConfig

from benchmark.base import Benchmark
from benchmark.config import BenchmarkConfig

from utils import INPUT_GENERATORS

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

    def execute(self, config: BenchmarkConfig) -> Tuple[Benchmark, np.ndarray]:
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

        return benchmark, np.stack(outputs)
