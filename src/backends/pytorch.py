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

from typing import Set, Tuple
from dataclasses import dataclass
from logging import getLogger

from tqdm import trange

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

from backends.base import Backend
from backends.config import BackendConfig

from benchmark.base import Benchmark
from benchmark.config import BenchmarkConfig

from utils import INPUT_GENERATORS

BACKEND_NAME = "pytorch"

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME

    bettertransformer: bool = False
    compile: bool = False
    no_grad: bool = True
    device: str = "cpu"

    @staticmethod
    def version() -> str:
        return torch.__version__

    @staticmethod
    def supported_keys() -> Set[str]:
        return BackendConfig.supported_keys().union(
            {'bettertransformer', 'compile', 'no_grad', 'device'}
        )


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME = BACKEND_NAME

    def __init__(self, model: str):
        super().__init__(model)

        self.pretrained_model = AutoModel.from_pretrained(model)
        self.input_names = AutoTokenizer.from_pretrained(
            model).model_input_names

        LOGGER.info(
            f"Allocated PyTorch Backend for model: {model}")

    @classmethod
    def allocate(cls, benchmark_config: BenchmarkConfig):
        backend = cls(benchmark_config.model)
        backend.configure(benchmark_config.backend)

        return backend

    def configure(self, backend_config: PyTorchConfig):
        LOGGER.info("Configuring PyTorch Benchmark:")

        # Move model to device
        self.pretrained_model.to(backend_config.device)
        LOGGER.info(f"\t+ Moved Module to device {backend_config.device}")

        # Disable gradients
        if backend_config.no_grad:
            torch.set_grad_enabled(False)
            LOGGER.info("\t+ Disabled gradients")

        # Turn on eval mode
        self.pretrained_model.eval()
        LOGGER.info("\t+ Turning eval mode on Module")

        # Turn on better transformer inference
        if backend_config.bettertransformer:
            LOGGER.info("\t+ Using BetterTransformer")
            self.pretrained_model = BetterTransformer.transform(
                self.pretrained_model, keep_original_model=False)

        # Compile model
        if backend_config.compile:
            LOGGER.info("\t+ Using compiled Module")
            self.pretrained_model = torch.compile(self.pretrained_model)

    def execute(self, config: BenchmarkConfig) -> Tuple[Benchmark, np.ndarray]:
        LOGGER.info("Running PyTorch benchmark")
        benchmark = Benchmark()

        dummy_inputs = {
            input_name: INPUT_GENERATORS[input_name](config)
            for input_name in self.input_names
        }

        # Warmup
        outputs = []
        for _ in trange(config.warmup_runs, desc="Warming up"):
            output = self.pretrained_model(
                **dummy_inputs,
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
