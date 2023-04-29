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

import torch
import numpy as np
from optimum.bettertransformer import BetterTransformer

from backend.base import Backend
from backend.config import BackendConfig

from benchmark.base import Benchmark
from benchmark.config import BenchmarkConfig

from utils import TASK_TO_AUTOMODEL

BACKEND_NAME = "pytorch"

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME

    use_besttransformer: bool = False
    use_compile: bool = False
    device: str = "cpu"

    @staticmethod
    def version() -> str:
        return torch.__version__

    @staticmethod
    def supported_keys() -> Set[str]:
        return BackendConfig.supported_keys().union({'use_bettertransformer', 'use_compile', 'device'})


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME = BACKEND_NAME

    def __init__(self, model: str, task: str):
        super().__init__(model, task)
        
        self.pretrained_model = TASK_TO_AUTOMODEL[task].from_pretrained(model)

        LOGGER.info(
            f"Allocated PyTorch Backend for model: {model} on task: {task}")

    @classmethod
    def allocate(cls, config: BenchmarkConfig):
        backend = cls(config.model, config.task)
        backend.configure(config.backend)

        return backend

    def configure(self, config: PyTorchConfig):
        # super().configure(config)

        LOGGER.info("Configuring PyTorch Benchmark:")

        # Move model to device
        self.pretrained_model.to(config.device)
        LOGGER.info(f"\t+ Moved Module to device {config.device}")

        # Disable gradients
        torch.set_grad_enabled(False)
        LOGGER.info("\t+ Disabled gradients")

        # Turn on eval mode
        self.pretrained_model.eval()
        LOGGER.info("\t+ Turning eval mode on Module")

        # Turn on better transformer inference
        if config.use_bettertransformer:
            LOGGER.info("\t+ Using BetterTransformer")
            self.pretrained_model = BetterTransformer.transform(
                self.pretrained_model, keep_original_model=False)

        # Compile model
        if config.use_compile:
            LOGGER.info("\t+ Using compiled Module")
            self.pretrained_model = torch.compile(self.pretrained_model)

    def get_dummy_inputs(self, config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
        if self.task == "sequence-classification":
            input_ids = torch.randint(
                low=0,
                high=self.pretrained_model.config.vocab_size,
                size=(config.batch_size, config.sequence_length),
                dtype=torch.long,
                device=config.backend.device,
            )

            attention_mask = torch.ones(
                config.batch_size,
                config.sequence_length,
                dtype=torch.long,
                device=config.backend.device,
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    def execute(self, config: BenchmarkConfig) -> Tuple[Benchmark, np.ndarray]:
        LOGGER.info("Running PyTorch benchmark")
        benchmark = Benchmark()

        dummy_inputs = self.get_dummy_inputs(config=config)

        # Warmup
        outputs = []
        for _ in trange(config.warmup_runs, desc="Warming up"):
            output = self.pretrained_model(
                **dummy_inputs, output_hidden_states=True)
            outputs.append(output.hidden_states[-1].cpu().numpy())

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
