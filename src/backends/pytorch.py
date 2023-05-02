from typing import Set, Tuple
from dataclasses import dataclass
from logging import getLogger

from tqdm import trange

import torch
from transformers import AutoModel, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

from src.backends.base import Backend
from src.backends.config import BackendConfig

from src.benchmark.base import Benchmark
from src.benchmark.config import BenchmarkConfig

from src.utils import INPUT_GENERATORS

BACKEND_NAME = "pytorch"

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME

    bettertransformer: bool = False
    torch_compile: bool = False
    no_grad: bool = True
    device: str = "cpu"

    @staticmethod
    def version() -> str:
        return torch.__version__

    @staticmethod
    def supported_keys() -> Set[str]:
        return BackendConfig.supported_keys().union(
            {'bettertransformer', 'torch_compile', 'no_grad', 'device'}
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
        if backend_config.torch_compile:
            LOGGER.info("\t+ Using compiled Module")
            self.pretrained_model = torch.compile(self.pretrained_model)

    def execute(self, config: BenchmarkConfig) -> Tuple[Benchmark, torch.Tensor]:
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
            outputs.append(output[-1])

        # Run benchmark
        while sum(benchmark.latencies) < config.benchmark_duration:
            with benchmark.track(device=config.backend.device):
                self.pretrained_model(
                    **dummy_inputs,
                )

        benchmark.finalize(config.benchmark_duration)

        return benchmark, torch.stack(outputs)
