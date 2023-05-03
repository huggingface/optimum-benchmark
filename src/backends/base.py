from typing import Generic, TypeVar, ClassVar, Tuple, Dict
from logging import getLogger
from psutil import cpu_count
from abc import ABC

import torch
from tqdm import trange
from transformers import AutoTokenizer
from optimum.exporters import TasksManager

from src.benchmark.base import Benchmark
from src.backends.config import BackendConfig
from src.benchmark.config import BenchmarkConfig
from src.utils import INPUT_GENERATORS

LOGGER = getLogger('backends')
BackendConfigT = TypeVar('BackendConfigT', bound=BackendConfig)


class Backend(Generic[BackendConfigT], ABC):
    NAME: ClassVar[str]

    def __init__(self, model: str):
        self.model = model
        self.task = TasksManager.infer_task_from_model(self.model)

    @classmethod
    def allocate(cls, config: BenchmarkConfig) -> 'Backend':
        backend = cls(config.model)
        backend.configure(config.backend)
        return backend

    def configure(self, config: BackendConfigT) -> None:
        if config.inter_op_num_threads is not None:
            if config.inter_op_num_threads == -1:
                config.inter_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.inter_op_num_threads to {config.inter_op_num_threads}")

        if config.intra_op_num_threads is not None:
            if config.intra_op_num_threads == -1:
                config.intra_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.intra_op_num_threads to {config.intra_op_num_threads}")

    def execute(self, config: BenchmarkConfig) -> Tuple[Benchmark, torch.Tensor]:
        LOGGER.info(f"Running {self.NAME} benchmark")
        benchmark = Benchmark()
        dummy_inputs = self.get_dummy_inputs(config)

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

        # Compute statistics
        benchmark.finalize(config.benchmark_duration)
        return benchmark, torch.stack(outputs)

    def get_dummy_inputs(self, config: BenchmarkConfig) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of dummy inputs for the given benchmark configuration.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        input_names = tokenizer.model_input_names

        LOGGER.info(f"Generating dummy inputs for {input_names}")
        dummy_inputs = {
            input_name: INPUT_GENERATORS[input_name](config)
            for input_name in input_names
        }

        return dummy_inputs
