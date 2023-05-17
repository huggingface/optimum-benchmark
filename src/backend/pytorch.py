from dataclasses import dataclass
from logging import getLogger
import statistics
from typing import Dict, List
from pandas import DataFrame

import torch
from transformers.utils.fx import symbolic_trace
from torch import Tensor, __version__ as torch_version
from optimum.bettertransformer import BetterTransformer
from optimum.exporters import TasksManager

from src.backend.base import Backend, BackendConfig
from src.backend.utils import SymbolicProfiler

BACKEND_NAME = "pytorch"

LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = torch_version

    # inference options
    disable_grad: bool = False
    eval_mode: bool = False

    # graph optimization options
    bettertransformer: bool = False
    torch_compile: bool = False

    symbolic_profiling: bool = False


class PyTorchBackend(Backend):
    NAME = BACKEND_NAME

    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

    def configure(self, config: PyTorchConfig) -> None:
        LOGGER.info("Configuring pytorch Backend:")
        super().configure(config)

        # Torch specific configuration
        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting pytorch inter_op_num_threads({config.inter_op_num_threads}))"
            )
            torch.set_num_threads(config.inter_op_num_threads)

        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting pytorch intra_op_num_threads({config.intra_op_num_threads}))"
            )
            torch.set_num_interop_threads(config.intra_op_num_threads)

        # Disable gradients
        if not config.disable_grad or config.eval_mode:
            LOGGER.info("\t+ Disabling gradients")
            torch.set_grad_enabled(False)

        # Load model
        automodel_class = TasksManager.get_model_class_for_task(self.task)
        LOGGER.info(f"\t+ Loading model {self.model} for task {self.task}")
        self.pretrained_model = automodel_class.from_pretrained(self.model)

        # Move model to device
        if self.pretrained_model.device.type != self.device:
            LOGGER.info(f"\t+ Moving model to device {self.device}")
            self.pretrained_model.to(self.device)

        # Turn on eval mode
        if config.eval_mode:
            LOGGER.info("\t+ Turning eval mode on model")
            self.pretrained_model.eval()

        # Turn on better transformer inference
        if config.bettertransformer:
            LOGGER.info("\t+ Using BetterTransformer Fastpath")
            self.pretrained_model = BetterTransformer.transform(
                self.pretrained_model, keep_original_model=False)

        # Compile model
        if config.torch_compile:
            LOGGER.info("\t+ Using torch.compile")
            self.pretrained_model = torch.compile(self.pretrained_model)

    def run_profiling(self, inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int) -> DataFrame:

        LOGGER.info("Symbolic tracing model")
        self.pretrained_model = symbolic_trace(
            model=self.pretrained_model,
            input_names=list(inputs.keys()),
        )

        LOGGER.info("Warming up symbolic model")
        for _ in range(warmup_runs):
            self.inference_latency(inputs)

        LOGGER.info("Creating symbolic profiler")
        symbolic_profiler = SymbolicProfiler(self.pretrained_model)

        LOGGER.info("Profiling symbolic model")
        while sum(symbolic_profiler.model_latencies) < benchmark_duration:
            symbolic_profiler.run(*inputs.values())

        profiling_results = DataFrame([
            {'Node name': str(node), 'Node Op': str(node.op),
             'Node latency mean (s)': statistics.mean(node_latency),
             'Node latency std (s)': statistics.stdev(node_latency)}
            for node, node_latency in symbolic_profiler.nodes_latencies.items()
        ])

        return profiling_results
