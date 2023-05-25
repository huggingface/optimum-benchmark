from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List

import gc
import torch
import statistics
from pandas import DataFrame
from torch import Tensor, __version__
from transformers import AutoConfig
from optimum.exporters import TasksManager
from transformers.utils.fx import symbolic_trace
from optimum.bettertransformer import BetterTransformer

from src.backend.base import Backend, BackendConfig
from src.symbolic_profiler import SymbolicProfiler
from src.tracker import Tracker

BACKEND_NAME = "pytorch"
LOGGER = getLogger(BACKEND_NAME)


@dataclass
class PyTorchConfig(BackendConfig):
    name: str = BACKEND_NAME
    version: str = __version__

    # inference options
    disable_grad: bool = False
    eval_mode: bool = False

    # graph optimization options
    fp16: bool = False
    bettertransformer: bool = False
    torch_compile: bool = False


class PyTorchBackend(Backend):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)
        self.model_type = AutoConfig.from_pretrained(self.model).model_type
        self.automodel_class = TasksManager.get_model_class_for_task(
            task=self.task, model_type=self.model_type
        )

    def configure(self, config: PyTorchConfig) -> None:
        LOGGER.info("Configuring pytorch Backend:")
        super().configure(config)

        # Torch specific environment variables
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
        LOGGER.info(f"\t+ Loading {self.model} with {self.automodel_class.__name__}")
        self.pretrained_model = self.automodel_class.from_pretrained(self.model)

        # Move model to device
        if self.pretrained_model.device.type != self.device:
            LOGGER.info(f"\t+ Moving model to device {self.device}")
            self.pretrained_model.to(self.device)

        # Turn on eval mode
        if config.eval_mode:
            LOGGER.info("\t+ Turning on eval mode")
            self.pretrained_model.eval()

        # Turn on better transformer inference
        if config.bettertransformer:
            LOGGER.info("\t+ Using optimum.bettertransformer")
            self.pretrained_model = BetterTransformer.transform(  # type: ignore
                self.pretrained_model, keep_original_model=False
            )

        # Compile model
        if config.torch_compile:
            LOGGER.info("\t+ Using torch.compile")
            self.pretrained_model.forward = torch.compile(self.pretrained_model.forward)

        # Turn on fp16
        if config.fp16:
            LOGGER.info("\t+ Turning on fp16")
            self.fp16 = True
        else:
            self.fp16 = False

    def run_inference(
        self, dummy_inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int
    ) -> List[float]:
        LOGGER.info("Running backend inference")

        LOGGER.info("\t+ Warming up the model")
        for _ in range(warmup_runs):
            with torch.cuda.amp.autocast(enabled=self.fp16):  # type: ignore
                self.pretrained_model(**dummy_inputs)

        LOGGER.info("\t+ Tracking inference latency")
        tracker = Tracker(device=self.device)
        while sum(tracker.tracked_latencies) < benchmark_duration:
            with tracker.track_latency():
                with torch.cuda.amp.autocast(enabled=self.fp16):  # type: ignore
                    self.pretrained_model(**dummy_inputs)

        return tracker.tracked_latencies

    def run_profiling(
        self, dummy_inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int
    ) -> DataFrame:
        LOGGER.info("Running backend profiling")

        LOGGER.info("\t+ Symbolic tracing model")
        self.pretrained_model = symbolic_trace(
            model=self.pretrained_model,  # type: ignore
            input_names=list(dummy_inputs.keys()),
        )

        LOGGER.info("\t+ Warming up symbolic model")
        for _ in range(warmup_runs):
            with torch.cuda.amp.autocast(enabled=self.fp16):  # type: ignore
                self.pretrained_model(**dummy_inputs)

        LOGGER.info("\t+ Creating symbolic profiler")
        symbolic_profiler = SymbolicProfiler(self.pretrained_model)

        LOGGER.info("\t+ Profiling symbolic model")
        tracker = Tracker(device=self.device)
        while sum(tracker.tracked_latencies) < benchmark_duration:
            with tracker.track_latency():
                with torch.cuda.amp.autocast(enabled=self.fp16):  # type: ignore
                    symbolic_profiler.run(*dummy_inputs.values())

        nodes_latencies = symbolic_profiler.nodes_latencies

        LOGGER.info("\t+ Calculating profiling results")
        profiling_results = DataFrame(
            [
                {
                    "Node name": str(node),
                    "Op name": str(node.op),
                    "Node latency mean (s)": statistics.mean(node_latency),
                    "Node latency median (s)": statistics.median(node_latency),
                    "Node latency stdev (s)": statistics.stdev(node_latency),
                }
                for node, node_latency in nodes_latencies.items()
            ]
        )

        return profiling_results

    def clean(self) -> None:
        del self.pretrained_model
        gc.collect()
