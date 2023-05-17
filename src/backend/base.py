from typing import ClassVar, Dict, Optional
from dataclasses import dataclass, MISSING
from abc import abstractmethod, ABC
from logging import getLogger
from psutil import cpu_count

import time
import torch
import statistics
from torch import Tensor
from pandas import DataFrame
from transformers import PreTrainedModel

LOGGER = getLogger('backend')


@dataclass
class BackendConfig(ABC):
    name: str = MISSING
    version: str = MISSING

    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None


class Backend(ABC):
    NAME: ClassVar[str]

    # every backend will have a pretrained model
    pretrained_model: Optional[PreTrainedModel] = None

    def __init__(self, model: str, task: str, device: str) -> None:
        self.model = model
        self.task = task
        self.device = device

    def configure(self, config: BackendConfig) -> None:
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

    def run_inference(self, inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int) -> DataFrame:
        LOGGER.info("Warming up model")
        for _ in range(warmup_runs):
            self.inference_latency(inputs)

        LOGGER.info("Tracking inference latency")
        latencies = []
        while sum(latencies) < benchmark_duration:
            latency = self.inference_latency(inputs)
            latencies.append(latency)

        inference_results = DataFrame({
            "Model latency mean (s)": statistics.mean(latencies),
            "Model latency std (s)": statistics.stdev(latencies),
            "Model Throughput (s^-1)": len(latencies) / benchmark_duration
        }, index=[0])

        return inference_results

    # Inference helper methods
    def inference_latency(self, dummy_inputs: Dict[str, Tensor]) -> float:
        if self.device == 'cuda':
            return self._cuda_inference_latency(dummy_inputs)
        else:
            return self._cpu_inference_latency(dummy_inputs)

    def _cuda_inference_latency(self, dummy_inputs: Dict[str, Tensor]) -> float:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        self.pretrained_model(**dummy_inputs)
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        latency = latency_ms / 1e3
        LOGGER.debug(f'Tracked CUDA latency took: {latency}s)')

        return latency

    def _cpu_inference_latency(self, dummy_inputs: Dict[str, Tensor]) -> float:
        start = time.perf_counter_ns()
        self.pretrained_model(**dummy_inputs)
        end = time.perf_counter_ns()
        latency_ns = end - start
        latency = latency_ns / 1e9
        LOGGER.debug(f'Tracked CPU latency took: {latency}s)')

        return latency

    @abstractmethod
    def run_profiling(self, inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int) -> DataFrame:
        raise NotImplementedError(
            "Backend must implement run_profiling method")
