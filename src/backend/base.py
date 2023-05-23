from dataclasses import dataclass, MISSING
from abc import abstractmethod, ABC
from typing import Dict, Optional
from logging import getLogger
from psutil import cpu_count

import time
import torch
from torch import Tensor
from pandas import DataFrame

LOGGER = getLogger("backend")


@dataclass
class BackendConfig(ABC):
    name: str = MISSING  # type: ignore
    version: str = MISSING  # type: ignore

    inter_op_num_threads: Optional[int] = None
    intra_op_num_threads: Optional[int] = None


class Backend(ABC):
    def __init__(self, model: str, task: str, device: str) -> None:
        self.model = model
        self.task = task
        self.device = device

    @abstractmethod
    def configure(self, config: BackendConfig) -> None:
        # generic configuration, can be moved to a resolver
        if config.inter_op_num_threads is not None:
            if config.inter_op_num_threads == -1:
                config.inter_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.inter_op_num_threads to cpu_count({config.inter_op_num_threads})"
                )

        if config.intra_op_num_threads is not None:
            if config.intra_op_num_threads == -1:
                config.intra_op_num_threads = cpu_count()
                LOGGER.info(
                    f"\t+ Setting backend.intra_op_num_threads to cpu_count({config.intra_op_num_threads})"
                )

    @abstractmethod
    def clean(self) -> None:
        raise NotImplementedError("Backend must implement clean method")

    @abstractmethod
    def run_inference(
        self, dummy_inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int
    ) -> DataFrame:
        raise NotImplementedError("Backend must implement run_inference method")

    @abstractmethod
    def run_profiling(
        self, dummy_inputs: Dict[str, Tensor], warmup_runs: int, benchmark_duration: int
    ) -> DataFrame:
        raise NotImplementedError("Backend must implement run_profiling method")

    # Inference helper methods
    def track_inference_latency(self, dummy_inputs: Dict[str, Tensor]) -> float:
        if self.device == "cuda":
            return self._cuda_inference_latency(dummy_inputs)
        else:
            return self._cpu_inference_latency(dummy_inputs)

    def _cuda_inference_latency(self, dummy_inputs: Dict[str, Tensor]) -> float:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record(stream=torch.cuda.current_stream())
        self.pretrained_model(**dummy_inputs)  # type: ignore
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
        latency = latency_ms / 1e3
        LOGGER.debug(f"Tracked CUDA latency took: {latency}s)")

        return latency

    def _cpu_inference_latency(self, dummy_inputs: Dict[str, Tensor]) -> float:
        start = time.perf_counter_ns()
        self.pretrained_model(**dummy_inputs)  # type: ignore
        end = time.perf_counter_ns()
        latency_ns = end - start
        latency = latency_ns / 1e9
        LOGGER.debug(f"Tracked CPU latency took: {latency}s)")

        return latency
