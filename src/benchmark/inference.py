from typing import List, Tuple, Dict
from dataclasses import dataclass
from logging import getLogger

import torch
import statistics
from torch import Tensor
from pandas import DataFrame
from transformers import AutoConfig
from optimum.exporters import TasksManager
from transformers.onnx.utils import get_preprocessor

from src.backend.base import Backend
from src.tracker.memory import MemoryTracker
from src.tracker.latency import LatencyTracker
from src.benchmark.base import Benchmark, BenchmarkConfig

BENCHMARK_NAME = "inference"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = BENCHMARK_NAME
    _target_: str = "src.benchmark.inference.InferenceBenchmark"

    # run options
    memory: bool = False
    profile: bool = False

    # loop options
    warmup_runs: int = 10
    benchmark_duration: int = 10

    # input options
    batch_size: int = 1
    # output options
    new_tokens: int = 100


class InferenceBenchmark(Benchmark):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

        self.forward_peak_memory: int = 0
        self.forward_latencies: List[float] = []
        self.generate_latencies: List[float] = []
        self.forward_profile: List[Tuple[str, str, float]] = []  # kernel/node, op, time

    def configure(self, config: InferenceConfig):
        super().configure(config)
        self.memory = config.memory
        self.profile = config.profile

        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

        self.batch_size = config.batch_size
        self.new_tokens = config.new_tokens

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running inference benchmark")

        if self.memory:
            # let's start with memory tracking (no warmup)
            self._run_memory_tracking(backend)

        self._run_forward_tracking(backend)

        self._run_generate_tracking(backend)

        if self.profile:
            self._run_forward_profile(backend)

    def _run_memory_tracking(self, backend: Backend) -> None:
        memory_inputs = self.generate_dummy_inputs(mode="forward")

        LOGGER.info("\t+ Tracking forward pass peak memory")
        memory_tracker = MemoryTracker(device=self.device)
        with memory_tracker.track(interval=self.forward_latency / 10):
            outputs = backend.forward(memory_inputs)

        self.forward_peak_memory = memory_tracker.get_peak_memory()
        LOGGER.info(f"\t+ Forward pass peak memory: {self.forward_peak_memory} (MB)")

    def _run_forward_tracking(self, backend: Backend) -> None:
        forward_inputs = self.generate_dummy_inputs(mode="forward")

        LOGGER.info("\t+ Warming up the forward pass")
        for _ in range(self.warmup_runs):
            outputs = backend.forward(forward_inputs)

        LOGGER.info("\t+ Tracking forward pass latency and throughput")
        latency_tracker = LatencyTracker(device=self.device)
        while sum(latency_tracker.get_latencies()) < self.benchmark_duration:
            with latency_tracker.track():
                outputs = backend.forward(forward_inputs)

        self.forward_latencies = latency_tracker.get_latencies()
        LOGGER.info(f"\t+ Forward pass latency: {self.forward_latency:.2e} (s)")
        LOGGER.info(
            f"\t+ Forward pass throughput: {self.forward_throughput:.2f} (samples/s)"
        )

    def _run_generate_tracking(self, backend: Backend) -> None:
        generate_inputs = self.generate_dummy_inputs(mode="generate")

        LOGGER.info("\t+ Testing and warming up the generation pass")
        try:
            outputs = backend.generate(generate_inputs, new_tokens=self.new_tokens)
        except:
            LOGGER.info("\t+ Generation pass failed or not supported")
            self.can_generate = False
            return
        else:
            self.can_generate = True

        LOGGER.info("\t+ Tracking generation throughput")
        latency_tracker = LatencyTracker(device=self.device)
        while sum(latency_tracker.get_latencies()) < self.benchmark_duration:
            with latency_tracker.track():
                outputs = backend.generate(generate_inputs, new_tokens=self.new_tokens)

        self.generate_latencies = latency_tracker.get_latencies()
        LOGGER.info(f"\t+ Generation pass latency: {self.generate_latency:.2e} (s)")

        LOGGER.info(
            f"\t+ Generation pass throughput: {self.generate_throughput:.2f} (tokens/s)"
        )

    def _run_forward_profile(self, backend: Backend) -> None:
        profile_inputs = self.generate_dummy_inputs(mode="forward")

        LOGGER.info("\t+ Preparing backend for profiling")
        backend.prepare_for_profiling(list(profile_inputs.keys()))

        LOGGER.info("\t+ Running profiling")
        backend.forward(profile_inputs)

        self.forward_profile = backend.pretrained_model.get_forward_profile()  # type: ignore

    # Metrics
    @property
    def forward_latency(self) -> float:
        return statistics.mean(self.forward_latencies)

    @property
    def forward_throughput(self) -> float:
        return self.batch_size / self.forward_latency

    @property
    def generate_latency(self) -> float:
        return statistics.mean(self.generate_latencies)

    @property
    def generate_throughput(self) -> float:
        return self.new_tokens * self.batch_size / self.generate_latency

    def get_results_df(self) -> DataFrame:
        results_dict = dict()

        results_dict["forward.latency(s)"] = significant_figures(self.forward_latency)
        results_dict["forward.throughput(samples/s)"] = significant_figures(
            self.forward_throughput
        )

        if self.memory:
            results_dict["forward.peak_memory(MB)"] = significant_figures(
                self.forward_peak_memory
            )

        if self.can_generate:
            results_dict["generate.latency(s)"] = significant_figures(
                self.generate_latency
            )
            results_dict["generate.throughput(tokens/s)"] = significant_figures(
                self.generate_throughput
            )

        return DataFrame(results_dict, index=[0])

    def get_profile_df(self) -> DataFrame:
        return DataFrame(
            self.forward_profile,
            columns=["Node/Kernel", "Operator", "Latency (s)"],
        )

    def save(self) -> None:
        LOGGER.info("Saving inference results")
        results_df = self.get_results_df()
        results_df.to_csv("inference_results.csv")

        if self.profile:
            LOGGER.info("Saving profiling results")
            profile_df = self.get_profile_df()
            profile_df.to_csv("inference_profile.csv")

    def generate_dummy_inputs(self, mode) -> Dict[str, Tensor]:
        # hacky way to get what we need
        auto_config = AutoConfig.from_pretrained(self.model)
        onnx_config = TasksManager._SUPPORTED_MODEL_TYPE[auto_config.model_type]["onnx"][self.task](auto_config)  # type: ignore
        normalized_config = onnx_config.NORMALIZED_CONFIG_CLASS(auto_config)
        generator_classes = onnx_config.DUMMY_INPUT_GENERATOR_CLASSES

        if mode == "forward":
            input_names = list(onnx_config.inputs.keys())
        elif mode == "generate":
            input_names = get_preprocessor(self.model).model_input_names
        else:
            raise ValueError(f"Unknown mode {mode}")

        dummy_inputs = dict()
        for input_name in input_names:
            generator = None
            for generator_class in generator_classes:
                if input_name in generator_class.SUPPORTED_INPUT_NAMES:
                    generator = generator_class(
                        task=self.task,
                        normalized_config=normalized_config,
                        batch_size=self.batch_size,
                    )

            if generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            dummy_inputs[input_name] = generator.generate(input_name).to(self.device)

            if input_name == "attention_mask":
                # this is for bettertransformer (for now)
                # since it only supports right padded attention mask
                dummy_inputs["attention_mask"] = torch.ones_like(
                    dummy_inputs["input_ids"]
                )

        return dummy_inputs


def significant_figures(x):
    return float(f"{x:.3g}")
