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
from src.benchmark.base import Benchmark, BenchmarkConfig
from src.tracker.memory import MemoryTracker
from src.tracker.latency import LatencyTracker

BENCHMARK_NAME = "inference"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = BENCHMARK_NAME

    memory: bool = False
    profile: bool = False

    warmup_runs: int = 10
    benchmark_duration: int = 10

    batch_size: int = 1
    new_tokens: int = 100


class InferenceBenchmark(Benchmark):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

        self.model_peak_memory: int = 0
        self.model_latencies: List[float] = []
        self.generation_latencies: List[float] = []

        self.profiling_records: List[Tuple[str, str, float]] = []

    def configure(self, config: InferenceConfig):
        self.memory = config.memory
        self.profile = config.profile
        self.generation = False

        self.warmup_runs = config.warmup_runs
        self.benchmark_duration = config.benchmark_duration

        self.batch_size = config.batch_size
        self.new_tokens = config.new_tokens

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running inference")

        self._run_with_forward_latency_tracking(backend)

        if self.memory:
            self._run_with_memory_tracking(backend)

        if backend.is_generator():
            self.generation = True
            self._run_with_generate_latency_tracking(backend)

        if self.profile:
            self._run_with_model_profile(backend)

    def _run_with_forward_latency_tracking(self, backend: Backend) -> None:
        forward_inputs = self.generate_dummy_inputs(mode="forward")

        LOGGER.info("\t+ Warming up the forward pass")
        for _ in range(self.warmup_runs):
            outputs = backend.forward(forward_inputs)

        LOGGER.info("\t+ Tracking model latency and throughput")
        latency_tracker = LatencyTracker(device=self.device)
        while sum(latency_tracker.get_latencies()) < self.benchmark_duration:
            with latency_tracker.track():
                outputs = backend.forward(forward_inputs)

        self.model_latencies = latency_tracker.get_latencies()
        LOGGER.info(f"\t+ Model Latency: {self.model_latency:.2e} (s)")
        LOGGER.info(f"\t+ Model Throughput: {self.model_throughput:.2f} (iter/s)")

    def _run_with_generate_latency_tracking(self, backend: Backend) -> None:
        generate_inputs = self.generate_dummy_inputs(mode="generate")

        LOGGER.info("\t+ Warming up the generation pass")
        for _ in range(1):
            outputs = backend.generate(generate_inputs, new_tokens=self.new_tokens)

        LOGGER.info("\t+ Tracking generation throughput")
        latency_tracker = LatencyTracker(device=self.device)
        while sum(latency_tracker.get_latencies()) < self.benchmark_duration:
            with latency_tracker.track():
                outputs = backend.generate(generate_inputs, new_tokens=self.new_tokens)

        self.generation_latencies = latency_tracker.get_latencies()
        LOGGER.info(
            f"\t+ Generation Throughput: {self.generation_throughput:.2f} (tok/s)"
        )

    def _run_with_memory_tracking(self, backend: Backend) -> None:
        memory_inputs = self.generate_dummy_inputs(mode="forward")

        LOGGER.info("\t+ Tracking model peak memory")
        memory_tracker = MemoryTracker(device=self.device)
        with memory_tracker.track(interval=self.model_latency / 10):
            outputs = backend.forward(memory_inputs)

        self.model_peak_memory = memory_tracker.get_peak_memory()
        LOGGER.info(f"\t+ Model Peak Memory: {self.model_peak_memory} (MB)")

    def _run_with_model_profile(self, backend: Backend) -> None:
        LOGGER.info("Preparing for profiling")
        profile_inputs = self.generate_dummy_inputs(mode="forward")
        backend.prepare_for_profiling(list(profile_inputs.keys()))
        LOGGER.info("Running profiling")
        backend.forward(profile_inputs)
        self.profiling_records = backend.pretrained_model.get_profiling_records()  # type: ignore

    @property
    def model_latency(self) -> float:
        return statistics.mean(self.model_latencies)

    @property
    def model_throughput(self) -> float:
        return len(self.model_latencies) / sum(self.model_latencies)

    @property
    def generation_throughput(self) -> float:
        return (
            self.new_tokens
            * len(self.generation_latencies)
            / sum(self.generation_latencies)
        )

    @property
    def results_df(self) -> DataFrame:
        results_dict = dict()

        results_dict["forward.latency(s)"] = significant_figures(self.model_latency)
        results_dict["forward.throughput(iter/s)"] = significant_figures(
            self.model_throughput
        )

        if self.memory:
            results_dict["forward.peak_memory(MB)"] = significant_figures(
                self.model_peak_memory
            )

        if self.generation:
            results_dict["generate.throughput(tok/s)"] = significant_figures(
                self.generation_throughput
            )

        return DataFrame(results_dict, index=[0])

    @property
    def profile_df(self) -> DataFrame:
        return DataFrame(
            self.profiling_records,
            columns=["Node/Kernel", "Operator", "Latency (s)"],
        )

    @property
    def objective(self) -> float:
        return self.model_latency

    def save(self) -> None:
        LOGGER.info("Saving inference results")
        self.results_df.to_csv("inference_results.csv")

        if self.profile:
            LOGGER.info("Saving profiling results")
            self.profile_df.to_csv("inference_profile.csv")

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

        dummy_input = dict()
        for input_name in input_names:
            generator = None
            for generator_class in generator_classes:
                if input_name in generator_class.SUPPORTED_INPUT_NAMES:
                    if (
                        input_name == "input_ids"
                        and mode == "generate"
                        and self.batch_size > 1
                    ):
                        LOGGER.warn(
                            f"\t+ batch_size={self.batch_size} is not "
                            "recommended for LLM throughput benchmarking."
                        )

                    generator = generator_class(
                        task=self.task,
                        normalized_config=normalized_config,
                        batch_size=self.batch_size,
                    )

            if generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            dummy_input[input_name] = generator.generate(input_name).to(self.device)

            # this is for bettertransformer since it does not support random attention mask
            if input_name == "attention_mask":
                dummy_input["attention_mask"] = torch.ones_like(
                    dummy_input["input_ids"]
                )

        return dummy_input


def significant_figures(x):
    return float(f"{x:.3g}")
