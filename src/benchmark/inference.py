from dataclasses import dataclass
from typing import List, Tuple, Dict
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
from src.trackers.memory import MemoryTracker
from src.trackers.latency import LatencyTracker

BENCHMARK_NAME = "inference"
LOGGER = getLogger(BENCHMARK_NAME)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = BENCHMARK_NAME

    track_memory: bool = False
    profile: bool = False

    warmup_runs: int = 10
    model_runs: int = 100
    generation_runs: int = 10

    batch_size: int = 1


class InferenceBenchmark(Benchmark):
    def __init__(self, model: str, task: str, device: str) -> None:
        super().__init__(model, task, device)

        self.model_peak_memory: int = 0
        self.model_latencies: List[float] = []

        self.generation_num_tokens: List[int] = []
        self.generation_latencies: List[float] = []

        self.profiling_records: List[Tuple[str, str, float]] = []

    def configure(self, config: InferenceConfig):
        self.profile = config.profile
        self.track_memory = config.track_memory

        self.warmup_runs = config.warmup_runs
        self.model_runs = config.model_runs

        self.generation_runs = config.generation_runs

        self.batch_size = config.batch_size

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running inference")

        self._run_with_forward_latency_tracking(backend)

        if self.track_memory:
            self._run_with_memory_tracking(backend)

        if backend.is_generator:
            self.is_generator = True
            self._run_with_generate_latency_tracking(backend)

        if self.profile:
            self._run_with_model_profile(backend)

    def _run_with_forward_latency_tracking(self, backend: Backend) -> None:
        forward_inputs = self.generate_dummy_inputs(mode="forward")

        LOGGER.info("\t+ Warming up the model")
        for _ in range(self.warmup_runs):
            backend.forward(forward_inputs)

        LOGGER.info("\t+ Tracking model latency and throughput")
        latency_tracker = LatencyTracker(device=self.device)
        for _ in range(self.model_runs):
            with latency_tracker.track():
                outputs = backend.forward(forward_inputs)

        self.model_latencies = latency_tracker.get_tracked_latencies()
        LOGGER.info(f"\t+ Model Latency: {self.model_latency:.2e} (s)")
        LOGGER.info(f"\t+ Model Throughput: {self.model_throughput:.2f} (iter/s)")

    def _run_with_generate_latency_tracking(self, backend: Backend) -> None:
        generate_inputs = self.generate_dummy_inputs(mode="generate")
        if "input_ids" in generate_inputs:
            if generate_inputs["input_ids"].shape[0] > 1:
                LOGGER.info(
                    f"\t+ Batch size is {generate_inputs['input_ids'].shape[0]}"
                    " but it's better to use batch size 1 for LLM benchmarking."
                )
            prefix_tokens = (
                generate_inputs["input_ids"].shape[-1]
                * generate_inputs["input_ids"].shape[0]
            )
        else:
            prefix_tokens = 0

        LOGGER.info("\t+ Tracking generation throughput")
        latency_tracker = LatencyTracker(device=self.device)
        for _ in range(self.generation_runs):
            with latency_tracker.track():
                outputs = backend.generate(
                    generate_inputs,
                )

            new_tokens = (outputs.shape[-1] * outputs.shape[0]) - prefix_tokens
            self.generation_num_tokens.append(new_tokens)
            LOGGER.debug(f"\t+ Generated {new_tokens} tokens")

        self.generation_latencies = latency_tracker.get_tracked_latencies()
        LOGGER.info(
            f"\t+ Generation Throughput: {self.generation_throughput:.2f} (tok/s)"
        )

    def _run_with_memory_tracking(self, backend: Backend) -> None:
        memory_inputs = self.generate_dummy_inputs(mode="forward")

        LOGGER.info("\t+ Tracking model peak memory")
        memory_tracker = MemoryTracker(device=self.device)
        with memory_tracker.track(interval=self.model_latency / 10):
            outputs = backend.forward(memory_inputs)

        self.model_peak_memory = memory_tracker.get_tracked_peak_memory()
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
        return sum(self.generation_num_tokens) / sum(self.generation_latencies)

    @property
    def results_df(self) -> DataFrame:
        results_dict = dict()

        results_dict["Model Latency (s)"] = self.model_latency
        results_dict["Model Throughput (iter/s)"] = self.model_throughput

        if self.track_memory:
            results_dict["Model Peak Memory (MB)"] = self.model_peak_memory

        if self.is_generator:
            results_dict["Generation Throughput (tok/s)"] = self.generation_throughput

        return DataFrame(results_dict, index=[0])

    @property
    def profile_df(self) -> DataFrame:
        return DataFrame(
            self.profiling_records,
            columns=["Node/Kernel", "Operator", "Latency (s)"],
        )

    def save(self) -> None:
        LOGGER.info("Saving inference results")
        self.results_df.to_csv("inference_results.csv")

        if self.profile:
            LOGGER.info("Saving profiling results")
            self.profile_df.to_csv("inference_profile.csv")

    @property
    def objective(self) -> float:
        return self.model_latency

    def generate_dummy_inputs(self, mode) -> Dict[str, Tensor]:
        auto_config = AutoConfig.from_pretrained(self.model)
        model_type = auto_config.model_type
        onnx_config = TasksManager._SUPPORTED_MODEL_TYPE[model_type]["onnx"][self.task](
            auto_config
        )
        normalized_config = onnx_config.NORMALIZED_CONFIG_CLASS(auto_config)  # type: ignore

        if mode == "forward":
            input_names = list(onnx_config.inputs.keys())  # type: ignore
        elif mode == "generate":
            input_names = get_preprocessor(self.model).model_input_names  # type: ignore
        else:
            raise ValueError(f"Unknown mode {mode}")

        dummy_input = dict()
        for input_name in input_names:
            dummy_input_generator = None
            for dummy_input_generator_class in onnx_config.DUMMY_INPUT_GENERATOR_CLASSES:  # type: ignore
                if input_name in dummy_input_generator_class.SUPPORTED_INPUT_NAMES:  # type: ignore
                    dummy_input_generator = dummy_input_generator_class(
                        task=self.task,
                        normalized_config=normalized_config,
                        batch_size=self.batch_size,
                    )

            if dummy_input_generator is None:
                raise ValueError(
                    f"Could not find dummy input generator for {input_name}"
                )

            dummy_input[input_name] = dummy_input_generator.generate(
                input_name, framework="pt"
            ).to(self.device)

            # this is for bettertransformer since it does not support random attention mask
            if input_name == "attention_mask":
                dummy_input["attention_mask"] = torch.ones_like(
                    dummy_input["input_ids"]
                )

        return dummy_input
