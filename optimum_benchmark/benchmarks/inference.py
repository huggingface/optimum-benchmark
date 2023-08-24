from dataclasses import dataclass, field
from typing import List, Dict, Optional
from logging import getLogger
from omegaconf import OmegaConf


from pandas import DataFrame
import statistics


from ..backends.base import Backend
from .base import Benchmark, BenchmarkConfig
from ..generators.input_generator import InputGenerator
from ..utils import TEXT_GENERATION_TASKS, DIFFUSION_TASKS
from ..trackers.memory import memory_tracker_class_for_backend
from ..trackers.latency import latency_tracker_class_for_backend
from .inference_utils import (
    three_sig_figs,
    DEFAULT_INPUT_SHAPES,
    DEFAULT_GENERATE_KWARGS,
    DEFAULT_DIFUSION_KWARGS,
)


LOGGER = getLogger("inference")

OmegaConf.register_new_resolver(
    "can_generate",
    lambda task: task in TEXT_GENERATION_TASKS,
)
OmegaConf.register_new_resolver(
    "can_diffuse",
    lambda task: task in DIFFUSION_TASKS,
)


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = "inference"
    _target_: str = "optimum_benchmark.benchmarks.inference.InferenceBenchmark"

    # benchmark options
    memory: bool = False
    warmup_runs: int = 10
    duration: int = 10
    # TODO: deprecate this and use `benchmark.duration`
    benchmark_duration: Optional[int] = None

    # input options
    input_shapes: Dict = field(
        default_factory=lambda: DEFAULT_INPUT_SHAPES,
    )

    # TODO: deprecate this and use `benchamrk.generate_kwargs`
    new_tokens: Optional[int] = None

    # forward options
    can_diffuse: bool = "${can_diffuse:${task}}"
    forward_kwargs: Optional[Dict] = None

    # generation options
    can_generate: bool = "${can_generate:${task}}"
    generate_kwargs: Optional[Dict] = None

    def __post_init__(self):
        if self.can_generate:
            self.generate_kwargs = OmegaConf.merge(
                self.generate_kwargs or {},
                DEFAULT_GENERATE_KWARGS,
            )

        if self.can_diffuse:
            self.forward_kwargs = OmegaConf.merge(
                self.forward_kwargs or {},
                DEFAULT_DIFUSION_KWARGS,
            )

        if self.new_tokens is not None:
            LOGGER.warning(
                "The `new_tokens` option is deprecated, please use `generate_kwargs` "
                "instead. `max_new_tokens` and `min_new_tokens` will be set to the "
                "value of `new_tokens`."
            )
            self.generate_kwargs["max_new_tokens"] = self.new_tokens
            self.generate_kwargs["min_new_tokens"] = self.new_tokens

        if self.generate_kwargs is not None:
            assert (
                self.generate_kwargs["max_new_tokens"]
                == self.generate_kwargs["min_new_tokens"]
            ), (
                "`max_new_tokens` and `min_new_tokens` "
                "must be equal for fixed length output"
            )

        if self.benchmark_duration is not None:
            LOGGER.warning(
                "The `benchmark_duration` option is deprecated, please use `duration` "
                "instead. `duration` will be set to the value of `benchmark_duration`."
            )
            self.duration = self.benchmark_duration


class InferenceBenchmark(Benchmark):
    name: str = "inference"
    config: InferenceConfig

    def __init__(self):
        # initialize inference results
        self.forward_peak_memory: int = 0
        self.forward_latencies: List[float] = []
        self.generate_latencies: List[float] = []

    def configure(self, config: InferenceConfig):
        super().configure(config)

        if self.config.forward_kwargs is None:
            self.config.forward_kwargs = {}

        if self.config.generate_kwargs is None:
            self.config.generate_kwargs = {}

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running inference benchmark")
        self.config.input_shapes.update(backend.model_shapes)

        self.input_generator = InputGenerator(
            task=backend.task,
            input_shapes=self.config.input_shapes,
            pretrained_config=backend.pretrained_config,
        )

        if self.config.memory:
            # if requested, run memory tracking
            self.run_memory_tracking(backend)

        # run forward pass tracking
        self.run_forward_tracking(backend)

        if self.config.can_generate:
            # if possible, run generation pass tracking
            self.run_generate_tracking(backend)

    def run_memory_tracking(self, backend: Backend) -> None:
        memory_input = self.input_generator.generate(
            mode="forward",
        )

        for key, value in memory_input.items():
            if key == "prompt":
                continue
            memory_input[key] = value.to(backend.device)

        # for backends that require compilation with static shapes
        backend.prepare_for_inference(input_shapes=self.config.input_shapes)

        LOGGER.info("\t+ Tracking forward pass peak memory")
        memory_tracker = memory_tracker_class_for_backend[backend.config.name](backend)
        with memory_tracker.track(interval=self.config.duration // 100):
            _ = backend.forward(memory_input)

        self.forward_peak_memory = memory_tracker.get_peak_memory()
        LOGGER.info(f"\t+ Forward pass peak memory: {self.forward_peak_memory} (MB)")

    def run_forward_tracking(self, backend: Backend) -> None:
        forward_input = self.input_generator.generate(
            mode="forward",
        )

        for key, value in forward_input.items():
            if key == "prompt":
                continue
            forward_input[key] = value.to(backend.device)

        # for backends that require compilation with static shapes
        backend.prepare_for_inference(input_shapes=self.config.input_shapes)

        LOGGER.info("\t+ Warming up the forward pass")
        for _ in range(self.config.warmup_runs):
            _ = backend.forward(forward_input, **self.config.forward_kwargs)

        LOGGER.info("\t+ Tracking forward pass latency and throughput")
        latency_tracker = latency_tracker_class_for_backend[backend.config.name](
            backend
        )
        while sum(self.forward_latencies) < self.config.duration:
            with latency_tracker.track():
                _ = backend.forward(forward_input, **self.config.forward_kwargs)
            self.forward_latencies = latency_tracker.get_latencies()

        LOGGER.info(f"\t+ Forward pass latency: {self.forward_latency:.2e} (s)")
        LOGGER.info(
            f"\t+ Forward pass throughput: {self.forward_throughput:.2f} (samples/s)"
        )

    def run_generate_tracking(self, backend: Backend) -> None:
        generate_input = self.input_generator.generate(
            mode="forward",
        )

        for key, value in generate_input.items():
            if key == "prompt":
                continue
            generate_input[key] = value.to(backend.device)

        LOGGER.info("\t+ Warming up the generation pass")
        _ = backend.generate(
            input=generate_input,
            **self.config.generate_kwargs,
        )

        LOGGER.info("\t+ Tracking generation latency and throughput")
        latency_tracker = latency_tracker_class_for_backend[backend.config.name](
            backend
        )
        while sum(self.generate_latencies) < self.config.duration:
            with latency_tracker.track():
                _ = backend.generate(
                    generate_input,
                    **self.config.generate_kwargs,
                )
            self.generate_latencies = latency_tracker.get_latencies()

        LOGGER.info(f"\t+ Generation pass latency: {self.generate_latency:.2e} (s)")

        LOGGER.info(
            f"\t+ Generation pass throughput: {self.generate_throughput:.2f} (tokens/s)"
        )

    # Metrics
    @property
    @three_sig_figs
    def forward_latency(self) -> float:
        return statistics.mean(self.forward_latencies)

    @property
    @three_sig_figs
    def forward_throughput(self) -> float:
        return (
            self.config.input_shapes["batch_size"]
            * self.config.forward_kwargs["num_images_per_prompt"]
            / self.forward_latency
            if self.config.can_diffuse
            else self.config.input_shapes["batch_size"] / self.forward_latency
        )

    @property
    @three_sig_figs
    def generate_latency(self) -> float:
        return statistics.mean(self.generate_latencies)

    @property
    @three_sig_figs
    def generate_throughput(self) -> float:
        return (
            self.config.generate_kwargs["min_new_tokens"]
            * self.config.input_shapes["batch_size"]
            / self.generate_latency
        )

    def get_results_df(self) -> DataFrame:
        results_dict = dict()

        if self.config.memory:
            results_dict["forward.peak_memory(MB)"] = self.forward_peak_memory

        results_dict["forward.latency(s)"] = self.forward_latency
        results_dict["forward.throughput(samples/s)"] = self.forward_throughput

        if self.config.can_generate:
            results_dict["generate.latency(s)"] = self.generate_latency
            results_dict["generate.throughput(tokens/s)"] = self.generate_throughput

        return DataFrame(results_dict, index=[0])

    def save(self) -> None:
        LOGGER.info("Saving inference results")
        results_df = self.get_results_df()
        results_df.to_csv("inference_results.csv")
