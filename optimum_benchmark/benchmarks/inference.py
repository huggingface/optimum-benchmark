import statistics
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
from pandas import DataFrame

from ..backends.base import Backend
from ..generators.input_generator import InputGenerator
from ..task_utils import DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ..trackers.latency import latency_tracker_class_for_backend
from ..trackers.memory import memory_tracker_class_for_backend
from .base import Benchmark, BenchmarkConfig
from .utils import three_significant_digits_wrapper

LOGGER = getLogger("inference")

OmegaConf.register_new_resolver(
    "can_generate",
    lambda task: task in TEXT_GENERATION_TASKS,
)
OmegaConf.register_new_resolver(
    "can_diffuse",
    lambda task: task in DIFFUSION_TASKS,
)

GENERATE_CONFIG = {
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": 0,
    "num_beams": 1,
}

DIFUSION_CONFIG = {
    "num_images_per_prompt": 1,
}


@dataclass
class InferenceConfig(BenchmarkConfig):
    name: str = "inference"
    _target_: str = "optimum_benchmark.benchmarks.inference.InferenceBenchmark"

    # benchmark options
    memory: bool = False
    duration: int = 10
    warmup_runs: int = 10
    benchmark_duration: Optional[int] = None

    # input options
    input_shapes: Dict = field(
        default_factory=lambda: {
            # used with all tasks
            "batch_size": 2,
            # used with text input tasks
            "sequence_length": 16,
            # used with multiple choice tasks where input
            # is of shape (batch_size, num_choices, sequence_length)
            "num_choices": 1,
            # used with audio input tasks
            "feature_size": 80,
            "nb_max_frames": 3000,
            "audio_sequence_length": 16000,
        },
    )

    # TODO: deprecate this and use `benchamrk.generate_kwargs`
    new_tokens: Optional[int] = None

    # forward options
    can_diffuse: bool = "${can_diffuse:${task}}"
    forward_kwargs: Dict[str, Any] = field(default_factory=dict)

    # generation options
    can_generate: bool = "${can_generate:${task}}"
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.can_diffuse:
            self.forward_kwargs = OmegaConf.to_container(OmegaConf.merge(self.forward_kwargs, DIFUSION_CONFIG))

        if self.can_generate:
            self.generate_kwargs = OmegaConf.to_container(OmegaConf.merge(self.generate_kwargs, GENERATE_CONFIG))

            if self.generate_kwargs["max_new_tokens"] != self.generate_kwargs["min_new_tokens"]:
                raise ValueError("`max_new_tokens` and `min_new_tokens` must be equal for fixed length output.")

        if self.new_tokens is not None:
            LOGGER.warning(
                "The `new_tokens` option is deprecated, please use `generate_kwargs` instead. "
                "`generate_kwargs.max_new_tokens` and `generate_kwargs.min_new_tokens` will be set to the value of `new_tokens`."
            )
            self.generate_kwargs["max_new_tokens"] = self.new_tokens
            self.generate_kwargs["min_new_tokens"] = self.new_tokens

        if self.benchmark_duration:
            LOGGER.warning(
                "The `benchmark_duration` option is deprecated, please use `duration` instead. "
                "`duration` will be set to the value of `benchmark_duration`."
            )
            self.duration = self.benchmark_duration


class InferenceBenchmark(Benchmark[InferenceConfig]):
    NAME = "inference"

    def __init__(self):
        # initialize inference results
        self.forward_peak_memory: int = 0
        self.forward_latencies: List[float] = []
        self.generate_latencies: List[float] = []

    def configure(self, config: InferenceConfig):
        super().configure(config)

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running inference benchmark")
        self.config.input_shapes.update(backend.model_shapes)

        self.input_generator = InputGenerator(
            task=backend.task,
            input_shapes=self.config.input_shapes,
            pretrained_config=backend.pretrained_config,
        )

        # run forward pass tracking
        self.run_forward_tracking(backend)

        if self.config.can_generate:
            # if possible, run generation pass tracking
            self.run_generate_tracking(backend)

    def run_forward_tracking(self, backend: Backend) -> None:
        forward_input = self.input_generator.generate(
            mode="forward",
        )

        # TODO: can be handled by the backend later
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
        latency_tracker = latency_tracker_class_for_backend[backend.config.name](backend)
        while sum(self.forward_latencies) < self.config.duration:
            with latency_tracker.track():
                _ = backend.forward(forward_input, **self.config.forward_kwargs)
            self.forward_latencies = latency_tracker.get_latencies()

        LOGGER.info(f"\t+ Forward pass latency: {self.forward_latency:.2e} (s)")
        LOGGER.info(f"\t+ Forward pass throughput: {self.forward_throughput:.2f} (samples/s)")

        if self.config.memory:
            LOGGER.info("\t+ Tracking forward pass peak memory")
            memory_tracker = memory_tracker_class_for_backend[backend.config.name](backend)
            with memory_tracker.track(interval=self.config.duration // 100):
                _ = backend.forward(forward_input)

            self.forward_peak_memory = memory_tracker.get_peak_memory()
            LOGGER.info(f"\t+ Forward pass peak memory: {self.forward_peak_memory} (MB)")

    def run_generate_tracking(self, backend: Backend) -> None:
        generate_input = self.input_generator.generate(
            mode="generate",
        )

        # TODO: can be handled by the backend later
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
        latency_tracker = latency_tracker_class_for_backend[backend.config.name](backend)
        while sum(self.generate_latencies) < self.config.duration:
            with latency_tracker.track():
                _ = backend.generate(
                    generate_input,
                    **self.config.generate_kwargs,
                )
            self.generate_latencies = latency_tracker.get_latencies()

        LOGGER.info(f"\t+ Generation pass latency: {self.generate_latency:.2e} (s)")
        LOGGER.info(f"\t+ Generation pass throughput: {self.generate_throughput:.2f} (tokens/s)")

    # Metrics
    @property
    @three_significant_digits_wrapper
    def forward_latency(self) -> float:
        return statistics.mean(self.forward_latencies)

    @property
    @three_significant_digits_wrapper
    def forward_throughput(self) -> float:
        if self.config.can_diffuse:
            return (
                self.config.input_shapes["batch_size"]
                * self.config.forward_kwargs["num_images_per_prompt"]
                / self.forward_latency
            )
        else:
            return self.config.input_shapes["batch_size"] / self.forward_latency

    @property
    @three_significant_digits_wrapper
    def generate_latency(self) -> float:
        return statistics.mean(self.generate_latencies)

    @property
    @three_significant_digits_wrapper
    def generate_throughput(self) -> float:
        return (
            self.config.generate_kwargs["min_new_tokens"]
            * self.config.input_shapes["batch_size"]
            / self.generate_latency
        )

    def get_results_df(self) -> DataFrame:
        results_dict = {}

        results_dict["forward.latency(s)"] = self.forward_latency

        if self.config.can_diffuse:
            results_dict["forward.throughput(images/s)"] = self.forward_throughput
        else:
            results_dict["forward.throughput(samples/s)"] = self.forward_throughput

        if self.config.memory:
            results_dict["forward.peak_memory(MB)"] = self.forward_peak_memory

        if self.config.can_generate:
            results_dict["generate.latency(s)"] = self.generate_latency
            results_dict["generate.throughput(tokens/s)"] = self.generate_throughput

        return DataFrame(results_dict, index=[0])

    def save(self) -> None:
        LOGGER.info("Saving inference results")
        results_df = self.get_results_df()
        results_df.to_csv("inference_results.csv")
