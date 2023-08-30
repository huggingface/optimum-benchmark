import statistics
from logging import getLogger
from typing import TYPE_CHECKING, Dict, List

from pandas import DataFrame

from ...generators.input_generator import InputGenerator
from ...trackers.energy import EnergyTracker
from ...trackers.latency import latency_tracker_class_for_backend
from ...trackers.memory import memory_tracker_class_for_backend
from ..base import Benchmark
from ..utils import three_significant_digits_wrapper
from .config import InferenceConfig

if TYPE_CHECKING:
    from ...backends.base import Backend

LOGGER = getLogger("inference")


class InferenceBenchmark(Benchmark[InferenceConfig]):
    NAME = "inference"

    def __init__(self):
        # initialize inference results
        self.forward_peak_memory: int = 0
        self.forward_latencies: List[float] = []
        self.forward_energies: Dict[str, float] = {}

        self.generate_peak_memory: int = 0
        self.generate_latencies: List[float] = []
        self.generate_energies: Dict[str, float] = {}

    def configure(self, config: "InferenceConfig"):
        super().configure(config)

    def run(self, backend: "Backend") -> None:
        LOGGER.info("Running inference benchmark")
        self.config.input_shapes.update(backend.model_shapes)

        self.input_generator = InputGenerator(
            task=backend.task,
            pretrained_config=backend.pretrained_config,
            input_shapes=self.config.input_shapes,
        )

        # run forward pass tracking
        self.run_forward_tracking(backend)

        if self.config.can_generate:
            # if possible, run generation pass tracking
            self.run_generate_tracking(backend)

    def run_forward_tracking(self, backend: "Backend") -> None:
        forward_input = self.input_generator.generate(mode="forward")

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
            with memory_tracker.track(interval=self.forward_latency / 10):
                _ = backend.forward(forward_input)

            self.forward_peak_memory = memory_tracker.get_peak_memory()
            LOGGER.info(f"\t+ Forward pass peak memory: {self.forward_peak_memory} (MB)")

        if self.config.energy:
            LOGGER.info("\t+ Tracking forward pass energy consumption")
            energy_tracker = EnergyTracker()
            with energy_tracker.track(interval=1, file_prefix="forward"):
                while energy_tracker.get_elapsed_time() < self.config.duration:
                    _ = backend.forward(forward_input, **self.config.forward_kwargs)

            self.forward_energies = energy_tracker.get_energies()
            LOGGER.info(f"\t+ Forward pass energy consumption: {self.forward_energy} (kWh)")

    def run_generate_tracking(self, backend: "Backend") -> None:
        generate_input = self.input_generator.generate(mode="generate")

        # TODO: can be handled by the backend later
        for key, value in generate_input.items():
            if key == "prompt":
                continue
            generate_input[key] = value.to(backend.device)

        LOGGER.info("\t+ Warming up the generation pass")
        _ = backend.generate(input=generate_input, **self.config.generate_kwargs)

        LOGGER.info("\t+ Tracking generation latency and throughput")
        latency_tracker = latency_tracker_class_for_backend[backend.config.name](backend)
        while sum(self.generate_latencies) < self.config.duration:
            with latency_tracker.track():
                _ = backend.generate(generate_input, **self.config.generate_kwargs)
            self.generate_latencies = latency_tracker.get_latencies()

        LOGGER.info(f"\t+ Generation pass latency: {self.generate_latency:.2e} (s)")
        LOGGER.info(f"\t+ Generation pass throughput: {self.generate_throughput:.2f} (tokens/s)")

        if self.config.memory:
            LOGGER.info("\t+ Tracking generation pass peak memory")
            memory_tracker = memory_tracker_class_for_backend[backend.config.name](backend)
            with memory_tracker.track(interval=self.generate_latency / 10):
                _ = backend.generate(generate_input, **self.config.generate_kwargs)
            self.generate_peak_memory = memory_tracker.get_peak_memory()
            LOGGER.info(f"\t+ Generation pass peak memory: {self.generate_peak_memory} (MB)")

        if self.config.energy:
            LOGGER.info("\t+ Tracking forward pass energy consumption")
            energy_tracker = EnergyTracker()
            with energy_tracker.track(interval=1, file_prefix="generate"):
                while energy_tracker.get_elapsed_time() < self.config.duration:
                    _ = backend.generate(generate_input, **self.config.generate_kwargs)

            self.generate_energies = energy_tracker.get_energies()
            LOGGER.info(f"\t+ Forward pass energy consumption: {self.forward_energy} (kWh)")

    # Metrics

    ## Forward pass metrics
    @property
    @three_significant_digits_wrapper
    def forward_latency(self) -> float:
        return statistics.mean(self.forward_latencies)

    @property
    @three_significant_digits_wrapper
    def forward_throughput(self) -> float:
        return self.config.input_shapes["batch_size"] / self.forward_latency

    @property
    @three_significant_digits_wrapper
    def forward_energy(self) -> float:
        return self.forward_energies["total_energy"] / len(self.forward_latencies)

    ## Generation pass metrics
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

    @property
    @three_significant_digits_wrapper
    def generate_energy(self) -> float:
        return self.generate_energies["total_energy"] / len(self.generate_latencies)

    ## Diffusion pass metrics
    @property
    @three_significant_digits_wrapper
    def diffusion_throughput(self) -> float:
        return (
            self.config.input_shapes["batch_size"]
            * self.config.forward_kwargs["num_images_per_prompt"]
            / self.forward_latency
        )

    def get_results_df(self) -> DataFrame:
        results_dict = {}

        results_dict["forward.latency(s)"] = self.forward_latency
        results_dict["forward.throughput(samples/s)"] = self.forward_throughput

        if self.config.memory:
            results_dict["forward.peak_memory(MB)"] = self.forward_peak_memory

        if self.config.energy:
            results_dict["forward.energy_consumption(kWh)"] = self.forward_energy

        if self.config.can_generate:
            results_dict["generate.latency(s)"] = self.generate_latency
            results_dict["generate.throughput(tokens/s)"] = self.generate_throughput

            if self.config.memory:
                results_dict["generate.peak_memory(MB)"] = self.generate_peak_memory

            if self.config.energy:
                results_dict["generate.energy_consumption(kWh)"] = self.generate_energy

        if self.config.can_diffuse:
            results_dict["diffusion.throughput(images/s)"] = self.diffusion_throughput

        return DataFrame(results_dict, index=[0])

    def save(self) -> None:
        LOGGER.info("Saving inference results")
        results_df = self.get_results_df()
        results_df.to_csv("inference_results.csv")
