import os
import statistics
from logging import getLogger
from typing import List, Dict, Any

from ..base import Benchmark
from .config import InferenceConfig
from ...backends.base import Backend
from ...trackers.energy import EnergyTracker
from ...trackers.memory import MemoryTracker
from ...trackers.latency import LatencyTracker
from ...generators.input_generator import InputGenerator
from ...task_utils import TEXT_GENERATION_TASKS, DIFFUSION_TASKS

LOGGER = getLogger("inference")

DIFFUSION_KWARGS = {
    "num_images_per_prompt": 1,
}

GENERATE_KWARGS = {
    "num_return_sequences": 1,
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "temperature": 1.0,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": 0,
    "num_beams": 1,
}


class InferenceBenchmark(Benchmark[InferenceConfig]):
    NAME = "inference"

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)

        self.forward_energy: float = 0
        self.forward_emissions: float = 0
        self.forward_max_memory_used: int = 0
        self.forward_max_memory_allocated: int = 0
        self.forward_max_memory_reserved: int = 0
        self.forward_latencies: List[float] = []

        self.generate_energy: float = 0
        self.generate_emissions: float = 0
        self.generate_max_memory_used: int = 0
        self.generate_max_memory_allocated: int = 0
        self.generate_max_memory_reserved: int = 0
        self.generate_latencies: List[float] = []

    def run(self, backend: Backend) -> None:
        self.can_diffuse = backend.config.task in DIFFUSION_TASKS
        self.can_generate = backend.config.task in TEXT_GENERATION_TASKS

        if self.can_diffuse:
            LOGGER.info("\t+ Updating forward kwargs with default values")
            self.config.forward_kwargs = {
                **DIFFUSION_KWARGS,
                **self.config.forward_kwargs,
            }
        if self.can_generate:
            LOGGER.info("\t+ Updating generate kwargs with default values")
            self.config.generate_kwargs = {
                **GENERATE_KWARGS,
                **self.config.generate_kwargs,
            }

        # compile with static shapes if needed
        LOGGER.info("\t+ Preparing backend for inference")
        backend.prepare_for_inference(
            **backend.model_shapes,
            **self.config.input_shapes,
            **self.config.forward_kwargs,
            **self.config.generate_kwargs,
        )

        LOGGER.info("\t+ Creating input generator")
        self.input_generator = InputGenerator(
            task=backend.config.task,
            model_shapes=backend.model_shapes,
            input_shapes=self.config.input_shapes,
        )

        # run memory tracking
        # we do this first to measure the memory on the first call to forward/generate
        if self.config.memory:
            self.run_forward_memory_tracking(backend)
            if self.can_generate:
                self.run_generate_memory_tracking(backend)

        # run lacency tracking
        self.run_forward_latency_tracking(backend)
        if self.can_generate:
            self.run_generate_latency_tracking(backend)

        # run energy tracking
        if self.config.energy:
            self.run_forward_energy_tracking(backend)
            if self.can_generate:
                self.run_generate_energy_tracking(backend)

    def run_forward_latency_tracking(self, backend: "Backend") -> None:
        forward_input = self.input_generator.generate(mode="forward")

        LOGGER.info("\t+ Preparing input for the forward pass")
        forward_input = backend.prepare_inputs(forward_input)

        LOGGER.info("\t+ Warming up the forward pass")
        for _ in range(self.config.warmup_runs):
            _ = backend.forward(forward_input, self.config.forward_kwargs)

        LOGGER.info("\t+ Tracking forward pass latency and throughput")
        latency_tracker = LatencyTracker(device=backend.config.device, backend=backend.config.name)
        while sum(self.forward_latencies) < self.config.duration:
            with latency_tracker.track():
                _ = backend.forward(forward_input, self.config.forward_kwargs)
            self.forward_latencies = latency_tracker.get_latencies()

        LOGGER.info(f"\t+ Forward pass latency: {self.forward_latency:.3g} (s)")
        LOGGER.info(f"\t+ Forward pass throughput: {self.forward_throughput:.3g} (samples/s)")

    def run_forward_energy_tracking(self, backend: Backend) -> None:
        forward_input = self.input_generator.generate(mode="forward")

        LOGGER.info("\t+ Preparing input for the forward pass")
        forward_input = backend.prepare_inputs(forward_input)

        LOGGER.info("\t+ Tracking forward pass energy consumption")
        num_forward_passes = 0
        energy_tracker = EnergyTracker()
        with energy_tracker.track(interval=1, file_prefix="forward"):
            while energy_tracker.get_elapsed_time() < self.config.duration:
                _ = backend.forward(forward_input, self.config.forward_kwargs)
                num_forward_passes += 1
        num_forward_samples = num_forward_passes * self.config.input_shapes["batch_size"]
        self.forward_energy = energy_tracker.get_total_energy() / num_forward_samples
        self.forward_emissions = energy_tracker.get_total_emissions() / num_forward_samples

        LOGGER.info(f"\t+ Forward pass energy consumption: {self.forward_energy:.3g} (kWh/sample)")
        LOGGER.info(f"\t+ Forward pass carbon emissions: {self.forward_emissions:.3g} (kgCO2eq/sample)")
        LOGGER.info(f"\t+ Full details in the CodeCarbon report: {os.getcwd()}/forward_codecarbon.csv")

    def run_forward_memory_tracking(self, backend: "Backend") -> None:
        forward_input = self.input_generator.generate(mode="forward")

        LOGGER.info("\t+ Preparing input for the forward pass")
        forward_input = backend.prepare_inputs(forward_input)

        LOGGER.info("\t+ Tracking forward pass peak memory")
        memory_tracker = MemoryTracker(device=backend.config.device, backend=backend.config.name)
        with memory_tracker.track():
            _ = backend.forward(forward_input, self.config.forward_kwargs)
        self.forward_max_memory_used = memory_tracker.get_max_memory_used()
        self.forward_max_memory_reserved = memory_tracker.get_max_memory_reserved()
        self.forward_max_memory_allocated = memory_tracker.get_max_memory_allocated()

        LOGGER.info(f"\t+ Forward pass max memory used: {self.forward_max_memory_used:.3g} (MB)")
        LOGGER.info(f"\t+ Forward pass max memory reserved: {self.forward_max_memory_reserved:.3g} (MB)")
        LOGGER.info(f"\t+ Forward pass max memory allocated: {self.forward_max_memory_allocated:.3g} (MB)")

    def run_generate_latency_tracking(self, backend: "Backend") -> None:
        generate_input = self.input_generator.generate(mode="generate")

        LOGGER.info("\t+ Preparing input for the generation pass")
        generate_input = backend.prepare_inputs(generate_input)

        LOGGER.info("\t+ Warming up the generation pass")
        _ = backend.generate(generate_input, self.config.generate_kwargs)

        LOGGER.info("\t+ Tracking generation latency and throughput")
        latency_tracker = LatencyTracker(device=backend.config.device, backend=backend.config.name)
        while sum(self.generate_latencies) < self.config.duration:
            with latency_tracker.track():
                _ = backend.generate(generate_input, self.config.generate_kwargs)
            self.generate_latencies = latency_tracker.get_latencies()

        LOGGER.info(f"\t+ Generation pass latency: {self.generate_latency:.3g} (s)")
        LOGGER.info(f"\t+ Generation pass throughput: {self.generate_throughput:.3g} (tokens/s)")

    def run_generate_energy_tracking(self, backend: Backend) -> None:
        generate_input = self.input_generator.generate(mode="generate")

        LOGGER.info("\t+ Preparing input for the generation pass")
        generate_input = backend.prepare_inputs(generate_input)

        LOGGER.info("\t+ Tracking generation pass energy consumption")
        num_generate_passes = 0
        energy_tracker = EnergyTracker()
        with energy_tracker.track(interval=1, file_prefix="generate"):
            while energy_tracker.get_elapsed_time() < self.config.duration:
                _ = backend.generate(generate_input, self.config.generate_kwargs)
                num_generate_passes += 1
        num_generated_tokens = (
            num_generate_passes
            * self.config.generate_kwargs["min_new_tokens"]
            * self.config.generate_kwargs["num_return_sequences"]
            * self.config.input_shapes["batch_size"]
        )
        self.generate_energy = energy_tracker.get_total_energy() / num_generated_tokens
        self.generate_emissions = energy_tracker.get_total_emissions() / num_generated_tokens

        LOGGER.info(f"\t+ Generation pass energy consumption: {self.generate_energy:.3g} (kWh/token)")
        LOGGER.info(f"\t+ Generation pass carbon emissions: {self.generate_emissions:.3g} (kgCO2eq/token)")
        LOGGER.info(f"\t+ Full details in the CodeCarbon report: {os.getcwd()}/generate_codecarbon.csv")

    def run_generate_memory_tracking(self, backend: "Backend") -> None:
        generate_input = self.input_generator.generate(mode="generate")

        LOGGER.info("\t+ Preparing input for the generation pass")
        generate_input = backend.prepare_inputs(generate_input)

        LOGGER.info("\t+ Tracking generation pass peak memory")
        memory_tracker = MemoryTracker(device=backend.config.device, backend=backend.config.name)
        with memory_tracker.track():
            _ = backend.generate(generate_input, self.config.generate_kwargs)
        self.generate_max_memory_used = memory_tracker.get_max_memory_used()
        self.generate_max_memory_reserved = memory_tracker.get_max_memory_reserved()
        self.generate_max_memory_allocated = memory_tracker.get_max_memory_allocated()

        LOGGER.info(f"\t+ Generation pass max memory used: {self.generate_max_memory_used:.3g} (MB)")
        LOGGER.info(f"\t+ Generation pass max memory reserved: {self.generate_max_memory_reserved:.3g} (MB)")
        LOGGER.info(f"\t+ Generation pass max memory allocated: {self.generate_max_memory_allocated:.3g} (MB)")

    # Metrics
    ## Forward pass metrics
    @property
    def forward_latency(self) -> float:
        return statistics.mean(self.forward_latencies)

    @property
    def forward_throughput(self) -> float:
        return self.config.input_shapes["batch_size"] / self.forward_latency

    ## Generation pass metrics
    @property
    def generate_latency(self) -> float:
        return statistics.mean(self.generate_latencies)

    @property
    def generate_throughput(self) -> float:
        return (
            self.config.generate_kwargs["min_new_tokens"]
            * self.config.generate_kwargs["num_return_sequences"]
            * self.config.input_shapes["batch_size"]
            / self.generate_latency
        )

    @property
    def decode_latency(self) -> float:
        return self.generate_latency - self.forward_latency

    @property
    def decode_throughput(self) -> float:
        return (
            (self.config.generate_kwargs["min_new_tokens"] - 1)
            * self.config.generate_kwargs["num_return_sequences"]
            * self.config.input_shapes["batch_size"]
            / self.decode_latency
        )

    ## Diffusion pass metrics
    @property
    def diffusion_throughput(self) -> float:
        return (
            self.config.input_shapes["batch_size"]
            * self.config.forward_kwargs["num_images_per_prompt"]
            / self.forward_latency
        )

    def report(self) -> Dict[str, Any]:
        report_dict = {"benchmark": self.NAME}

        report_dict["forward.latency(s)"] = self.forward_latency
        report_dict["forward.throughput(samples/s)"] = self.forward_throughput

        if self.can_diffuse:
            report_dict["diffusion.throughput(images/s)"] = self.diffusion_throughput

        if self.config.memory:
            report_dict["forward.peak_memory(MB)"] = self.forward_max_memory_used
            report_dict["forward.max_memory_used(MB)"] = self.forward_max_memory_used
            report_dict["forward.max_memory_allocated(MB)"] = self.forward_max_memory_allocated
            report_dict["forward.max_memory_reserved(MB)"] = self.forward_max_memory_reserved

        if self.config.energy:
            report_dict["forward.energy_consumption(kWh/sample)"] = self.forward_energy
            report_dict["forward.carbon_emissions(kgCO2eq/sample)"] = self.forward_emissions

        if self.can_generate:
            report_dict["generate.latency(s)"] = self.generate_latency
            report_dict["generate.throughput(tokens/s)"] = self.generate_throughput

            report_dict["decode.latency(s)"] = self.decode_latency
            report_dict["decode.throughput(tokens/s)"] = self.decode_throughput

            if self.config.memory:
                report_dict["generate.peak_memory(MB)"] = self.generate_max_memory_used
                report_dict["generate.max_memory_used(MB)"] = self.generate_max_memory_used
                report_dict["generate.max_memory_allocated(MB)"] = self.generate_max_memory_allocated
                report_dict["generate.max_memory_reserved(MB)"] = self.generate_max_memory_reserved

            if self.config.energy:
                report_dict["generate.energy_consumption(kWh/token)"] = self.generate_energy
                report_dict["generate.carbon_emissions(kgCO2eq/token)"] = self.generate_emissions

        return report_dict
