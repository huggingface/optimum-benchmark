from logging import getLogger
from dataclasses import dataclass

from ..base import Benchmark
from .config import InferenceConfig
from ...trackers.memory import MemoryTracker
from ...report import BenchmarkReport, Measurements
from ...backends.base import Backend, BackendConfigT
from ...generators.input_generator import InputGenerator
from ...trackers.energy import EnergyTracker, Efficiency
from ...trackers.latency import LatencyTracker, Throughput
from ...import_utils import is_torch_distributed_available
from ...task_utils import TEXT_GENERATION_TASKS, IMAGE_DIFFUSION_TASKS

if is_torch_distributed_available():
    import torch.distributed

LOGGER = getLogger("inference")

IMAGE_DIFFUSION_KWARGS = {
    "num_inference_steps": 30,
    "num_images_per_prompt": 1,
}

TEXT_GENERATION_KWARGS = {
    "num_return_sequences": 1,
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "temperature": 1.0,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": 0,
    "num_beams": 1,
}

EFFICIENCY_UNIT = "samples/kWh"
THROUGHPUT_UNIT = "samples/s"

PREFILL_THROUGHPUT_UNIT = "tokens/s"
DECODE_THROUGHPUT_UNIT = "tokens/s"
CALL_THROUGHPUT_UNIT = "images/s"

PREFILL_EFFICIENCY_UNIT = "tokens/kWh"
DECODE_EFFICIENCY_UNIT = "tokens/kWh"
CALL_EFFICIENCY_UNIT = "images/kWh"


@dataclass
class InferenceReport(BenchmarkReport):
    forward: Measurements = Measurements()


@dataclass
class ImageDiffusionReport(BenchmarkReport):
    call: Measurements = Measurements()


@dataclass
class TextGenerationReport(BenchmarkReport):
    prefill: Measurements = Measurements()
    decode: Measurements = Measurements()


class InferenceBenchmark(Benchmark[InferenceConfig]):
    NAME = "inference"

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT]) -> None:
        if is_torch_distributed_available() and torch.distributed.is_initialized():
            LOGGER.info("\t+ Distributing batch size across processes")
            if self.config.input_shapes["batch_size"] % torch.distributed.get_world_size() != 0:
                raise ValueError(
                    "The batch size must be divisible by the number of processes in a distributed environment"
                )
            self.config.input_shapes["batch_size"] //= torch.distributed.get_world_size()

        LOGGER.info("\t+ Creating input generator")
        self.input_generator = InputGenerator(
            task=backend.config.task,
            model_shapes=backend.model_shapes,
            input_shapes=self.config.input_shapes,
        )

        if backend.config.task in TEXT_GENERATION_TASKS:
            LOGGER.info("\t+ Generating and preparing Text Generation input")
            self.forward_inputs = self.input_generator(mode="forward")
            self.generate_input = self.input_generator(mode="generate")
            self.forward_inputs = backend.prepare_inputs(self.forward_inputs)
            self.generate_input = backend.prepare_inputs(self.generate_input)
            LOGGER.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_KWARGS, **self.config.generate_kwargs}
            LOGGER.info("\t+ Initializing Text Generation report")
            self.report = TextGenerationReport()

        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            LOGGER.info("\t+ Generating and preparing Image Diffusion input")
            self.diffuse_input = self.input_generator(mode="call")
            self.diffuse_input = backend.prepare_inputs(self.diffuse_input)
            LOGGER.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.forward_kwargs = {**IMAGE_DIFFUSION_KWARGS, **self.config.forward_kwargs}
            LOGGER.info("\t+ Initializing Image Diffusion report")
            self.report = ImageDiffusionReport()

        else:
            LOGGER.info("\t+ Generating and preparing Inference input")
            self.forward_inputs = self.input_generator(mode="forward")
            self.forward_inputs = backend.prepare_inputs(self.forward_inputs)
            LOGGER.info("\t+ Initializing Inference report")
            self.report = InferenceReport()

        LOGGER.info("\t+ Preparing backend for Inference")
        backend.prepare_for_inference(
            **backend.model_shapes,
            **self.config.input_shapes,
            **self.config.forward_kwargs,
            **self.config.generate_kwargs,
        )

        LOGGER.info("\t+ Warming up backend for Inference")
        for _ in range(self.config.warmup_runs):
            if backend.config.task in TEXT_GENERATION_TASKS:
                _ = backend.generate(self.generate_input, {"max_new_tokens": 2, "min_new_tokens": 2})
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                _ = backend.call(self.diffuse_input, {"num_inference_steps": 2})
            else:
                _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        if self.config.memory:
            LOGGER.info("\t+ Creating inference memory tracker")
            self.memory_tracker = MemoryTracker(
                backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
            )
            if backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_memory_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_memory_tracking(backend)
            else:
                self.run_inference_memory_tracking(backend)

            self.report.log_memory()

        if self.config.latency:
            LOGGER.info("\t+ Creating inference latency tracker")
            self.latency_tracker = LatencyTracker(backend=backend.config.name, device=backend.config.device)
            if backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_latency_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_latency_tracking(backend)
            else:
                self.run_latency_inference_tracking(backend)

            self.report.log_latency()
            self.report.log_throughput()

        if self.config.energy:
            LOGGER.info("\t+ Creating inference energy tracker")
            self.energy_tracker = EnergyTracker(device=backend.config.device, device_ids=backend.config.device_ids)
            if backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_energy_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_energy_tracking(backend)
            else:
                self.run_inference_energy_tracking(backend)

            self.report.log_energy()
            self.report.log_efficiency()

    ## Memory tracking
    def run_text_generation_memory_tracking(self, backend: Backend):
        LOGGER.info("\t+ Running memory tracking")
        self.memory_tracker.reset()
        with self.memory_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.prefill.memory = self.memory_tracker.get_max_memory()

        self.memory_tracker.reset()
        with self.memory_tracker.track():
            _ = backend.generate(self.generate_input, self.config.generate_kwargs)

        self.report.decode.memory = self.memory_tracker.get_max_memory()

    def run_image_diffusion_memory_tracking(self, backend: Backend):
        LOGGER.info("\t+ Running memory tracking")
        self.memory_tracker.reset()
        with self.memory_tracker.track():
            _ = backend.call(self.diffuse_input, self.config.forward_kwargs)

        self.report.call.memory = self.memory_tracker.get_max_memory()

    def run_inference_memory_tracking(self, backend: Backend):
        LOGGER.info("\t+ Running memory tracking")
        self.memory_tracker.reset()
        with self.memory_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.forward.memory = self.memory_tracker.get_max_memory()

    ## Latency tracking
    def run_text_generation_latency_tracking(self, backend: Backend):
        LOGGER.info("\t+ Running latency tracking")
        self.latency_tracker.reset()
        while self.latency_tracker.get_elapsed_time() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.prefill.latency = self.latency_tracker.get_latency()
        self.report.prefill.throughput = Throughput.from_latency(
            self.report.prefill.latency, self.prefill_volume, unit=PREFILL_THROUGHPUT_UNIT
        )

        self.latency_tracker.reset()
        while self.latency_tracker.get_elapsed_time() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.generate(self.generate_input, self.config.generate_kwargs)

        self.report.decode.latency = self.latency_tracker.get_latency() - self.report.prefill.latency.mean
        self.report.decode.throughput = Throughput.from_latency(
            self.report.decode.latency, self.decode_volume, unit=DECODE_THROUGHPUT_UNIT
        )

    def run_image_diffusion_latency_tracking(self, backend: Backend):
        LOGGER.info("\t+ Running latency tracking")
        self.latency_tracker.reset()
        while self.latency_tracker.get_elapsed_time() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.call(self.diffuse_input, self.config.forward_kwargs)

        self.report.call.latency = self.latency_tracker.get_latency()
        self.report.call.throughput = Throughput.from_latency(
            self.report.call.latency, self.call_volume, unit=CALL_THROUGHPUT_UNIT
        )

    def run_latency_inference_tracking(self, backend: Backend):
        LOGGER.info("\t+ Running latency tracking")
        self.latency_tracker.reset()
        while self.latency_tracker.get_elapsed_time() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.forward.latency = self.latency_tracker.get_latency()
        self.report.forward.throughput = Throughput.from_latency(
            self.report.forward.latency, self.forward_volume, unit=THROUGHPUT_UNIT
        )

    ## Energy tracking
    def run_text_generation_energy_tracking(self, backend: Backend):
        LOGGER.info("\t+ Running energy tracking")
        self.energy_tracker.reset()
        with self.energy_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.prefill.energy = self.energy_tracker.get_energy()
        self.report.prefill.efficiency = Efficiency.from_energy(
            self.report.prefill.energy, self.prefill_volume, unit=PREFILL_EFFICIENCY_UNIT
        )

        self.energy_tracker.reset()
        with self.energy_tracker.track():
            _ = backend.generate(self.generate_input, self.config.generate_kwargs)

        self.report.decode.energy = self.energy_tracker.get_energy() - self.report.prefill.energy
        self.report.decode.efficiency = Efficiency.from_energy(
            self.report.decode.energy, self.decode_volume, unit=DECODE_EFFICIENCY_UNIT
        )

    def run_image_diffusion_energy_tracking(self, backend: Backend):
        LOGGER.info("\t+ Running energy tracking")
        self.energy_tracker.reset()
        with self.energy_tracker.track():
            _ = backend.call(self.diffuse_input, self.config.forward_kwargs)

        self.report.call.energy = self.energy_tracker.get_energy()
        self.report.call.efficiency = Efficiency.from_energy(
            self.report.call.energy, self.call_volume, unit=CALL_EFFICIENCY_UNIT
        )

    def run_inference_energy_tracking(self, backend: Backend):
        LOGGER.info("\t+ Running energy tracking")
        self.energy_tracker.reset()
        with self.energy_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.forward.energy = self.energy_tracker.get_energy()
        self.report.forward.efficiency = Efficiency.from_energy(
            self.report.forward.energy, self.forward_volume, unit=EFFICIENCY_UNIT
        )

    @property
    def forward_volume(self) -> int:  # in samples
        return self.config.input_shapes["batch_size"]

    @property
    def prefill_volume(self) -> int:  # in tokens
        return self.config.input_shapes["batch_size"] * self.config.input_shapes["sequence_length"]

    @property
    def call_volume(self) -> int:  # in images
        return self.config.input_shapes["batch_size"] * self.config.forward_kwargs["num_images_per_prompt"]

    @property
    def decode_volume(self) -> int:  # in tokens
        return (
            self.config.input_shapes["batch_size"]
            * self.config.generate_kwargs["num_return_sequences"]
            * self.config.generate_kwargs["max_new_tokens"]
        )

    def get_report(self) -> InferenceReport:
        return self.report
