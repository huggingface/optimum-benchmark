from dataclasses import dataclass
from logging import getLogger

from transformers import LogitsProcessorList

from ...backends.base import Backend, BackendConfigT
from ...generators.input_generator import InputGenerator
from ...import_utils import is_torch_distributed_available
from ...task_utils import IMAGE_DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ...trackers.energy import Efficiency, EnergyTracker
from ...trackers.latency import LatencyLogitsProcessor, LatencyTracker, Throughput
from ...trackers.memory import MemoryTracker
from ..base import Benchmark
from ..report import BenchmarkMeasurements, BenchmarkReport
from .config import InferenceConfig
from .inputs_utils import extract_text_generation_inputs

if is_torch_distributed_available():
    import torch.distributed

LOGGER = getLogger("inference")

PER_TOKEN_BACKENDS = ["pytorch", "onnxruntime", "openvino", "neural-compressor"]

IMAGE_DIFFUSION_KWARGS = {"num_inference_steps": 30, "num_images_per_prompt": 1}
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


TEXT_GENERATION_THROUGHPUT_UNIT = "tokens/s"
IMAGE_DIFFUSION_THROUGHPUT_UNIT = "images/s"
INFERENCE_THROUGHPUT_UNIT = "samples/s"

TEXT_GENERATION_EFFICIENCY_UNIT = "tokens/kWh"
IMAGE_DIFFUSION_EFFICIENCY_UNIT = "images/kWh"
INFERENCE_EFFICIENCY_UNIT = "samples/kWh"


@dataclass
class TextGenerationReport(BenchmarkReport):
    prefill: BenchmarkMeasurements
    decode: BenchmarkMeasurements
    per_token: BenchmarkMeasurements


@dataclass
class ImageDiffusionReport(BenchmarkReport):
    call: BenchmarkMeasurements


@dataclass
class InferenceReport(BenchmarkReport):
    forward: BenchmarkMeasurements


class InferenceBenchmark(Benchmark[InferenceConfig]):
    NAME = "inference"

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT][BackendConfigT]) -> None:
        if is_torch_distributed_available() and torch.distributed.is_initialized():
            LOGGER.info("\t+ Distributing batch size across processes")
            if self.config.input_shapes["batch_size"] % torch.distributed.get_world_size() != 0:
                raise ValueError(
                    "The batch size must be divisible by the number of processes in a distributed environment"
                )
            self.config.input_shapes["batch_size"] //= torch.distributed.get_world_size()

        LOGGER.info("\t+ Creating input generator")
        self.input_generator = InputGenerator(
            task=backend.config.task, model_shapes=backend.model_shapes, input_shapes=self.config.input_shapes
        )

        if backend.config.task in TEXT_GENERATION_TASKS:
            LOGGER.info("\t+ Generating Text Generation inputs")
            self.forward_inputs = self.input_generator()
            LOGGER.info("\t+ Preparing Text Generation inputs")
            self.forward_inputs = backend.prepare_inputs(self.forward_inputs)
            self.generate_inputs = extract_text_generation_inputs(self.forward_inputs)
            LOGGER.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_KWARGS, **self.config.generate_kwargs}
            LOGGER.info("\t+ Initializing Text Generation report")
            self.report = TextGenerationReport(
                decode=BenchmarkMeasurements(), prefill=BenchmarkMeasurements(), per_token=BenchmarkMeasurements()
            )

        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            LOGGER.info("\t+ Generating Image Diffusion inputs")
            self.call_inputs = self.input_generator()
            LOGGER.info("\t+ Preparing Image Diffusion inputs")
            self.call_inputs = backend.prepare_inputs(self.call_inputs)
            LOGGER.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.call_kwargs = {**IMAGE_DIFFUSION_KWARGS, **self.config.call_kwargs}
            LOGGER.info("\t+ Initializing Image Diffusion report")
            self.report = ImageDiffusionReport(call=BenchmarkMeasurements())

        else:
            LOGGER.info("\t+ Generating Inference inputs")
            self.forward_inputs = self.input_generator()
            LOGGER.info("\t+ Preparing Inference inputs")
            self.forward_inputs = backend.prepare_inputs(self.forward_inputs)
            LOGGER.info("\t+ Initializing Inference report")
            self.report = InferenceReport(forward=BenchmarkMeasurements())

        LOGGER.info("\t+ Preparing backend for Inference")
        backend.prepare_for_inference(
            **backend.model_shapes,
            **self.config.input_shapes,
            **self.config.generate_kwargs,
            **self.config.forward_kwargs,
            **self.config.call_kwargs,
        )

        LOGGER.info("\t+ Warming up backend for Inference")
        for _ in range(self.config.warmup_runs):
            if backend.config.task in TEXT_GENERATION_TASKS:
                _ = backend.generate(self.generate_inputs, {"max_new_tokens": 2, "min_new_tokens": 2})
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                _ = backend.call(self.call_inputs, {"num_inference_steps": 2})
            else:
                _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        if backend.config.task in TEXT_GENERATION_TASKS:
            LOGGER.info("\t+ Additional warmup for Text Generation")
            _ = backend.generate(self.generate_inputs, self.config.generate_kwargs)
        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            LOGGER.info("\t+ Additional warmup for Image Diffusion")
            _ = backend.call(self.call_inputs, self.config.call_kwargs)

        if self.config.memory:
            if backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_memory_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_memory_tracking(backend)
            else:
                self.run_inference_memory_tracking(backend)

            self.report.log_memory()

        if self.config.latency:
            if backend.config.task in TEXT_GENERATION_TASKS:
                if backend.config.name in PER_TOKEN_BACKENDS:
                    self.run_fine_grained_text_generation_latency_tracking(backend)
                else:
                    self.run_text_generation_latency_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_latency_tracking(backend)
            else:
                self.run_latency_inference_tracking(backend)

            self.report.log_latency()
            self.report.log_throughput()

        if self.config.energy:
            if backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_energy_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_energy_tracking(backend)
            else:
                self.run_inference_energy_tracking(backend)

            self.report.log_energy()
            self.report.log_efficiency()

    ## Memory tracking
    def run_text_generation_memory_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running memory tracking")
        self.memory_tracker = MemoryTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )
        self.memory_tracker.reset()

        with self.memory_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.prefill.memory = self.memory_tracker.get_max_memory()

        self.memory_tracker.reset()
        with self.memory_tracker.track():
            _ = backend.generate(self.generate_inputs, self.config.generate_kwargs)

        self.report.decode.memory = self.memory_tracker.get_max_memory()

    def run_image_diffusion_memory_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running memory tracking")
        self.memory_tracker = MemoryTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )

        with self.memory_tracker.track():
            _ = backend.call(self.call_inputs, self.config.call_kwargs)

        self.report.call.memory = self.memory_tracker.get_max_memory()

    def run_inference_memory_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running memory tracking")
        self.memory_tracker = MemoryTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )

        with self.memory_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.forward.memory = self.memory_tracker.get_max_memory()

    ## Latency tracking
    def run_fine_grained_text_generation_latency_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running fine-grained Text Generation latency tracking")
        self.logits_processor = LatencyLogitsProcessor(device=backend.config.device, backend=backend.config.name)
        self.config.generate_kwargs["logits_processor"] = LogitsProcessorList(
            [self.logits_processor, *self.config.generate_kwargs.get("logits_processor", [])]
        )

        while self.logits_processor.get_elapsed_time() < self.config.duration:
            with self.logits_processor.track():
                _ = backend.generate(self.generate_inputs, self.config.generate_kwargs)

        self.report.per_token.latency = self.logits_processor.get_per_token_latency()
        self.report.prefill.latency = self.logits_processor.get_prefill_latency()
        self.report.decode.latency = self.logits_processor.get_decode_latency()

        self.report.per_token.throughput = Throughput.from_latency(
            self.report.per_token.latency, self.text_generation_per_token_volume, unit=TEXT_GENERATION_THROUGHPUT_UNIT
        )
        self.report.prefill.throughput = Throughput.from_latency(
            self.report.prefill.latency, self.text_generation_prefill_volume, unit=TEXT_GENERATION_THROUGHPUT_UNIT
        )
        self.report.decode.throughput = Throughput.from_latency(
            self.report.decode.latency, self.text_generation_decode_volume, unit=TEXT_GENERATION_THROUGHPUT_UNIT
        )

    def run_text_generation_latency_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running Text Generation latency tracking")
        self.latency_tracker = LatencyTracker(backend=backend.config.name, device=backend.config.device)

        while self.latency_tracker.get_elapsed_time() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)
        forward_latency = self.latency_tracker.get_latency()

        self.report.prefill.latency = forward_latency
        self.report.prefill.throughput = Throughput.from_latency(
            self.report.prefill.latency, self.text_generation_prefill_volume, unit=TEXT_GENERATION_THROUGHPUT_UNIT
        )

        self.latency_tracker.reset()
        while self.latency_tracker.get_elapsed_time() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.generate(self.generate_inputs, self.config.generate_kwargs)
        generate_latency = self.latency_tracker.get_latency()

        self.report.decode.latency = generate_latency - forward_latency
        self.report.decode.throughput = Throughput.from_latency(
            self.report.decode.latency, self.text_generation_decode_volume, unit=TEXT_GENERATION_THROUGHPUT_UNIT
        )

    def run_image_diffusion_latency_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running latency tracking")
        self.latency_tracker = LatencyTracker(backend=backend.config.name, device=backend.config.device)

        while self.latency_tracker.get_elapsed_time() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.call(self.call_inputs, self.config.call_kwargs)

        self.report.call.latency = self.latency_tracker.get_latency()
        self.report.call.throughput = Throughput.from_latency(
            self.report.call.latency, self.image_diffusion_volume, unit=IMAGE_DIFFUSION_THROUGHPUT_UNIT
        )

    def run_latency_inference_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running latency tracking")
        self.latency_tracker = LatencyTracker(backend=backend.config.name, device=backend.config.device)

        while self.latency_tracker.get_elapsed_time() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.forward.latency = self.latency_tracker.get_latency()
        self.report.forward.throughput = Throughput.from_latency(
            self.report.forward.latency, self.inference_volume, unit=INFERENCE_THROUGHPUT_UNIT
        )

    ## Energy tracking
    def run_text_generation_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running energy tracking")
        self.energy_tracker = EnergyTracker(device=backend.config.device, device_ids=backend.config.device_ids)

        with self.energy_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)
        forward_energy = self.energy_tracker.get_energy()

        self.report.prefill.energy = forward_energy
        self.report.prefill.efficiency = Efficiency.from_energy(
            self.report.prefill.energy, self.text_generation_prefill_volume, unit=TEXT_GENERATION_EFFICIENCY_UNIT
        )

        self.energy_tracker.reset()
        with self.energy_tracker.track():
            _ = backend.generate(self.generate_inputs, self.config.generate_kwargs)
        generate_energy = self.energy_tracker.get_energy()

        self.report.decode.energy = generate_energy - forward_energy
        self.report.decode.efficiency = Efficiency.from_energy(
            self.report.decode.energy, self.text_generation_decode_volume, unit=TEXT_GENERATION_EFFICIENCY_UNIT
        )

    def run_image_diffusion_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running energy tracking")
        self.energy_tracker = EnergyTracker(device=backend.config.device, device_ids=backend.config.device_ids)

        with self.energy_tracker.track():
            _ = backend.call(self.call_inputs, self.config.call_kwargs)

        self.report.call.energy = self.energy_tracker.get_energy()
        self.report.call.efficiency = Efficiency.from_energy(
            self.report.call.energy, self.image_diffusion_volume, unit=IMAGE_DIFFUSION_EFFICIENCY_UNIT
        )

    def run_inference_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running energy tracking")
        self.energy_tracker = EnergyTracker(device=backend.config.device, device_ids=backend.config.device_ids)

        with self.energy_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        self.report.forward.energy = self.energy_tracker.get_energy()
        self.report.forward.efficiency = Efficiency.from_energy(
            self.report.forward.energy, self.inference_volume, unit=INFERENCE_EFFICIENCY_UNIT
        )

    @property
    def inference_volume(self) -> int:  # in samples
        return self.config.input_shapes["batch_size"]

    @property
    def image_diffusion_volume(self) -> int:  # in images
        return self.config.input_shapes["batch_size"] * self.config.call_kwargs["num_images_per_prompt"]

    @property
    def text_generation_prefill_volume(self) -> int:  # in tokens
        return self.config.input_shapes["batch_size"] * self.config.input_shapes["sequence_length"]

    @property
    def text_generation_per_token_volume(self) -> int:  # in tokens
        return self.config.input_shapes["batch_size"] * self.config.generate_kwargs["num_return_sequences"]

    @property
    def text_generation_decode_volume(self) -> int:  # in tokens
        return (
            self.config.input_shapes["batch_size"]
            * self.config.generate_kwargs["num_return_sequences"]
            * (self.config.generate_kwargs["max_new_tokens"] - 1)  # 1 token is generated during prefill
        )

    def get_report(self) -> InferenceReport:
        return self.report
