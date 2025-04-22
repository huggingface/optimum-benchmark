import time
from contextlib import ExitStack, contextmanager

from transformers import LogitsProcessorList

from ...backends.base import Backend, BackendConfigT
from ...benchmark.report import BenchmarkReport
from ...generators.input_generator import InputGenerator
from ...task_utils import IMAGE_DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ...trackers.energy import Efficiency, EnergyTracker
from ...trackers.latency import (
    LatencySessionTracker,
    PerStepLatencySessionTrackerPipelineCallback,
    PerTokenLatencySessionTrackerLogitsProcessor,
    Throughput,
)
from ...trackers.memory import MemoryTracker
from ..base import Scenario
from .config import InferenceConfig

PER_TOKEN_BACKENDS = ["pytorch", "onnxruntime", "openvino", "neural-compressor", "ipex"]

TEXT_GENERATION_DEFAULT_KWARGS = {
    "num_return_sequences": 1,
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "do_sample": False,
    "use_cache": True,
    "num_beams": 1,
}
TEXT_GENERATION_PREFILL_OVERRIDES = {
    "max_new_tokens": 1,
    "min_new_tokens": 1,
}
TEXT_GENERATION_WARMUP_OVERRIDES = {
    "max_new_tokens": 2,
    "min_new_tokens": 2,
}

IMAGE_DIFFUSION_DEFAULT_KWARGS = {
    "num_inference_steps": 30,
    "num_images_per_prompt": 1,
}
IMAGE_DIFFUSION_WARMUP_OVERRIDES = {
    "num_inference_steps": 2,
}

GENERATE_THROUGHPUT_UNIT = "samples/s"
FORWARD_THROUGHPUT_UNIT = "samples/s"
PREFILL_THROUGHPUT_UNIT = "samples/s"
DECODE_THROUGHPUT_UNIT = "tokens/s"
CALL_THROUGHPUT_UNIT = "images/s"

GENERATE_EFFICIENCY_UNIT = "samples/kWh"
FORWARD_EFFICIENCY_UNIT = "samples/kWh"
PREFILL_EFFICIENCY_UNIT = "samples/kWh"
DECODE_EFFICIENCY_UNIT = "tokens/kWh"
CALL_EFFICIENCY_UNIT = "images/kWh"


class InferenceScenario(Scenario[InferenceConfig]):
    NAME = "inference"

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT]) -> BenchmarkReport:
        self.backend = backend

        if self.backend.config.task in TEXT_GENERATION_TASKS:
            self.logger.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_DEFAULT_KWARGS, **self.config.generate_kwargs}
        elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
            self.logger.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.call_kwargs = {**IMAGE_DIFFUSION_DEFAULT_KWARGS, **self.config.call_kwargs}

        if self.backend.config.task in TEXT_GENERATION_TASKS:
            self.logger.info("\t+ Initializing Text Generation targets list")
            targets = ["load_model", "first_generate", "generate", "prefill", "decode"]
            if self.backend.config.name in PER_TOKEN_BACKENDS:
                targets.append("per_token")
        elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
            self.logger.info("\t+ Initializing Image Diffusion targets list")
            targets = ["load_model", "first_call", "call", "per_step"]
        else:
            self.logger.info("\t+ Initializing Inference targets list")
            targets = ["load_model", "first_forward", "forward"]

        self.report = BenchmarkReport.from_list(targets=targets)

        if self.config.latency:
            self.logger.info("\t+ Initializing Latency tracker")
            self.latency_tracker = LatencySessionTracker(
                device=self.backend.config.device, backend=self.backend.config.name
            )
            if self.backend.config.task in TEXT_GENERATION_TASKS and self.backend.config.name in PER_TOKEN_BACKENDS:
                self.logger.info("\t+ Initializing Per-Token Latency tracker")
                self.per_token_latency_tracker = PerTokenLatencySessionTrackerLogitsProcessor(
                    device=self.backend.config.device, backend=self.backend.config.name
                )
                self.config.generate_kwargs["logits_processor"] = LogitsProcessorList([self.per_token_latency_tracker])
            elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.logger.info("\t+ Initializing Diffusion Step Latency tracker")
                self.per_step_latency_tracker = PerStepLatencySessionTrackerPipelineCallback(
                    device=self.backend.config.device, backend=self.backend.config.name
                )
                self.config.call_kwargs["callback_on_step_end"] = self.per_step_latency_tracker

        if self.config.memory:
            self.logger.info("\t+ Initializing Memory tracker")
            self.memory_tracker = MemoryTracker(
                backend=self.backend.config.name,
                device=self.backend.config.device,
                device_ids=self.backend.config.device_ids,
            )

        if self.config.energy:
            self.logger.info("\t+ Initializing Energy tracker")
            self.energy_tracker = EnergyTracker(
                backend=self.backend.config.name,
                device=self.backend.config.device,
                device_ids=self.backend.config.device_ids,
            )

        self.logger.info(f"\t+ Generating inputs for task {self.backend.config.task}")
        self.inputs = InputGenerator(
            task=self.backend.config.task,
            model_shapes=self.backend.model_shapes,
            model_type=self.backend.config.model_type,
            input_shapes=self.config.input_shapes,
        )()

        self.run_model_loading_tracking()

        self.logger.info(f"\t+ Preparing inputs for backend {self.backend.config.name}")
        self.inputs = self.backend.prepare_inputs(inputs=self.inputs)

        if self.config.warmup_runs > 0:
            if self.backend.config.task in TEXT_GENERATION_TASKS:
                self.warmup_text_generation()
            elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.warmup_image_diffusion()
            else:
                self.warmup_inference()

        if self.config.latency:
            if self.backend.config.task in TEXT_GENERATION_TASKS:
                if self.backend.config.name in PER_TOKEN_BACKENDS:
                    self.run_per_token_text_generation_latency_tracking()
                else:
                    self.run_text_generation_latency_tracking()
            elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_latency_tracking()
            else:
                self.run_inference_latency_tracking()

        if self.config.memory:
            if self.backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_memory_tracking()
            elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_memory_tracking()
            else:
                self.run_inference_memory_tracking()

        if self.config.energy:
            if self.backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_energy_tracking()
            elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_energy_tracking()
            else:
                self.run_inference_energy_tracking()

        return self.report

    @contextmanager
    def global_tracking(self, target: str):
        with ExitStack() as context_stack:
            if self.config.energy:
                context_stack.enter_context(self.energy_tracker.track(task_name=target))
            if self.config.memory:
                context_stack.enter_context(self.memory_tracker.track())
            if self.config.latency:
                context_stack.enter_context(self.latency_tracker.session())
                context_stack.enter_context(self.latency_tracker.track())

            yield

        if self.config.latency:
            getattr(self.report, target).latency = self.latency_tracker.get_latency()
        if self.config.memory:
            getattr(self.report, target).memory = self.memory_tracker.get_max_memory()
        if self.config.energy:
            getattr(self.report, target).energy = self.energy_tracker.get_energy()

    # Model loading
    def run_model_loading_tracking(self):
        self.logger.info("\t+ Running model loading tracking")
        with self.global_tracking(target="load_model"):
            self.backend.load()

    # Warmup & cold start
    def warmup_text_generation(self):
        self.logger.info("\t+ Warming up backend for Text Generation")
        self.backend.generate(self.inputs, self.config.generate_kwargs)

        with self.global_tracking(target="first_generate"):
            self.backend.generate(self.inputs, self.config.generate_kwargs)

        for _ in range(self.config.warmup_runs):
            self.backend.generate(self.inputs, {**self.config.generate_kwargs, **TEXT_GENERATION_WARMUP_OVERRIDES})

    def warmup_image_diffusion(self):
        self.logger.info("\t+ Warming up backend for Image Diffusion")

        with self.global_tracking(target="first_call"):
            self.backend.call(self.inputs, self.config.call_kwargs)

        for _ in range(self.config.warmup_runs):
            self.backend.call(self.inputs, {**self.config.call_kwargs, **IMAGE_DIFFUSION_WARMUP_OVERRIDES})

    def warmup_inference(self):
        self.logger.info("\t+ Warming up backend for Inference")

        with self.global_tracking(target="first_forward"):
            self.backend.forward(self.inputs, self.config.forward_kwargs)

        for _ in range(self.config.warmup_runs):
            self.backend.forward(self.inputs, self.config.forward_kwargs)

    # Memory tracking
    def run_text_generation_memory_tracking(self):
        prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}

        self.logger.info("\t+ Running Text Generation memory tracking")

        with self.memory_tracker.track():
            self.backend.prefill(self.inputs, prefill_kwargs)

        self.report.prefill.memory = self.memory_tracker.get_max_memory()

        with self.memory_tracker.track():
            self.backend.generate(self.inputs, self.config.generate_kwargs)

        self.report.decode.memory = self.memory_tracker.get_max_memory()

    def run_image_diffusion_memory_tracking(self):
        self.logger.info("\t+ Running Image Diffusion memory tracking")

        with self.memory_tracker.track():
            self.backend.call(self.inputs, self.config.call_kwargs)

        self.report.call.memory = self.memory_tracker.get_max_memory()

    def run_inference_memory_tracking(self):
        self.logger.info("\t+ Running Inference memory tracking")

        with self.memory_tracker.track():
            self.backend.forward(self.inputs, self.config.forward_kwargs)

        self.report.forward.memory = self.memory_tracker.get_max_memory()

    # Latency tracking
    def run_per_token_text_generation_latency_tracking(self):
        self.logger.info("\t+ Running Per-Token Text Generation latency tracking")

        with self.per_token_latency_tracker.session():
            while (
                self.per_token_latency_tracker.elapsed() < self.config.duration
                or self.per_token_latency_tracker.count() < self.config.iterations
            ):
                with self.per_token_latency_tracker.track():
                    self.backend.generate(self.inputs, self.config.generate_kwargs)

        per_token_latency = self.per_token_latency_tracker.get_per_token_latency()
        generate_latency = self.per_token_latency_tracker.get_generate_latency()
        prefill_latency = self.per_token_latency_tracker.get_prefill_latency()
        decode_latency = self.per_token_latency_tracker.get_decode_latency()

        self.report.per_token.latency = per_token_latency
        self.report.generate.latency = generate_latency
        self.report.prefill.latency = prefill_latency
        self.report.decode.latency = decode_latency

        # we don't register a per-token throughput,
        # it's a confusing metric and the same signal as the decode throughput
        self.report.generate.throughput = Throughput.from_latency(
            generate_latency, self.atomic_decode_volume, unit=GENERATE_THROUGHPUT_UNIT
        )
        self.report.prefill.throughput = Throughput.from_latency(
            prefill_latency, self.atomic_prefill_volume, unit=PREFILL_THROUGHPUT_UNIT
        )
        self.report.decode.throughput = Throughput.from_latency(
            decode_latency, self.atomic_decode_volume, unit=DECODE_THROUGHPUT_UNIT
        )

    def run_text_generation_latency_tracking(self):
        self.logger.info("\t+ Running Text Generation latency tracking")

        prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}

        with self.latency_tracker.session():
            while (
                self.latency_tracker.elapsed() < self.config.duration
                or self.latency_tracker.count() < self.config.iterations
            ):
                with self.latency_tracker.track():
                    self.backend.prefill(self.inputs, prefill_kwargs)

        prefill_latency = self.latency_tracker.get_latency()

        self.report.prefill.latency = prefill_latency
        self.report.prefill.throughput = Throughput.from_latency(
            prefill_latency, self.atomic_prefill_volume, unit=PREFILL_THROUGHPUT_UNIT
        )

        with self.latency_tracker.session():
            while (
                self.latency_tracker.elapsed() < self.config.duration
                or self.latency_tracker.count() < self.config.iterations
            ):
                with self.latency_tracker.track():
                    self.backend.generate(self.inputs, self.config.generate_kwargs)

        generate_latency = self.latency_tracker.get_latency()
        self.report.generate.latency = generate_latency
        self.report.generate.throughput = Throughput.from_latency(
            generate_latency, self.atomic_generate_volume, unit=GENERATE_THROUGHPUT_UNIT
        )

        decode_latency = generate_latency - prefill_latency
        self.report.decode.latency = decode_latency
        self.report.decode.throughput = Throughput.from_latency(
            decode_latency, self.atomic_decode_volume, unit=DECODE_THROUGHPUT_UNIT
        )

    def run_image_diffusion_latency_tracking(self):
        self.logger.info("\t+ Running Image Diffusion latency tracking")

        with self.per_step_latency_tracker.session():
            while (
                self.per_step_latency_tracker.elapsed() < self.config.duration
                or self.per_step_latency_tracker.count() < self.config.iterations
            ):
                with self.per_step_latency_tracker.track():
                    self.backend.call(self.inputs, self.config.call_kwargs)

        call_latency = self.per_step_latency_tracker.get_call_latency()
        self.report.call.latency = call_latency
        self.report.call.throughput = Throughput.from_latency(
            call_latency, self.atomic_call_volume, unit=CALL_THROUGHPUT_UNIT
        )

        per_step_latency = self.per_step_latency_tracker.get_step_latency()
        self.report.per_step.latency = per_step_latency

    def run_inference_latency_tracking(self):
        self.logger.info("\t+ Running Inference latency tracking")

        with self.latency_tracker.session():
            while (
                self.latency_tracker.elapsed() < self.config.duration
                or self.latency_tracker.count() < self.config.iterations
            ):
                with self.latency_tracker.track():
                    self.backend.forward(self.inputs, self.config.forward_kwargs)

        forward_latency = self.latency_tracker.get_latency()
        self.report.forward.latency = forward_latency
        self.report.forward.throughput = Throughput.from_latency(
            forward_latency, self.atomic_forward_volume, unit=FORWARD_THROUGHPUT_UNIT
        )

    # Energy tracking
    def run_text_generation_energy_tracking(self):
        self.logger.info("\t+ Running Text Generation energy tracking")
        prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}

        count, elapsed, start_time = 0, 0, time.perf_counter()

        with self.energy_tracker.track(task_name="prefill"):
            while elapsed < self.config.duration or count < self.config.iterations:
                self.backend.prefill(self.inputs, prefill_kwargs)
                elapsed = time.perf_counter() - start_time
                count += 1

        prefill_energy = self.energy_tracker.get_energy() / count

        self.report.prefill.energy = prefill_energy
        self.report.prefill.efficiency = Efficiency.from_energy(
            prefill_energy, self.atomic_prefill_volume, unit=PREFILL_EFFICIENCY_UNIT
        )

        count, elapsed, start_time = 0, 0, time.perf_counter()

        with self.energy_tracker.track(task_name="generate"):
            while elapsed < self.config.duration or count < self.config.iterations:
                self.backend.generate(self.inputs, self.config.generate_kwargs)
                elapsed = time.perf_counter() - start_time
                count += 1

        generate_energy = self.energy_tracker.get_energy() / count
        self.report.generate.energy = generate_energy
        self.report.generate.efficiency = Efficiency.from_energy(
            generate_energy, self.atomic_generate_volume, unit=GENERATE_EFFICIENCY_UNIT
        )

        decode_energy = generate_energy - prefill_energy
        self.report.decode.energy = decode_energy
        self.report.decode.efficiency = Efficiency.from_energy(
            decode_energy, self.atomic_decode_volume, unit=DECODE_EFFICIENCY_UNIT
        )

    def run_image_diffusion_energy_tracking(self):
        self.logger.info("\t+ Running Image Diffusion energy tracking")

        count = 0
        elapsed = 0
        start_time = time.perf_counter()

        with self.energy_tracker.track(task_name="call"):
            while elapsed < self.config.duration or count < self.config.iterations:
                self.backend.call(self.inputs, self.config.call_kwargs)
                elapsed = time.perf_counter() - start_time
                count += 1

        call_energy = self.energy_tracker.get_energy() / count

        self.report.call.energy = call_energy
        self.report.call.efficiency = Efficiency.from_energy(
            call_energy, self.atomic_call_volume, unit=CALL_EFFICIENCY_UNIT
        )

    def run_inference_energy_tracking(self):
        self.logger.info("\t+ Running energy tracking")

        count = 0
        elapsed = 0
        start_time = time.perf_counter()

        with self.energy_tracker.track(task_name="forward"):
            while elapsed < self.config.duration or count < self.config.iterations:
                self.backend.forward(self.inputs, self.config.forward_kwargs)
                elapsed = time.perf_counter() - start_time
                count += 1

        forward_energy = self.energy_tracker.get_energy() / count

        self.report.forward.energy = forward_energy
        self.report.forward.efficiency = Efficiency.from_energy(
            forward_energy, self.atomic_forward_volume, unit=FORWARD_EFFICIENCY_UNIT
        )

    @property
    def atomic_forward_volume(self) -> int:  # in terms of processed samples
        return self.config.input_shapes["batch_size"]

    @property
    def atomic_generate_volume(self) -> int:  # in terms of processed samples
        return self.config.input_shapes["batch_size"]

    @property
    def atomic_prefill_volume(self) -> int:  # in terms of processed samples
        return self.config.input_shapes["batch_size"]

    @property
    def atomic_decode_volume(self) -> int:  # in terms of generated tokens
        return (
            self.config.input_shapes["batch_size"]
            * self.config.generate_kwargs["num_beams"]  # at each beam stage there are num_beams tokens generated
            * (self.config.generate_kwargs["max_new_tokens"] - 1)  # 1 token is generated during prefill
        )

    @property
    def atomic_call_volume(self) -> int:  # in terms of generated images
        if self.backend.config.task == "text-to-image":
            return self.config.input_shapes["batch_size"] * self.config.call_kwargs["num_images_per_prompt"]
        else:
            return self.config.input_shapes["batch_size"]
