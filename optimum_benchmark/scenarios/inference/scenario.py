import time
from contextlib import ExitStack

from transformers import LogitsProcessorList

from ...backends.base import Backend, BackendConfigT
from ...benchmark.report import BenchmarkReport
from ...generators.input_generator import InputGenerator
from ...task_utils import IMAGE_DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ...trackers.energy import Efficiency, EnergyTracker
from ...trackers.latency import LatencyTracker, PerTokenLatencyLogitsProcessor, Throughput
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


FORWARD_THROUGHPUT_UNIT = "samples/s"
PREFILL_THROUGHPUT_UNIT = "samples/s"
DECODE_THROUGHPUT_UNIT = "tokens/s"
CALL_THROUGHPUT_UNIT = "images/s"


FORWARD_EFFICIENCY_UNIT = "samples/kWh"
PREFILL_EFFICIENCY_UNIT = "samples/kWh"
DECODE_EFFICIENCY_UNIT = "tokens/kWh"
CALL_EFFICIENCY_UNIT = "images/kWh"


class InferenceScenario(Scenario[InferenceConfig]):
    NAME = "inference"

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT]) -> BenchmarkReport:
        self.task = backend.config.task

        if backend.config.task in TEXT_GENERATION_TASKS:
            self.logger.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_DEFAULT_KWARGS, **self.config.generate_kwargs}
            self.logger.info("\t+ Initializing Text Generation report")
            self.report = BenchmarkReport.from_list(targets=["load_model", "prefill", "decode", "per_token"])
        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            self.logger.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.call_kwargs = {**IMAGE_DIFFUSION_DEFAULT_KWARGS, **self.config.call_kwargs}
            self.logger.info("\t+ Initializing Image Diffusion report")
            self.report = BenchmarkReport.from_list(targets=["load_model", "call"])
        else:
            self.logger.info("\t+ Updating Inference kwargs with default values")
            self.forward_kwargs = {**self.config.forward_kwargs}
            self.logger.info("\t+ Initializing Inference report")
            self.report = BenchmarkReport.from_list(targets=["load_model", "forward"])

        self.run_model_loading_tracking(backend)

        self.logger.info("\t+ Creating input generator")
        self.input_generator = InputGenerator(
            task=backend.config.task,
            model_shapes=backend.model_shapes,
            input_shapes=self.config.input_shapes,
            model_type=backend.config.model_type,
        )
        self.logger.info("\t+ Generating inputs")
        self.inputs = self.input_generator()
        self.logger.info("\t+ Preparing inputs for backend")
        self.inputs = backend.prepare_inputs(inputs=self.inputs)

        if self.config.warmup_runs > 0:
            if backend.config.task in TEXT_GENERATION_TASKS:
                self.warmup_text_generation(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.warmup_image_diffusion(backend)
            else:
                self.warmup_inference(backend)

        if self.config.latency:
            if backend.config.task in TEXT_GENERATION_TASKS:
                if backend.config.name in PER_TOKEN_BACKENDS:
                    self.run_per_token_text_generation_latency_tracking(backend)
                else:
                    self.run_text_generation_latency_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_latency_tracking(backend)
            else:
                self.run_latency_inference_tracking(backend)

        if self.config.memory:
            if backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_memory_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_memory_tracking(backend)
            else:
                self.run_inference_memory_tracking(backend)

        if self.config.energy:
            if backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_energy_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_energy_tracking(backend)
            else:
                self.run_inference_energy_tracking(backend)

        return self.report

    # Model loading tracking
    def run_model_loading_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running model loading tracking")

        if self.config.memory:
            memory_tracker = MemoryTracker(
                backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
            )
        if self.config.latency:
            latency_tracker = LatencyTracker(backend=backend.config.name, device=backend.config.device)
        if self.config.energy:
            energy_tracker = EnergyTracker(
                backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
            )

        with ExitStack() as context_stack:
            if self.config.energy:
                context_stack.enter_context(energy_tracker.track(task_name="load_model"))
            if self.config.memory:
                context_stack.enter_context(memory_tracker.track())
            if self.config.latency:
                context_stack.enter_context(latency_tracker.track())

            backend.load()

        if self.config.latency:
            self.report.load_model.latency = latency_tracker.get_latency()
        if self.config.memory:
            self.report.load_model.memory = memory_tracker.get_max_memory()
        if self.config.energy:
            self.report.load_model.energy = energy_tracker.get_energy()

    # Warmup
    def warmup_text_generation(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Warming up backend for Text Generation")
        _ = backend.generate(self.inputs, self.config.generate_kwargs)
        for _ in range(self.config.warmup_runs):
            _ = backend.generate(self.inputs, {**self.config.generate_kwargs, **TEXT_GENERATION_WARMUP_OVERRIDES})

    def warmup_image_diffusion(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Warming up backend for Image Diffusion")
        _ = backend.call(self.inputs, self.config.call_kwargs)
        for _ in range(self.config.warmup_runs):
            _ = backend.call(self.inputs, {**self.config.call_kwargs, **IMAGE_DIFFUSION_WARMUP_OVERRIDES})

    def warmup_inference(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Warming up backend for Inference")
        for _ in range(self.config.warmup_runs):
            _ = backend.forward(self.inputs, self.config.forward_kwargs)

    ## Memory tracking
    def run_text_generation_memory_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Text Generation memory tracking")
        memory_tracker = MemoryTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )
        prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}

        with memory_tracker.track():
            _ = backend.prefill(self.inputs, prefill_kwargs)

        self.report.prefill.memory = memory_tracker.get_max_memory()

        with memory_tracker.track():
            _ = backend.generate(self.inputs, self.config.generate_kwargs)

        self.report.decode.memory = memory_tracker.get_max_memory()

    def run_image_diffusion_memory_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Image Diffusion memory tracking")
        memory_tracker = MemoryTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )

        with memory_tracker.track():
            _ = backend.call(self.inputs, self.config.call_kwargs)

        self.report.call.memory = memory_tracker.get_max_memory()

    def run_inference_memory_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Inference memory tracking")
        memory_tracker = MemoryTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )

        with memory_tracker.track():
            _ = backend.forward(self.inputs, self.config.forward_kwargs)

        self.report.forward.memory = memory_tracker.get_max_memory()

    ## Latency tracking
    def run_per_token_text_generation_latency_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Per-Token Text Generation latency tracking")
        latency_tracker = PerTokenLatencyLogitsProcessor(device=backend.config.device, backend=backend.config.name)
        per_token_kwargs = {**self.config.generate_kwargs, "logits_processor": LogitsProcessorList([latency_tracker])}

        while latency_tracker.elapsed() < self.config.duration or latency_tracker.count() < self.config.iterations:
            with latency_tracker.track():
                _ = backend.generate(self.inputs, per_token_kwargs)

        per_token_latency = latency_tracker.get_per_token_latency()
        prefill_latency = latency_tracker.get_prefill_latency()
        decode_latency = latency_tracker.get_decode_latency()

        prefill_volume = self.atomic_prefill_volume
        decode_volume = self.atomic_decode_volume

        self.report.per_token.latency = per_token_latency
        self.report.prefill.latency = prefill_latency
        self.report.decode.latency = decode_latency

        # we don't register a per-token throughput, as it's a confusing metric and the same as the decode throughput
        self.report.prefill.throughput = Throughput.from_latency(
            prefill_latency, prefill_volume, unit=PREFILL_THROUGHPUT_UNIT
        )
        self.report.decode.throughput = Throughput.from_latency(
            decode_latency, decode_volume, unit=DECODE_THROUGHPUT_UNIT
        )

    def run_text_generation_latency_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Text Generation latency tracking")
        latency_tracker = LatencyTracker(backend=backend.config.name, device=backend.config.device)
        prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}

        while latency_tracker.elapsed() < self.config.duration or latency_tracker.count() < self.config.iterations:
            with latency_tracker.track():
                _ = backend.prefill(self.inputs, prefill_kwargs)

        prefill_latency = latency_tracker.get_latency()
        prefill_volume = self.atomic_prefill_volume

        self.report.prefill.latency = prefill_latency
        self.report.prefill.throughput = Throughput.from_latency(
            prefill_latency, prefill_volume, unit=PREFILL_THROUGHPUT_UNIT
        )

        latency_tracker.reset()
        while latency_tracker.elapsed() < self.config.duration or latency_tracker.count() < self.config.iterations:
            with latency_tracker.track():
                _ = backend.generate(self.inputs, self.config.generate_kwargs)

        generate_latency = latency_tracker.get_latency()
        decode_latency = generate_latency - prefill_latency
        decode_volume = self.atomic_decode_volume

        self.report.decode.latency = decode_latency
        self.report.decode.throughput = Throughput.from_latency(
            decode_latency, decode_volume, unit=DECODE_THROUGHPUT_UNIT
        )

    def run_image_diffusion_latency_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Image Diffusion latency tracking")
        latency_tracker = LatencyTracker(backend=backend.config.name, device=backend.config.device)

        while latency_tracker.elapsed() < self.config.duration or latency_tracker.count() < self.config.iterations:
            with latency_tracker.track():
                _ = backend.call(self.inputs, self.config.call_kwargs)

        call_latency = latency_tracker.get_latency()
        call_volume = self.atomic_call_volume

        self.report.call.latency = call_latency
        self.report.call.throughput = Throughput.from_latency(call_latency, call_volume, unit=CALL_THROUGHPUT_UNIT)

    def run_latency_inference_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Inference latency tracking")
        latency_tracker = LatencyTracker(backend=backend.config.name, device=backend.config.device)

        while latency_tracker.elapsed() < self.config.duration or latency_tracker.count() < self.config.iterations:
            with latency_tracker.track():
                _ = backend.forward(self.inputs, self.config.forward_kwargs)

        forward_latency = latency_tracker.get_latency()
        forward_volume = self.atomic_forward_volume

        self.report.forward.latency = forward_latency
        self.report.forward.throughput = Throughput.from_latency(
            forward_latency, forward_volume, unit=FORWARD_THROUGHPUT_UNIT
        )

    ## Energy tracking
    def run_text_generation_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Text Generation energy tracking")
        energy_tracker = EnergyTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )
        prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}

        count = 0
        elapsed = 0
        start_time = time.perf_counter()

        with energy_tracker.track(task_name="prefill"):
            while elapsed < self.config.duration or count < self.config.iterations:
                _ = backend.prefill(self.inputs, prefill_kwargs)
                elapsed = time.perf_counter() - start_time
                count += 1

        prefill_energy = energy_tracker.get_energy() / count
        prefill_volume = self.atomic_prefill_volume

        self.report.prefill.energy = prefill_energy
        self.report.prefill.efficiency = Efficiency.from_energy(
            prefill_energy, prefill_volume, unit=PREFILL_EFFICIENCY_UNIT
        )

        count = 0
        elapsed = 0
        start_time = time.perf_counter()

        with energy_tracker.track(task_name="generate"):
            while elapsed < self.config.duration or count < self.config.iterations:
                _ = backend.generate(self.inputs, self.config.generate_kwargs)
                elapsed = time.perf_counter() - start_time
                count += 1

        generate_energy = energy_tracker.get_energy() / count
        decode_energy = generate_energy - prefill_energy
        decode_volume = self.atomic_decode_volume

        self.report.decode.energy = decode_energy
        self.report.decode.efficiency = Efficiency.from_energy(
            decode_energy, decode_volume, unit=DECODE_EFFICIENCY_UNIT
        )

    def run_image_diffusion_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Image Diffusion energy tracking")
        energy_tracker = EnergyTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )

        count = 0
        elapsed = 0
        start_time = time.perf_counter()

        with energy_tracker.track(task_name="call"):
            while elapsed < self.config.duration or count < self.config.iterations:
                _ = backend.call(self.inputs, self.config.call_kwargs)
                elapsed = time.perf_counter() - start_time
                count += 1

        call_energy = energy_tracker.get_energy() / count
        call_volume = self.atomic_call_volume

        self.report.call.energy = call_energy
        self.report.call.efficiency = Efficiency.from_energy(call_energy, call_volume, unit=CALL_EFFICIENCY_UNIT)

    def run_inference_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running energy tracking")
        energy_tracker = EnergyTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )

        count = 0
        elapsed = 0
        start_time = time.perf_counter()

        with energy_tracker.track(task_name="forward"):
            while elapsed < self.config.duration or count < self.config.iterations:
                _ = backend.forward(self.inputs, self.config.forward_kwargs)
                elapsed = time.perf_counter() - start_time
                count += 1

        forward_energy = energy_tracker.get_energy() / count
        forward_volume = self.atomic_forward_volume

        self.report.forward.energy = forward_energy
        self.report.forward.efficiency = Efficiency.from_energy(
            forward_energy, forward_volume, unit=FORWARD_EFFICIENCY_UNIT
        )

    @property
    def atomic_forward_volume(self) -> int:  # in terms of processed samples
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
        if self.task == "text-to-image":
            return self.config.input_shapes["batch_size"] * self.config.call_kwargs["num_images_per_prompt"]
        else:
            return self.config.input_shapes["batch_size"]
