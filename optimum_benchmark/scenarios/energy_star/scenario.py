from contextlib import ExitStack, contextmanager

from datasets import load_dataset
from tqdm import tqdm

from ...backends.base import Backend, BackendConfigT
from ...benchmark.report import BenchmarkReport
from ...preprocessors.dataset_preprocessor import TASKS_TO_PREPROCESSORS
from ...task_utils import IMAGE_DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ...trackers.energy import Efficiency, EnergyTracker
from ...trackers.latency import LatencyTracker, Throughput
from ...trackers.memory import MemoryTracker
from ..base import Scenario
from .config import EnergyStarConfig

TEXT_GENERATION_DEFAULT_KWARGS = {
    "num_return_sequences": 1,
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "temperature": 1.0,
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


PREPROCESS_EFFICIENCY_UNIT = "samples/kWh"
FORWARD_EFFICIENCY_UNIT = "samples/kWh"
PREFILL_EFFICIENCY_UNIT = "samples/kWh"
DECODE_EFFICIENCY_UNIT = "tokens/kWh"
CALL_EFFICIENCY_UNIT = "images/kWh"

PREPROCESS_THROUGHPUT_UNIT = "samples/s"
FORWARD_THROUGHPUT_UNIT = "samples/s"
PREFILL_THROUGHPUT_UNIT = "samples/s"
DECODE_THROUGHPUT_UNIT = "tokens/s"
CALL_THROUGHPUT_UNIT = "images/s"


class EnergyStarScenario(Scenario[EnergyStarConfig]):
    NAME = "energy-star"

    def __init__(self, config: EnergyStarConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT]) -> BenchmarkReport:
        self.task = backend.config.task

        if backend.config.task in TEXT_GENERATION_TASKS:
            self.logger.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_DEFAULT_KWARGS, **self.config.generate_kwargs}
            self.logger.info("\t+ Initializing Text Generation report")
            self.report = BenchmarkReport.from_list(
                targets=["load_dataset", "preprocess_dataset", "load_model", "prefill", "decode"]
            )
        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            self.logger.info("\t+ Updating Image Diffusion kwargs with default values")
            self.call_kwargs = {**IMAGE_DIFFUSION_DEFAULT_KWARGS, **self.config.call_kwargs}
            self.logger.info("\t+ Initializing Image Diffusion report")
            self.report = BenchmarkReport.from_list(
                targets=["load_dataset", "preprocess_dataset", "load_model", "call"]
            )
        else:
            self.logger.info("\t+ Updating Inference kwargs with default values")
            self.forward_kwargs = {**self.config.forward_kwargs}
            self.logger.info("\t+ Initializing Inference report")
            self.report = BenchmarkReport.from_list(
                targets=["load_dataset", "preprocess_dataset", "load_model", "forward"]
            )

        self.init_trackers(backend)
        self.run_model_loading_tracking(backend)
        self.run_dataset_loading_tracking(backend)
        self.run_dataset_preprocessing_tracking(backend)

        self.logger.info("\t+ Preparing sample inputs for model warmup")
        self.sample_inputs = self.dataset[: self.config.input_shapes["batch_size"]]
        self.sample_inputs = backend.prepare_inputs(self.sample_inputs)

        if backend.config.task in TEXT_GENERATION_TASKS:
            if self.config.warmup_runs > 0:
                self.warmup_text_generation(backend)
            self.run_text_generation_tracking(backend)
        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            if self.config.warmup_runs > 0:
                self.warmup_image_diffusion(backend)
            self.run_image_diffusion_tracking(backend)
        else:
            if self.config.warmup_runs > 0:
                self.warmup_inference(backend)
            self.run_inference_tracking(backend)

        return self.report

    def init_trackers(self, backend: Backend[BackendConfigT]):
        if self.config.latency:
            self.latency_tracker = LatencyTracker(
                backend=backend.config.name,
                device=backend.config.device,
            )
        if self.config.memory:
            self.memory_tracker = MemoryTracker(
                backend=backend.config.name,
                device=backend.config.device,
                device_ids=backend.config.device_ids,
            )
        if self.config.energy:
            self.energy_tracker = EnergyTracker(
                backend=backend.config.name,
                device=backend.config.device,
                device_ids=backend.config.device_ids,
            )

    @contextmanager
    def track(self, task_name: str):
        with ExitStack() as context_stack:
            if self.config.energy:
                context_stack.enter_context(self.energy_tracker.track(task_name=task_name))
            if self.config.memory:
                context_stack.enter_context(self.memory_tracker.track())
            if self.config.latency:
                context_stack.enter_context(self.latency_tracker.track())
            yield

    def reset_trackers(self):
        if self.config.latency:
            self.latency_tracker.reset()
        if self.config.memory:
            self.memory_tracker.reset()
        if self.config.energy:
            self.energy_tracker.reset()

    # Dataset loading tracking
    def run_dataset_loading_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running dataset loading tracking")

        self.reset_trackers()
        with self.track(task_name="load_dataset"):
            self.dataset = load_dataset(
                self.config.dataset_name, self.config.dataset_config, split=self.config.dataset_split
            )

        if self.config.energy:
            self.report.load_dataset.energy = self.energy_tracker.get_energy()
        if self.config.latency:
            self.report.load_dataset.latency = self.latency_tracker.get_latency()
        if self.config.memory:
            self.report.load_dataset.memory = self.memory_tracker.get_max_memory()

    # Dataset preprocessing tracking
    def run_dataset_preprocessing_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running dataset preprocessing tracking")

        self.reset_trackers()
        with self.track(task_name="preprocess_dataset"):
            self.dataset = TASKS_TO_PREPROCESSORS[backend.config.task](
                dataset=self.dataset,
                scenario_config=self.config,
                pretrained_config=backend.pretrained_config,
                pretrained_processor=backend.pretrained_processor,
            )

        if self.config.energy:
            preprocess_energy = self.energy_tracker.get_energy()
            preprocess_volume = self.dataset_preprocess_volume
            self.report.preprocess_dataset.energy = preprocess_energy
            self.report.preprocess_dataset.efficiency = Efficiency.from_energy(
                preprocess_energy, preprocess_volume, unit=PREPROCESS_EFFICIENCY_UNIT
            )
        if self.config.latency:
            preprocess_latency = self.latency_tracker.get_latency()
            preprocess_volume = self.dataset_preprocess_volume
            self.report.preprocess_dataset.latency = preprocess_latency
            self.report.preprocess_dataset.throughput = Throughput.from_latency(
                preprocess_latency, preprocess_volume, unit=PREPROCESS_THROUGHPUT_UNIT
            )
        if self.config.memory:
            self.report.preprocess_dataset.memory = self.memory_tracker.get_max_memory()

    # Model loading tracking
    def run_model_loading_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running model loading energy tracking")

        self.reset_trackers()
        with self.track(task_name="load_model"):
            backend.load()

        if self.config.latency:
            self.report.load_model.latency = self.latency_tracker.get_latency()
        if self.config.memory:
            self.report.load_model.memory = self.memory_tracker.get_max_memory()
        if self.config.energy:
            self.report.load_model.energy = self.energy_tracker.get_energy()

    # Text Generation warmup
    def warmup_text_generation(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Warming up backend for Text Generation")
        backend.generate(self.sample_inputs, self.config.generate_kwargs)
        warmup_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_WARMUP_OVERRIDES}
        for _ in range(self.config.warmup_runs):
            backend.generate(self.sample_inputs, warmup_kwargs)

    # Image Diffusion warmup
    def warmup_image_diffusion(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Warming up backend for Image Diffusion")
        backend.call(self.sample_inputs, self.call_kwargs)
        warmup_kwargs = {**self.call_kwargs, **IMAGE_DIFFUSION_WARMUP_OVERRIDES}
        for _ in range(self.config.warmup_runs):
            backend.call(self.sample_inputs, warmup_kwargs)

    # Inference warmup
    def warmup_inference(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Warming up backend for Inference")
        warmup_kwargs = {**self.forward_kwargs}
        for _ in range(self.config.warmup_runs):
            backend.forward(self.sample_inputs, warmup_kwargs)

    # Text Generation energy tracking
    def run_text_generation_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Text Generation tracking")

        prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}

        self.reset_trackers()
        with self.track(task_name="prefill"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                backend.prefill(inputs, prefill_kwargs)

        if self.config.energy:
            prefill_energy = self.energy_tracker.get_energy()
            decode_energy = self.dataset_prefill_volume
            self.report.prefill.energy = prefill_energy
            self.report.prefill.efficiency = Efficiency.from_energy(
                prefill_energy, decode_energy, unit=PREFILL_EFFICIENCY_UNIT
            )
        if self.config.latency:
            prefill_latency = self.latency_tracker.get_latency()
            prefill_volume = self.dataset_prefill_volume
            self.report.prefill.latency = prefill_latency
            self.report.prefill.throughput = Throughput.from_latency(
                prefill_latency, prefill_volume, unit=PREFILL_THROUGHPUT_UNIT
            )
        if self.config.memory:
            self.report.prefill.memory = self.memory_tracker.get_max_memory()

        self.reset_trackers()
        with self.track(task_name="generate"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                backend.generate(inputs, self.config.generate_kwargs)

        if self.config.energy:
            generate_energy = self.energy_tracker.get_energy()
            decode_energy = generate_energy - prefill_energy
            decode_volume = self.dataset_decode_volume
            self.report.decode.energy = decode_energy
            self.report.decode.efficiency = Efficiency.from_energy(
                decode_energy, decode_volume, unit=DECODE_EFFICIENCY_UNIT
            )
        if self.config.latency:
            generate_latency = self.latency_tracker.get_latency()
            decode_latency = generate_latency - prefill_latency
            decode_volume = self.dataset_decode_volume
            self.report.decode.latency = decode_latency
            self.report.decode.throughput = Throughput.from_latency(
                decode_latency, decode_volume, unit=DECODE_THROUGHPUT_UNIT
            )
        if self.config.memory:
            self.report.decode.memory = self.memory_tracker.get_max_memory()

    # Image Diffusion tracking
    def run_image_diffusion_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Image Diffusion tracking")

        self.reset_trackers()
        with self.track(task_name="call"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                backend.call(inputs, self.call_kwargs)

        if self.config.energy:
            call_energy = self.energy_tracker.get_energy()
            call_volume = self.dataset_call_volume
            self.report.call.energy = call_energy
            self.report.call.efficiency = Efficiency.from_energy(call_energy, call_volume, unit=CALL_EFFICIENCY_UNIT)
        if self.config.latency:
            call_latency = self.latency_tracker.get_latency()
            call_volume = self.dataset_call_volume
            self.report.call.latency = call_latency
            self.report.call.throughput = Throughput.from_latency(call_latency, call_volume, unit=CALL_THROUGHPUT_UNIT)
        if self.config.memory:
            self.report.call.memory = self.memory_tracker.get_max_memory()

    # Inference tracking
    def run_inference_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Inference tracking")

        self.reset_trackers()
        with self.track(task_name="forward"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                backend.forward(inputs, self.forward_kwargs)

        if self.config.energy:
            forward_energy = self.energy_tracker.get_energy()
            forward_volume = self.dataset_forward_volume
            self.report.forward.energy = forward_energy
            self.report.forward.efficiency = Efficiency.from_energy(
                forward_energy, forward_volume, unit=FORWARD_EFFICIENCY_UNIT
            )
        if self.config.latency:
            forward_latency = self.latency_tracker.get_latency()
            forward_volume = self.dataset_forward_volume
            self.report.forward.latency = forward_latency
            self.report.forward.throughput = Throughput.from_latency(
                forward_latency, forward_volume, unit=FORWARD_THROUGHPUT_UNIT
            )
        if self.config.memory:
            self.report.forward.memory = self.memory_tracker.get_max_memory()

    @property
    def dataset_preprocess_volume(self) -> int:  # in terms of processed samples
        return self.config.num_samples

    @property
    def dataset_forward_volume(self) -> int:  # in terms of processed samples
        return self.config.num_samples

    @property
    def dataset_prefill_volume(self) -> int:  # in terms of processed samples
        return self.config.num_samples

    @property
    def dataset_decode_volume(self) -> int:  # in terms of generated tokens
        return (
            self.config.num_samples
            * self.config.generate_kwargs["num_beams"]  # at each beam stage there are num_beams tokens generated
            * (self.config.generate_kwargs["max_new_tokens"] - 1)  # 1 token is generated during prefill
        )

    @property
    def dataset_call_volume(self) -> int:  # in terms of generated images
        if self.task == "text-to-image":
            return self.config.num_samples * self.call_kwargs["num_images_per_prompt"]
        else:
            return self.config.num_samples
