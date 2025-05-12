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
        self.backend = backend

        if self.backend.config.task in TEXT_GENERATION_TASKS:
            self.logger.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_DEFAULT_KWARGS, **self.config.generate_kwargs}
            self.logger.info("\t+ Initializing Text Generation report")
            self.report = BenchmarkReport.from_list(
                targets=["load_dataset", "preprocess_dataset", "load_model", "prefill", "decode"]
            )
        elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
            self.logger.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.call_kwargs = {**IMAGE_DIFFUSION_DEFAULT_KWARGS, **self.config.call_kwargs}
            self.logger.info("\t+ Initializing Image Diffusion report")
            self.report = BenchmarkReport.from_list(
                targets=["load_dataset", "preprocess_dataset", "load_model", "call"]
            )
        else:
            self.logger.info("\t+ Initializing Inference report")
            self.report = BenchmarkReport.from_list(
                targets=["load_dataset", "preprocess_dataset", "load_model", "forward"]
            )

        if self.config.latency:
            self.logger.info("\t+ Initializing Latency tracker")
            self.latency_tracker = LatencyTracker(
                backend=self.backend.config.name,
                device=self.backend.config.device,
            )
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

        # we start with loading/preprocessing the dataset as it takes no vram
        self.run_dataset_loading_tracking()
        self.run_dataset_preprocessing_tracking()
        self.run_model_loading_tracking()

        if self.config.warmup_runs > 0:
            self.logger.info("\t+ Preparing sample inputs for warmup")
            self.sample_inputs = self.dataset[: self.config.input_shapes["batch_size"]]
            self.sample_inputs = self.backend.prepare_inputs(self.sample_inputs)

            if self.backend.config.task in TEXT_GENERATION_TASKS:
                self.warmup_text_generation()
            elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.warmup_image_diffusion()
            else:
                self.warmup_inference()

        if self.backend.config.task in TEXT_GENERATION_TASKS:
            self.run_text_generation_tracking()
        elif self.backend.config.task in IMAGE_DIFFUSION_TASKS:
            self.run_image_diffusion_tracking()
        else:
            self.run_inference_tracking()

        return self.report

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

    # Dataset loading tracking
    def run_dataset_loading_tracking(self):
        self.logger.info("\t+ Running dataset loading tracking")

        with self.track(task_name="load_dataset"):
            self.dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.dataset_split,
            )

        if self.config.energy:
            self.report.load_dataset.energy = self.energy_tracker.get_energy()
        if self.config.latency:
            self.report.load_dataset.latency = self.latency_tracker.get_latency()
        if self.config.memory:
            self.report.load_dataset.memory = self.memory_tracker.get_max_memory()

    # Dataset preprocessing tracking
    def run_dataset_preprocessing_tracking(self):
        self.logger.info("\t+ Running dataset preprocessing tracking")

        with self.track(task_name="preprocess_dataset"):
            self.dataset = TASKS_TO_PREPROCESSORS[self.backend.config.task](
                dataset=self.dataset,
                scenario_config=self.config,
                pretrained_config=self.backend.pretrained_config,
                pretrained_processor=self.backend.pretrained_processor,
            )

        if self.config.energy:
            preprocess_energy = self.energy_tracker.get_energy()

            self.report.preprocess_dataset.energy = preprocess_energy
            self.report.preprocess_dataset.efficiency = Efficiency.from_energy(
                preprocess_energy, self.dataset_preprocess_volume, unit=PREPROCESS_EFFICIENCY_UNIT
            )
        if self.config.latency:
            preprocess_latency = self.latency_tracker.get_latency()

            self.report.preprocess_dataset.latency = preprocess_latency
            self.report.preprocess_dataset.throughput = Throughput.from_latency(
                preprocess_latency, self.dataset_preprocess_volume, unit=PREPROCESS_THROUGHPUT_UNIT
            )
        if self.config.memory:
            self.report.preprocess_dataset.memory = self.memory_tracker.get_max_memory()

    # Model loading tracking
    def run_model_loading_tracking(self):
        self.logger.info("\t+ Running model loading energy tracking")

        with self.track(task_name="load_model"):
            self.backend.load()

        if self.config.latency:
            self.report.load_model.latency = self.latency_tracker.get_latency()
        if self.config.memory:
            self.report.load_model.memory = self.memory_tracker.get_max_memory()
        if self.config.energy:
            self.report.load_model.energy = self.energy_tracker.get_energy()

    # Text Generation warmup
    def warmup_text_generation(self):
        warmup_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_WARMUP_OVERRIDES}
        self.logger.info("\t+ Warming up backend for Text Generation")
        self.backend.generate(self.sample_inputs, self.config.generate_kwargs)
        for _ in range(self.config.warmup_runs):
            self.backend.generate(self.sample_inputs, warmup_kwargs)

    # Image Diffusion warmup
    def warmup_image_diffusion(self):
        warmup_kwargs = {**self.config.call_kwargs, **IMAGE_DIFFUSION_WARMUP_OVERRIDES}
        self.logger.info("\t+ Warming up backend for Image Diffusion")
        self.backend.call(self.sample_inputs, self.config.call_kwargs)
        for _ in range(self.config.warmup_runs):
            self.backend.call(self.sample_inputs, warmup_kwargs)

    # Inference warmup
    def warmup_inference(self):
        self.logger.info("\t+ Warming up backend for Inference")
        for _ in range(self.config.warmup_runs):
            self.backend.forward(self.sample_inputs, self.config.forward_kwargs)

    # Text Generation tracking
    def run_text_generation_tracking(self):
        self.logger.info("\t+ Running Text Generation tracking")

        prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}

        with self.track(task_name="prefill"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = self.backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                self.backend.prefill(inputs, prefill_kwargs)

        if self.config.energy:
            prefill_energy = self.energy_tracker.get_energy()

            self.report.prefill.energy = prefill_energy
            self.report.prefill.efficiency = Efficiency.from_energy(
                prefill_energy, self.dataset_prefill_volume, unit=PREFILL_EFFICIENCY_UNIT
            )
        if self.config.latency:
            prefill_latency = self.latency_tracker.get_latency()

            self.report.prefill.latency = prefill_latency
            self.report.prefill.throughput = Throughput.from_latency(
                prefill_latency, self.dataset_prefill_volume, unit=PREFILL_THROUGHPUT_UNIT
            )
        if self.config.memory:
            self.report.prefill.memory = self.memory_tracker.get_max_memory()

        with self.track(task_name="generate"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = self.backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                self.backend.generate(inputs, self.config.generate_kwargs)

        if self.config.energy:
            generate_energy = self.energy_tracker.get_energy()
            decode_energy = generate_energy - prefill_energy

            self.report.decode.energy = decode_energy
            self.report.decode.efficiency = Efficiency.from_energy(
                decode_energy, self.dataset_decode_volume, unit=DECODE_EFFICIENCY_UNIT
            )
        if self.config.latency:
            generate_latency = self.latency_tracker.get_latency()
            decode_latency = generate_latency - prefill_latency

            self.report.decode.latency = decode_latency
            self.report.decode.throughput = Throughput.from_latency(
                decode_latency, self.dataset_decode_volume, unit=DECODE_THROUGHPUT_UNIT
            )
        if self.config.memory:
            self.report.decode.memory = self.memory_tracker.get_max_memory()

    # Image Diffusion tracking
    def run_image_diffusion_tracking(self):
        self.logger.info("\t+ Running Image Diffusion tracking")

        with self.track(task_name="call"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = self.backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                self.backend.call(inputs, self.config.call_kwargs)

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
    def run_inference_tracking(self):
        self.logger.info("\t+ Running Inference tracking")

        with self.track(task_name="forward"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = self.backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                self.backend.forward(inputs, self.config.forward_kwargs)

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
        if self.backend.config.task == "text-to-image":
            return self.config.num_samples * self.config.call_kwargs["num_images_per_prompt"]
        else:
            return self.config.num_samples
